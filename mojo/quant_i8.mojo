# BF16/F32 → symmetric INT8 per block (B=64, zp=0, scale=amax/127).
# Mojo twin of cuda/quant_i8.cu. Offline pin convert. Not train.
#
# Same semantics as the CUDA kernels: float32 amax, scale = amax/127 (1.0 for
# an all-zero block), q = clamp(round_half_even(w / scale), -127, 127).
#
# Built as a Python extension module (see build.sh). Every entry point takes raw
# buffer addresses so mojo_quant.py can pass ctypes buffers, mirroring the
# libquant_i8.so ABI used by gpu_quant.py. Return codes match quant_i8.cu:
# 0 ok, 2 bad args, 6 no accelerator.
from std.math import ceildiv
from std.memory import bitcast, Pointer, stack_allocation
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import has_accelerator
from max.algorithm import parallelize
from max.gpu import thread_idx, block_idx, barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace

comptime BLOCK = 64


@always_inline
def bf16_to_f32(h: UInt16) -> Float32:
    return bitcast[DType.float32](UInt32(h) << 16)


@always_inline
def quant_one(w: Float32, s: Float32) -> Int8:
    var v = round(w / s)
    if v > 127.0:
        v = 127.0
    if v < -127.0:
        v = -127.0
    return Int8(Int(v))


@always_inline
def scale_of(amax: Float32) -> Float32:
    if amax > 0.0:
        return amax / 127.0
    return 1.0


# ---------------------------------------------------------------- CPU path
# One work item per 64-element block; blocks are independent, so this scales
# across cores with no synchronization.


def quant_cpu[bf16: Bool](in_addr: Int, n: Int, q_addr: Int, s_addr: Int):
    var q = Pointer[Int8, MutAnyOrigin](unsafe_from_address=q_addr)
    var sc = Pointer[Float32, MutAnyOrigin](unsafe_from_address=s_addr)
    var pb = Pointer[UInt16, MutAnyOrigin](unsafe_from_address=in_addr)
    var pf = Pointer[Float32, MutAnyOrigin](unsafe_from_address=in_addr)

    def load(i: Int) {var pb, var pf} -> Float32:
        comptime if bf16:
            return bf16_to_f32(pb[unsafe_offset=i])
        else:
            return pf[unsafe_offset=i]

    def work(blk: Int) {var q, var sc, var load, var n} -> None:
        var start = blk * BLOCK
        var end = min(start + BLOCK, n)
        var amax = Float32(0.0)
        for i in range(start, end):
            amax = max(amax, abs(load(i)))
        var s = scale_of(amax)
        sc[unsafe_offset=blk] = s
        for i in range(start, end):
            q[unsafe_offset=i] = quant_one(load(i), s)

    parallelize(work, ceildiv(n, BLOCK))


# ---------------------------------------------------------------- GPU path
# One thread block per quant block, 64 threads, shared-memory amax reduction.
# Line-for-line with k_quant_bf16 / k_quant_f32 in cuda/quant_i8.cu so the two
# can be twin-compared on the RTX 3060.


def k_quant[
    elem: DType
](
    inp: Pointer[Scalar[elem], MutAnyOrigin],
    n: Int64,
    q: Pointer[Int8, MutAnyOrigin],
    sc: Pointer[Float32, MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var i = blk * BLOCK + t
    var sh = stack_allocation[BLOCK, Float32, address_space = AddressSpace.SHARED]()
    var w = Float32(0.0)
    var in_range = i < Int(n)
    if in_range:
        comptime if elem == DType.uint16:
            w = bf16_to_f32(UInt16(inp[unsafe_offset=i]))
        else:
            w = Float32(inp[unsafe_offset=i])
    sh[unsafe_offset=t] = abs(w)
    barrier()
    var off = BLOCK // 2
    while off > 0:
        if t < off:
            sh[unsafe_offset=t] = max(sh[unsafe_offset=t], sh[unsafe_offset=t + off])
        barrier()
        off //= 2
    var s = scale_of(sh[unsafe_offset=0])
    if t == 0:
        sc[unsafe_offset=blk] = s
    if in_range:
        q[unsafe_offset=i] = quant_one(w, s)


def quant_gpu[
    bf16: Bool
](ctx: DeviceContext, in_addr: Int, n: Int, q_addr: Int, s_addr: Int) raises:
    comptime elem = DType.uint16 if bf16 else DType.float32
    var nblk = ceildiv(n, BLOCK)
    var h_in = Pointer[Scalar[elem], MutAnyOrigin](unsafe_from_address=in_addr)
    var h_q = Pointer[Int8, MutAnyOrigin](unsafe_from_address=q_addr)
    var h_s = Pointer[Float32, MutAnyOrigin](unsafe_from_address=s_addr)
    var d_in = ctx.enqueue_create_buffer[elem](n)
    var d_q = ctx.enqueue_create_buffer[DType.int8](n)
    var d_s = ctx.enqueue_create_buffer[DType.float32](nblk)
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=h_in)
    ctx.enqueue_function[k_quant[elem]](
        d_in.unsafe_ptr(),
        Int64(n),
        d_q.unsafe_ptr(),
        d_s.unsafe_ptr(),
        grid_dim=nblk,
        block_dim=BLOCK,
    )
    ctx.enqueue_copy(dst_ptr=h_q, src_buf=d_q)
    ctx.enqueue_copy(dst_ptr=h_s, src_buf=d_s)
    ctx.synchronize()


# ---------------------------------------------------------------- Python ABI


def _args(
    in_addr: PythonObject, n: PythonObject, q_addr: PythonObject, s_addr: PythonObject
) raises -> Tuple[Int, Int, Int, Int]:
    return (Int(py=in_addr), Int(py=n), Int(py=q_addr), Int(py=s_addr))


def quant_bf16_i8_cpu(
    in_addr: PythonObject, n: PythonObject, q_addr: PythonObject, s_addr: PythonObject
) raises -> PythonObject:
    var a = _args(in_addr, n, q_addr, s_addr)
    if a[1] < 1:
        return PythonObject(2)
    quant_cpu[True](a[0], a[1], a[2], a[3])
    return PythonObject(0)


def quant_f32_i8_cpu(
    in_addr: PythonObject, n: PythonObject, q_addr: PythonObject, s_addr: PythonObject
) raises -> PythonObject:
    var a = _args(in_addr, n, q_addr, s_addr)
    if a[1] < 1:
        return PythonObject(2)
    quant_cpu[False](a[0], a[1], a[2], a[3])
    return PythonObject(0)


def _run_gpu[bf16: Bool](a: Tuple[Int, Int, Int, Int]) -> PythonObject:
    # has_accelerator() is answered at compile time (--target-accelerator), so
    # probe the device at run time: no driver/GPU -> 6, as quant_i8.cu callers expect.
    if not has_accelerator():
        return PythonObject(6)
    try:
        var ctx = DeviceContext()
        try:
            quant_gpu[bf16](ctx, a[0], a[1], a[2], a[3])
        except:
            return PythonObject(5)
    except:
        return PythonObject(6)
    return PythonObject(0)


def quant_bf16_i8_gpu(
    in_addr: PythonObject, n: PythonObject, q_addr: PythonObject, s_addr: PythonObject
) raises -> PythonObject:
    var a = _args(in_addr, n, q_addr, s_addr)
    if a[1] < 1:
        return PythonObject(2)
    return _run_gpu[True](a)


def quant_f32_i8_gpu(
    in_addr: PythonObject, n: PythonObject, q_addr: PythonObject, s_addr: PythonObject
) raises -> PythonObject:
    var a = _args(in_addr, n, q_addr, s_addr)
    if a[1] < 1:
        return PythonObject(2)
    return _run_gpu[False](a)


@export
def PyInit_quant_i8_mojo() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("quant_i8_mojo")
        m.def_function[quant_bf16_i8_cpu]("quant_bf16_i8_cpu")
        m.def_function[quant_f32_i8_cpu]("quant_f32_i8_cpu")
        m.def_function[quant_bf16_i8_gpu]("quant_bf16_i8_gpu")
        m.def_function[quant_f32_i8_gpu]("quant_f32_i8_gpu")
        return m.finalize()
    except e:
        abort(String("quant_i8_mojo init failed: ", e))
