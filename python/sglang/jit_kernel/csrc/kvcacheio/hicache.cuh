#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace device {

namespace details {

template <typename T, uint32_t N>
struct LocalStorage {
  T data[N];
};

template <int kUnit>
inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

template <int kUnit>
using PackageType = decltype(get_mem_package<kUnit>());

SGL_DEVICE uint1 load_nc(const uint1* __restrict__ src) {
  uint32_t tmp;
  asm volatile("ld.global.L1::no_allocate.b32 %0,[%1];" : "=r"(tmp) : "l"(src));
  return uint1{tmp};
}

SGL_DEVICE uint2 load_nc(const uint2* __restrict__ src) {
  uint32_t tmp0, tmp1;
  asm volatile("ld.global.L1::no_allocate.v2.b32 {%0,%1},[%2];" : "=r"(tmp0), "=r"(tmp1) : "l"(src));
  return uint2{tmp0, tmp1};
}

SGL_DEVICE uint4 load_nc(const uint4* __restrict__ src) {
  uint32_t tmp0, tmp1, tmp2, tmp3;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3},[%4];"
               : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
               : "l"(src));
  return uint4{tmp0, tmp1, tmp2, tmp3};
}

SGL_DEVICE void store_nc(uint1* __restrict__ dst, const uint1& value) {
  uint32_t tmp = value.x;
  asm volatile("st.global.L1::no_allocate.b32 [%0],%1;" ::"l"(dst), "r"(tmp));
}

SGL_DEVICE void store_nc(uint2* __restrict__ dst, const uint2& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  asm volatile("st.global.L1::no_allocate.v2.b32 [%0],{%1,%2};" ::"l"(dst), "r"(tmp0), "r"(tmp1));
}

SGL_DEVICE void store_nc(uint4* __restrict__ dst, const uint4& value) {
  uint32_t tmp0 = value.x;
  uint32_t tmp1 = value.y;
  uint32_t tmp2 = value.z;
  uint32_t tmp3 = value.w;
  asm volatile(
      "st.global.L1::no_allocate.v4.b32 [%0],{%1,%2,%3,%4};" ::"l"(dst), "r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
}

}  // namespace details

template <int64_t kBytes, uint32_t kNumThreads>
SGL_DEVICE auto load_vec(const void* __restrict__ src) {
  static_assert(kBytes % 128 == 0, "kBytes must be multiple of 128 bytes");
  static_assert(128 % kNumThreads == 0, "kNumThreads must divide 128 bytes");
  constexpr uint32_t kLoopCount = kBytes / 128;
  using Package = details::PackageType<128 / kNumThreads>;
  using Storage = details::LocalStorage<Package, kLoopCount>;

  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kNumThreads;
  Storage vec;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    vec.data[i] = details::load_nc(&src_packed[j]);
  }

  return vec;
}

template <int64_t kBytes, uint32_t kNumThreads, typename Storage>
SGL_DEVICE void store_vec(void* __restrict__ dst, const Storage& vec) {
  using Package = std::decay_t<decltype(vec.data[0])>;
  constexpr uint32_t kBytesPerLoop = sizeof(Package) * kNumThreads;
  constexpr uint32_t kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0, "Invalid Storage configuration");

  const auto dst_packed = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kNumThreads;

#pragma unroll kLoopCount
  for (uint32_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kNumThreads + lane_id;
    details::store_nc(&dst_packed[j], vec.data[i]);
  }
}

}  // namespace device

namespace {

#define SGL_HICACHE_KERNEL __global__ __launch_bounds__(kBlockSize, 1)

struct HicacheKernelParams {
  void* __restrict__ k_cache_dst;
  void* __restrict__ v_cache_dst;
  const void* __restrict__ indices_dst;
  void* __restrict__ k_cache_src;
  void* __restrict__ v_cache_src;
  const void* __restrict__ indices_src;
  int64_t kv_cache_src_stride;
  int64_t kv_cache_dst_stride;
  uint32_t length;
  uint32_t num_layers = 0;  // only used in all_layer transfer
};

// [主线] 搬运当前 Transformer 层中由 indices_src/indices_dst 指定的
// 多个 token slot。每个逻辑 worker 负责一对 slot，并由一组 CUDA 线程
// 合作复制该 token 的完整 K；对于 Qwen2.5 GQA，还会继续复制完整 V。
//
// 模板参数：
// - T：indices_src/indices_dst 的元素类型，只能是 int32_t 或 int64_t；
// - kElementSize：一个 token 的单个 K（或 V）占用的字节数；
// - kUnroll：控制每个 worker 使用的线程数和每线程向量读写宽度；
// - kBlockQuota：本次搬运最多使用的 CUDA block 数；
// - kBlockSize：每个 CUDA block 的线程数，当前 JIT 固定传入 1024；
// - kIsMLA：是否使用没有独立 V buffer 的 MLA 路径。Qwen2.5 是 GQA，
//   因此当前主线实例化为 false，同时搬 K 和 V。
template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_per_layer(const __grid_constant__ HicacheKernelParams params) {
  // __grid_constant__ 表示 params 是本次 grid 只读的 kernel 启动参数。
  // SGL_HICACHE_KERNEL 展开为 __global__ 和 __launch_bounds__，所以该函数
  // 由 CPU 侧 run_one() 启动并在 GPU 上执行。
  using namespace device;

  // 编译期检查线程组织能够整除：
  // 1. 一个 block 必须能拆成整数个 warp；
  // 2. 一个 warp 必须能按 kUnroll 拆成整数大小的 worker。
  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  // 一个逻辑 worker 由 kNumThreads 个相邻 CUDA 线程组成，这些线程合作
  // 复制一个 token 的 kElementSize 字节。以 CUDA warp=32 为例：
  // unroll=1/2/4 时，每个 worker 分别使用 32/16/8 个线程。
  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;

  // 一个 block 中可同时容纳多少个逻辑 worker。例如 block_size=1024、
  // unroll=1 时，1024/32=32 个 worker，即一次处理 32 对 slot。
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;

  // 按最大 block quota 计算本轮所有逻辑 worker 的数量。它也是下面
  // grid-stride loop 的步长；长任务会由同一批 worker 循环处理后续 slot。
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  // 从只读参数结构中取出源/目标指针和元数据：
  // - k/v_cache_src、k/v_cache_dst：当前层 Host/GPU K/V buffer 的起始地址；
  // - indices_src、indices_dst：逐元素对应的 Host/GPU 物理 slot；
  // - src/dst_stride：相邻 token slot 行起点之间的字节数；
  // - length：本次需要处理的索引对数量；
  // - _：num_layers，单层 kernel 不使用。
  const auto& [
    k_cache_dst, v_cache_dst, indices_dst, // dst
    k_cache_src, v_cache_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, _ // metadata
  ] = params;

  // threadIdx.x / kNumThreads 将相邻线程划入同一个逻辑 worker。
  // work_id 是该 worker 首先负责的 indices 数组位置；同一 worker 内的
  // kNumThreads 个线程得到相同 work_id，并合作搬运同一行 K/V。
  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;

  // [主线] i 是 indices 数组的位置，不是 token id。若待搬 token 多于
  // kNumWorkers，worker 按固定步长继续处理 i+kNumWorkers，直到覆盖 length。
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    // 读取第 i 对物理 slot。例如 pos_src=37、pos_dst=105 表示：
    // Host KV slot 37 -> GPU KV slot 105。
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];

    // pointer::offset() 默认按字节偏移。源和目标使用各自 stride，因此
    // page_first Host 布局和 layer_first GPU 布局的行间距可以不同。
    const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
    const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);

    // kNumThreads 个线程合作读取该 token 的完整 kElementSize 字节 K，
    // 暂存到各线程的局部向量，再合作写入目标 GPU slot。
    const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
    store_vec<kElementSize, kNumThreads>(dst_k, vec_k);

    // [主线] Qwen2.5 GQA 的 kIsMLA=false，因此除 K 外还要复制独立的 V。
    // if constexpr 在编译期决定是否保留该代码；MLA 实例不会产生 V 搬运指令。
    if constexpr (!kIsMLA) {
      const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
      store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
    }
  }
}

template <
    typename T,
    int64_t kElementSize,
    uint32_t kUnroll,
    uint32_t kBlockQuota,
    uint32_t kBlockSize,
    bool kIsMLA = false>
SGL_HICACHE_KERNEL void hicache_transfer_all_layer(const __grid_constant__ HicacheKernelParams params) {
  using namespace device;
  using src_ptr_t = const void*;
  using dst_ptr_t = void*;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kWarpThreads % kUnroll == 0);

  constexpr uint32_t kNumThreads = kWarpThreads / kUnroll;
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kNumThreads;
  constexpr uint32_t kNumWorkers = kWorkersPerBlock * kBlockQuota;

  const auto& [
    k_ptr_dst, v_ptr_dst, indices_dst, // dst
    k_ptr_src, v_ptr_src, indices_src, // src
    kv_cache_src_stride, kv_cache_dst_stride, length, num_layers // metadata
  ] = params;

  const uint32_t work_id = blockIdx.x * kWorkersPerBlock + threadIdx.x / kNumThreads;
  for (uint32_t i = work_id; i < length; i += kNumWorkers) {
    const auto pos_src = static_cast<const T*>(indices_src)[i];
    const auto pos_dst = static_cast<const T*>(indices_dst)[i];
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
      const auto k_cache_src = static_cast<const src_ptr_t*>(k_ptr_src)[layer];
      const auto k_cache_dst = static_cast<const dst_ptr_t*>(k_ptr_dst)[layer];
      const auto src_k = pointer::offset(k_cache_src, pos_src * kv_cache_src_stride);
      const auto dst_k = pointer::offset(k_cache_dst, pos_dst * kv_cache_dst_stride);
      const auto vec_k = load_vec<kElementSize, kNumThreads>(src_k);
      store_vec<kElementSize, kNumThreads>(dst_k, vec_k);
      if constexpr (!kIsMLA) {
        const auto v_cache_src = static_cast<const src_ptr_t*>(v_ptr_src)[layer];
        const auto v_cache_dst = static_cast<const dst_ptr_t*>(v_ptr_dst)[layer];
        const auto src_v = pointer::offset(v_cache_src, pos_src * kv_cache_src_stride);
        const auto dst_v = pointer::offset(v_cache_dst, pos_dst * kv_cache_dst_stride);
        const auto vec_v = load_vec<kElementSize, kNumThreads>(src_v);
        store_vec<kElementSize, kNumThreads>(dst_v, vec_v);
      }
    }
  }
}

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize>
struct HiCacheKernel {
  template <typename T>
  static constexpr auto kernel_one = hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;
  template <typename T>
  static constexpr auto kernel_all = hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize>;
  template <typename T>
  static constexpr auto kernel_one_mla =
      hicache_transfer_per_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize, true>;
  template <typename T>
  static constexpr auto kernel_all_mla =
      hicache_transfer_all_layer<T, kElementSize, kUnroll, kBlockQuota, kBlockSize, true>;

  static void run_one(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src kv stride"};
    auto M = SymbolicSize{"dst kv stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_src)
        .verify(v_cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(k_cache_dst)
        .verify(v_cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel: cache dimension mismatch.");

    const auto k_cache_dst_ptr = k_cache_dst.data_ptr();
    const auto v_cache_dst_ptr = v_cache_dst.data_ptr();
    const auto k_cache_src_ptr = k_cache_src.data_ptr();
    const auto v_cache_src_ptr = v_cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto kv_cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto kv_cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_cache_src_stride,
        .kv_cache_dst_stride = kv_cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one<int32_t> : kernel_one<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all(
      const tvm::ffi::TensorView k_ptr_dst,
      const tvm::ffi::TensorView v_ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t kv_src_stride_bytes,
      const int64_t kv_dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(k_ptr_src)
        .verify(v_ptr_src)
        .verify(k_ptr_dst)
        .verify(v_ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    // verify dimension match
    const auto k_cache_dst_ptr = k_ptr_dst.data_ptr();
    const auto v_cache_dst_ptr = v_ptr_dst.data_ptr();
    const auto k_cache_src_ptr = k_ptr_src.data_ptr();
    const auto v_cache_src_ptr = v_ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = k_cache_dst_ptr,
        .v_cache_dst = v_cache_dst_ptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = k_cache_src_ptr,
        .v_cache_src = v_cache_src_ptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = kv_src_stride_bytes,
        .kv_cache_dst_stride = kv_dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all<int32_t> : kernel_all<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_one_mla(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView cache_src,
      const tvm::ffi::TensorView indices_src) {
    using namespace host;

    auto D = SymbolicSize{"head dimension"};
    auto N = SymbolicSize{"src stride"};
    auto M = SymbolicSize{"dst stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    TensorMatcher({-1, D})  //
        .with_strides({N, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(cache_src);
    TensorMatcher({-1, D})  //
        .with_strides({M, 1})
        .with_dtype(cache_dtype)
        .with_device<kDLCUDA, kDLCUDAHost, kDLCPU>()
        .verify(cache_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLCUDA>(indices_device)
        .verify(indices_src)
        .verify(indices_dst);

    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto element_bytes = D.unwrap() * dtype_size;
    RuntimeCheck(kElementSize == element_bytes, "HicacheKernel MLA: cache dimension mismatch.");

    const auto cache_dst_ptr = cache_dst.data_ptr();
    const auto cache_src_ptr = cache_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto cache_src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    const auto cache_dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size);
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    const auto device = indices_device.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = cache_dst_ptr,
        .v_cache_dst = nullptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = cache_src_ptr,
        .v_cache_src = nullptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = cache_src_stride,
        .kv_cache_dst_stride = cache_dst_stride,
        .length = length,
    };
    const auto kernel = use_int32 ? kernel_one_mla<int32_t> : kernel_one_mla<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }

  static void run_all_mla(
      const tvm::ffi::TensorView ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t src_stride_bytes,
      const int64_t dst_stride_bytes) {
    using namespace host;

    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(ptr_src)
        .verify(ptr_dst);
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices_src)
        .verify(indices_dst);

    const auto cache_dst_ptr = ptr_dst.data_ptr();
    const auto cache_src_ptr = ptr_src.data_ptr();
    const auto indices_dst_ptr = indices_dst.data_ptr();
    const auto indices_src_ptr = indices_src.data_ptr();
    const auto length = static_cast<uint32_t>(L.unwrap());
    const auto use_int32 = dtype_.unwrap().bits == 32;
    const auto device = device_.unwrap();

    constexpr auto kWorkersPerBlock = kBlockSize / (device::kWarpThreads / kUnroll);
    const auto num_blocks = std::min(div_ceil(length, kWorkersPerBlock), kBlockQuota);
    const auto params = HicacheKernelParams{
        .k_cache_dst = cache_dst_ptr,
        .v_cache_dst = nullptr,
        .indices_dst = indices_dst_ptr,
        .k_cache_src = cache_src_ptr,
        .v_cache_src = nullptr,
        .indices_src = indices_src_ptr,
        .kv_cache_src_stride = src_stride_bytes,
        .kv_cache_dst_stride = dst_stride_bytes,
        .length = length,
        .num_layers = static_cast<uint32_t>(N.unwrap()),
    };
    const auto kernel = use_int32 ? kernel_all_mla<int32_t> : kernel_all_mla<int64_t>;
    LaunchKernel(num_blocks, kBlockSize, device)(kernel, params);
  }
};

#undef SGL_HICACHE_KERNEL

}  // namespace
