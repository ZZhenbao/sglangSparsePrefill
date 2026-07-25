from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

DEFAULT_BLOCK_QUOTA = 2


# [主线] 为当前 KV 元素大小和并行参数创建（或复用）HiCache JIT 模块。
# cache_once 以函数参数为 key 缓存返回的 Module；同一进程内再次请求完全相同的
# (element_size, unroll, block_quota) 时，不会重复进入函数或重新加载模块。
@cache_once
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    # 把 Python 参数转换成可以直接拼入 C++ 模板实例的字符串列表。
    # 例如 (2048, 1, 2) 会连同固定 block size 生成：
    #   HiCacheKernel<2048, 1, 2, 1024>
    #
    # 四个模板参数依次表示：
    # - element_size：一个 token 的单个 K（或 V）占用的字节数；
    # - unroll：控制每个 worker 的线程数和每线程向量读写宽度；
    # - block_quota：本次搬运最多启动的 CUDA block 数；
    # - 1024：每个 CUDA block 使用的线程数（kBlockSize）。
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        1024,  # num_threads, can be tuned for performance
    )

    # 编译或加载名为 hicache 的 CUDA JIT 模块。load_jit() 会把模板参数、
    # CUDA 架构、TVM-FFI ABI 和源文件内容 hash 纳入缓存标识：
    # - 磁盘已有匹配的 .so 时直接加载；
    # - 没有时编译 hicache.cuh 并生成 .so；
    # - 编译/加载失败会向上抛出，由 can_use_hicache_jit_kernel() 捕获，
    #   使上层回退到通用 KV 搬运 kernel。
    return load_jit(
        "hicache",
        *args,
        # 源文件相对于 python/sglang/jit_kernel/csrc/。其中定义了
        # HiCacheKernel 模板、run_one/run_all 入口和实际 CUDA kernel。
        cuda_files=[
            "kvcacheio/hicache.cuh",
        ],
        # 为指定的 C++ 静态函数生成 TVM-FFI Python 导出方法。
        # 每个二元组格式为：
        #   ("Python 侧方法名", "实例化后的 C++ 函数地址")
        #
        # [主线] 当前 L2 Host -> L1 GPU 逐层恢复调用 launch_one，
        # 它绑定到 HiCacheKernel<...>::run_one。
        # launch_all 用于一次处理所有层；带 _mla 的两个入口服务 MLA KV。
        # 当前 Qwen2.5 是 GQA，但使用标准 K/V 分离缓存和 MHA/GQA 通用的
        # MHATokenToKVPoolHost，因此关注 launch_one，MLA 入口可以先跳过。
        cuda_wrappers=[
            ("launch_one", f"&HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"&HiCacheKernel<{args}>::run_all"),
            ("launch_one_mla", f"&HiCacheKernel<{args}>::run_one_mla"),
            ("launch_all_mla", f"&HiCacheKernel<{args}>::run_all_mla"),
        ],
    )


@cache_once
def _jit_hicache_staged_module(
    *, element_size: int, unroll: int, block_quota: int
) -> Module:
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        1024,  # num_threads, kept for template compatibility
    )
    return load_jit(
        "hicache_staged",
        *args,
        cuda_files=[
            "kvcacheio/staged_write_back.cuh",
        ],
        cuda_wrappers=[
            (
                "launch_all_lf_pf_staged",
                f"&HiCacheStagedWriteBackKernel<{args}>::run_all_lf_pf_staged",
            ),
            (
                "launch_all_mla_lf_pf_staged",
                f"&HiCacheStagedWriteBackKernel<{args}>::run_all_mla_lf_pf_staged",
            ),
        ],
    )


def can_use_hicache_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    logger = logging.getLogger(__name__)
    if element_size % 128 != 0:
        logger.warning(f"Unsupported {element_size = } for JIT HiCache kernel")
        return False
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT HiCache kernel: {e}")
        return False


def can_use_write_back_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    logger = logging.getLogger(__name__)
    if element_size % 16 != 0:
        logger.warning(f"Unsupported {element_size = } for staged JIT HiCache kernel")
        return False
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_staged_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load staged JIT HiCache kernel: {e}")
        return False


def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4

    if element_size <= 1024:
        return 2

    # fallback: no unroll
    return 1


# [主线] 单层 HiCache Host -> GPU 搬运的 Python/JIT 包装函数。
# debug_kernel_api 只用于按环境变量记录 kernel 调用信息；默认关闭时直接返回
# 原函数，不参与实际的数据搬运和同步。
@debug_kernel_api
def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    # 函数整体作用：把某一 Transformer 层中由 indices_src 指定的 K/V，
    # 搬到 indices_dst 指定的目标 slot。对每个位置 j，逻辑操作为：
    #   k_cache_dst[indices_dst[j]] = k_cache_src[indices_src[j]]
    #   v_cache_dst[indices_dst[j]] = v_cache_src[indices_src[j]]
    #
    # 参数含义：
    # - k/v_cache_src：当前层位于 L2 Host KV Pool 的 K/V；
    # - k/v_cache_dst：当前层位于 L1 GPU KV Pool 的 K/V；
    # - indices_src：按 prefix token 顺序排列的 Host 源 slot；
    # - indices_dst：与 indices_src 一一对应的 GPU 目标 slot；
    # - element_dim：一个 token 的单个 K 或 V 含有多少个标量元素；
    # - unroll：JIT kernel 的线程分组和向量化展开参数；
    # - block_quota：本次搬运最多使用的 CUDA block 数，用于降低其对
    #   同时运行的 forward 计算的干扰。
    #
    # 调用方位于 load_stream 上下文中，因此最后 launch_one() 提交的 kernel
    # 进入 load_stream；本函数不会在 CPU 侧等待搬运完成。

    # [主线] 调用方通常显式传入 head_num * head_dim。没有传入时，使用
    # GPU 目标 K buffer 的最后一维作为单个 token 的元素数。
    element_dim = element_dim or k_cache_dst.size(-1)

    # 把不同 Host 内存布局提供的当前层 tensor 统一重解释为二维视图：
    #   [token_slot, element_dim]
    # 这里只创建 view，不复制 K/V 数据；源和目标的行 stride 可以不同，
    # 底层 run_one() 会分别读取 src/dst stride。
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)

    # 一个 token 的单个 K（或单个 V）占用的字节数。该值不是本次总搬运量，
    # 而是 JIT 编译模板的 element size；总数据量还要乘索引数量以及 K/V 两份。
    element_size = element_dim * k_cache_dst.element_size()

    # 默认最多使用 DEFAULT_BLOCK_QUOTA（当前为 2）个 CUDA block。
    # 较小的 quota 限制搬运 kernel 占用的 GPU 计算资源，为并行的 forward
    # 留出资源；也可以由调用方显式覆盖以做性能调优。
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA

    # 根据每个 K/V token 的字节数选择默认向量化展开参数：
    # <=512 B 使用 4，<=1024 B 使用 2，更大时使用 1。
    # 它决定每个 worker 的线程数和每线程读写宽度，并作为 CUDA 模板参数，
    # 不会在每个 token 搬运时执行运行时分支。
    unroll = unroll or _default_unroll(element_size)

    # [主线] 获取针对 (element_size, unroll, block_quota) 编译的 JIT CUDA
    # 模块。同一参数组合会复用缓存；首次不可用或编译失败时，上层
    # can_use_hicache_jit_kernel() 会让调用路径回退到通用 kernel。
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )

    # [主线] 真正向当前 load_stream 提交单层 K/V 搬运 kernel。
    # C++/CUDA run_one() 根据每对 indices_src[j]/indices_dst[j] 计算源、
    # 目标地址，同时复制 K 和 V；本调用只负责提交，不执行 CPU 同步。
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


@debug_kernel_api
def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    if element_size is None:  # assume both contiguous
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def transfer_hicache_one_layer_mla(
    cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = element_dim or cache_dst.size(-1)
    cache_src = cache_src.view(-1, element_dim)
    cache_dst = cache_dst.view(-1, element_dim)
    element_size = element_dim * cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one_mla(
        cache_dst,
        indices_dst,
        cache_src,
        indices_src,
    )


def transfer_hicache_all_layer_mla(
    ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    cache_src_stride_bytes: int,
    cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    if element_size is None:
        assert cache_dst_stride_bytes == cache_src_stride_bytes
        element_size = cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all_mla(
        ptr_dst,
        indices_dst,
        ptr_src,
        indices_src,
        cache_src_stride_bytes,
        cache_dst_stride_bytes,
    )


@debug_kernel_api
def transfer_hicache_all_layer_staged_lf_pf(
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    staging_k: torch.Tensor,
    staging_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    *,
    page_size: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = staging_k[0, 0].numel()
    element_size = element_size or (element_dim * staging_k.element_size())
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    src_page_indices = src_indices[::page_size].contiguous()
    module = _jit_hicache_staged_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    staging_page_capacity = staging_k.shape[0] // page_size
    staging_k = staging_k.view(staging_k.shape[0], staging_k.shape[1], -1)
    staging_v = staging_v.view(staging_v.shape[0], staging_v.shape[1], -1)
    dst_k = dst_k.view(dst_k.shape[0], dst_k.shape[1], -1)
    dst_v = dst_v.view(dst_v.shape[0], dst_v.shape[1], -1)
    for page_begin in range(0, src_page_indices.numel(), staging_page_capacity):
        chunk_pages = min(staging_page_capacity, src_page_indices.numel() - page_begin)
        chunk_tokens = chunk_pages * page_size
        module.launch_all_lf_pf_staged(
            dst_k,
            dst_v,
            dst_indices[
                page_begin * page_size : (page_begin + chunk_pages) * page_size
            ],
            staging_k[:chunk_tokens],
            staging_v[:chunk_tokens],
            src_page_indices[page_begin : page_begin + chunk_pages],
            k_ptr_src,
            v_ptr_src,
            page_size,
        )


@debug_kernel_api
def transfer_hicache_all_layer_mla_staged_lf_pf(
    ptr_src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    staging: torch.Tensor,
    dst: torch.Tensor,
    *,
    page_size: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = staging[0, 0].numel()
    element_size = element_size or (element_dim * staging.element_size())
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    src_page_indices = src_indices[::page_size].contiguous()
    module = _jit_hicache_staged_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    staging_page_capacity = staging.shape[0] // page_size
    staging = staging.view(staging.shape[0], staging.shape[1], -1)
    dst = dst.view(dst.shape[0], dst.shape[1], -1)
    for page_begin in range(0, src_page_indices.numel(), staging_page_capacity):
        chunk_pages = min(staging_page_capacity, src_page_indices.numel() - page_begin)
        chunk_tokens = chunk_pages * page_size
        module.launch_all_mla_lf_pf_staged(
            dst,
            dst_indices[
                page_begin * page_size : (page_begin + chunk_pages) * page_size
            ],
            staging[:chunk_tokens],
            src_page_indices[page_begin : page_begin + chunk_pages],
            ptr_src,
            page_size,
        )
