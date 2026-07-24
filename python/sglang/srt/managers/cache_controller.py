from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.pool_host import HostKVCache

from sglang.srt.distributed import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        self.storage_stop_event = threading.Event()

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _stop_storage_threads(self):
        """Stop storage prefetch/backup threads and drain internal queues.

        Caller should ensure no in-flight requests.
        """
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear storage_stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Best-effort joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(self.prefetch_thread)
        if hasattr(self, "backup_thread"):
            threads.append(self.backup_thread)
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(self.prefetch_io_aux_thread)

        for t in threads:
            try:
                t.join(timeout=10)
            except Exception:
                pass

        alive = [t for t in threads if getattr(t, "is_alive", lambda: False)()]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s",
                [getattr(t, "name", repr(t)) for t in alive],
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            # Budget speculative prefetch at half the host pool, leaving the rest for the write-back staging path.
            self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)
            # tracking the number of tokens locked in prefetching, updated by the main scheduler thread
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_register_draft_with_storage()

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_attention_tp_rank()
            self.tp_size = get_attention_tp_size()
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_rank = 0

        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_size = get_pipeline_model_parallel_world_size()

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
        )

    def reset(self):
        self.storage_stop_event.set()

        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_thread.join()
            self.backup_thread.join()
            self.prefetch_queue.queue.clear()
            self.backup_queue.queue.clear()
            self.prefetch_revoke_queue.queue.clear()
            self.ack_backup_queue.queue.clear()
            self.host_mem_release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

        self.storage_stop_event.clear()

        if self.enable_storage:
            self.prefetch_thread = threading.Thread(
                target=self.prefetch_thread_func, daemon=True
            )
            self.backup_thread = threading.Thread(
                target=self.backup_thread_func, daemon=True
            )
            self.prefetch_thread.start()
            self.backup_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        # Page-first write-back JIT kernels can keep destination host indices on CPU.
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices, device_indices = op.host_indices, op.device_indices
        else:
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        # 函数整体作用：为一批 L2 Host KV 预留等量的 L1 GPU slot，并把
        # Host slot -> GPU slot 映射登记到 load_queue。该函数由 scheduler
        # 线程同步调用，只进行资源分配和任务入队；实际逐层 Host -> GPU copy
        # 由后续 start_loading() 在独立 load_stream 上启动。
        #
        # 参数含义：
        # - host_indices：按 prefix token 顺序排列的 L2 Host 物理 slot；
        # - node_id：本次 load-back 最深 Radix 节点的 id，随 ack 返回，供
        #   loading_check() 查找 ongoing_load_back 并释放在途锁；
        # - priority：可选任务优先级，未传入时 CacheOperation 使用创建顺序。

        # [主线] 按待恢复的 L2 token 数，从 L1 KV allocator 分配相同数量的
        # GPU 物理 slot。返回 tensor 中的每个位置与 host_indices 一一对应。
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))

        # 没有足够的 L1 slot 时不创建 CacheOperation，也不修改 load_queue。
        # 调用方 HiRadixCache.load_back() 可以先淘汰其他 L1 KV 后再重试。
        if device_indices is None:
            return None

        # 保存本次 L2 -> L1 的源/目标 slot 映射和追踪 node_id。
        # start_loading() 会合并当前 load_queue 中的操作，逐层执行实际 KV copy，
        # 并在完成事件的 ack_list 中带回各操作的 node_id。
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )

        # 返回已经预留的 L1 slot 索引，供 Radix 节点 value 和
        # req.prefix_indices 立即登记。此时 slot 中的 KV 数据尚不保证就绪。
        return device_indices

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # 函数整体作用：按照 io_backend 的调用要求，调整 L2 源 slot 索引和
        # L1 目标 slot 索引所在的设备及排列顺序。
        #
        # 本函数处理的是索引 tensor，不会搬运 KV 数据，也不会重新分配 L1
        # slot。输入和输出始终保持逐元素映射：
        #     host_indices[i] 中的 KV -> device_indices[i] 对应的 GPU slot

        if self.io_backend == "kernel":
            # CUDA/ROCm kernel 直接读取索引，因此 host_indices 也需要位于
            # accelerator 上；device_indices 由 L1 allocator 返回，已经位于
            # kernel 所需设备。
            if not host_indices.is_cuda:
                # non_blocking=True 允许索引复制异步提交。start_loading() 随后
                # 记录 start_event，并让 load_stream 等待该索引复制完成。
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices

        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                # direct 后端在 CPU 侧读取索引，将 L1 目标索引移到 CPU。
                device_indices = device_indices.cpu()

                # 按 L2 Host slot 升序排列源索引。idx 是原数组的排序置换，
                # 使用相同置换重排 device_indices，保证每个 L2 -> L1 映射
                # 在排序后仍然一一对应。
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)

            elif self.mem_pool_host.layout == "page_first_direct":
                # page_first_direct 保留原有 Host 索引顺序，只确保 direct
                # 搬运实现可以从 CPU 读取 L1 目标 slot 索引。
                return host_indices, device_indices.cpu()

            else:
                # direct 后端目前只支持上述两种 Host KV 内存布局。
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )

        elif self.io_backend == "kernel_ascend":
            # Ascend 搬运接口在 CPU 侧读取目标 slot 索引。
            return host_indices, device_indices.cpu()

        else:
            # 尽早拒绝未实现的后端，避免以错误的索引设备启动 KV 搬运。
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        # 函数整体作用：消费此前 load() 登记在 load_queue 中的 L2 -> L1
        # 搬运任务，在独立 load_stream 上逐层提交 Host KV -> GPU KV copy，
        # 并返回本轮事件组的索引。scheduler 会把该索引交给 forward，使
        # forward 在读取第 i 层 KV 前等待对应的第 i 层复制完成。
        #
        # 本函数只异步提交搬运，不在 CPU 侧等待全部 KV 复制完成；最后一层
        # 完成后产生的 ack 由后续 loading_check() 收割并释放 load-back 在途锁。

        # 本轮没有 L2 -> L1 任务，无需建立同步关系。-1 会使 forward 跳过
        # HiCache 的逐层等待。
        if len(self.load_queue) == 0:
            return -1

        # [主线] 从循环复用的事件组中取得一个 producer_id。本轮每层复制的
        # 完成事件都会记录在该事件组中；返回后它也会成为 batch 的
        # hicache_consumer_index。
        producer_id = self.layer_done_counter.update_producer()

        # 将本轮各请求分别登记的 CacheOperation 合并成一组连续索引：
        # host_indices 是 L2 源 slot，device_indices 是已提前分配的 L1 目标
        # slot，node_ids 保留每个 load-back 的完成确认 id。
        # batch_size=1 时通常只有一个操作，merge_ops() 会直接返回它。
        op = CacheOperation.merge_ops(self.load_queue)

        # 根据 io_backend 调整索引所在设备及排列方式，供底层 copy 实现使用；
        # 源/目标 slot 的对应关系保持一致。
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )

        # 任务已由局部变量 op 接管，清空队列，避免下一轮重复提交。
        self.load_queue.clear()

        # 当前计算 stream 记录起点事件。若 move_indices() 在当前 stream 上
        # 异步移动过索引，load_stream 必须先等待该事件再读取这些索引。
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        # [主线] 切换到独立 load_stream。with 块只负责按顺序提交 GPU 工作，
        # 离开该块不代表所有 Host -> GPU copy 已经完成。
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                # 把所有 host_indices 对应的第 i 层 KV 写入已分配的
                # device_indices。按层搬运允许 forward 在前几层就绪后开始计算，
                # 与后续层的 KV 搬运重叠。
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )

                # [当前可跳过] 推测解码启用 draft KV pool 时，同步恢复 draft
                # 模型对应层的 KV；当前未启用 speculative decoding 时不进入。
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )

                # 在 load_stream 上记录“第 i 层复制已完成”事件。forward 使用
                # producer_id 对应的 consumer event，在读取该层 GPU slot 前等待。
                producer_event.complete(i)

            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            # 告知 PyTorch caching allocator：这些索引 tensor 仍被 load_stream
            # 异步使用，在该 stream 完成前不能回收或复用其底层显存。
            if host_indices.is_cuda:
                host_indices.record_stream(self.load_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.load_stream)

        # 将本轮最后一层的完成事件和所有 node_id 放入确认队列。
        # loading_check() 等 finish_event 就绪后，会用 node_ids 找到
        # ongoing_load_back 中的节点并释放对应的在途保护引用。
        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )

        # 返回事件组索引；scheduler 写入 new_batch.hicache_consumer_index，
        # 随后的 forward 据此逐层等待本轮 L2 -> L1 数据就绪。
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers."""
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        logger.info(
            "HiCache draft KV registered: %s (host %d slots)",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation."""
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            if self.storage_config.should_split_heads:
                logger.warning(
                    "HiCache draft L3 disabled: should_split_heads not yet "
                    "supported on the mooncake v2 path."
                )
                return
            self.storage_backend.register_mem_host_pool_v2(
                self.mem_pool_host_draft, PoolName.DRAFT
            )
            self.draft_page_get_func = self._draft_page_get_v2
            self.draft_page_set_func = self._draft_page_set_v2
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor):
        if host_indices.numel() == 0:
            return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            self.host_mem_release_queue.put(page)

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.has_draft:
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :]
                )
            except Empty:
                continue

    def prefetch_rate_limited(self) -> bool:
        """
        Rate limit the prefetching operations to avoid overwhelming the storage backend.
        """
        # cancel prefetch if too much memory is occupied
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        # todo: more sophisticated rate limiting based on storage backend performance
        return False

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        """查询 L3 中从本地锚点开始的最长连续 page 命中，不读取实际 KV。"""

        # last_hash 是 L1/L2 本地命中终点的最后一个 page hash；没有本地
        # prefix 时为 None，后续 page hash 将从空链开始计算。
        last_hash = operation.last_hash

        # token_ids 保存本地 L1/L2 尚未覆盖、准备查询 L3 的 page 对齐后缀。
        # 当前调用中它实际可以是 RadixKey，但支持 len/迭代，能直接参与 hash 计算。
        tokens_to_fetch = operation.token_ids

        # 某些 backend 需要完整祖先 hash 上下文。使用副本是为了在分批查询时
        # 追加本批 hash，而不修改 operation 中保存的原始元数据；File backend 忽略它。
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        # storage_query_count 以 token slot 为单位累计连续命中长度；
        # hash_value 保存对应的 page hash 字符串，不包含任何 KV 数据。
        storage_query_count = 0
        hash_value = []

        # 按 page_size 将待查询 token 分组，并从 last_hash 开始计算链式 hash。
        # 返回列表中每个元素对应一个 page，后一个 page hash 依赖前一个 page hash。
        page_hashes = self.get_hash_str(
            tokens_to_fetch, last_hash, page_size=self.page_size
        )

        # page 数量可能很大，按 STORAGE_BATCH_SIZE 分批向 storage backend
        # 查询文件/key 是否存在，避免一次提交过大的 exists 请求。
        for start in range(0, len(page_hashes), STORAGE_BATCH_SIZE):
            batch_hashes = page_hashes[start : start + STORAGE_BATCH_SIZE]

            # 可选 backend 元数据。当前 File/SSD backend 只按 page hash 查文件，
            # 不读取 prefix_keys；需要层级上下文的 backend 可以使用它。
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)

            # batch_exists() 返回当前 batch 从第一个 hash 开始连续存在的 page 数。
            # 例如 [存在, 存在, 缺失, 存在] 返回 2，不会跨过中间缺页。
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)

            # 只保留本批连续命中的 page hash，并把 page 数转换为 token slot 数。
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size

            # 当前 batch 没有全部命中，说明连续 prefix 在第一个缺页处结束；
            # 后续 page 即使单独存在，也不能作为当前请求的连续 L3 prefix 复用。
            if hit_page_num < len(batch_hashes):
                break

            # 完整命中本批后，为需要完整 hash 链的 backend 补充上下文，
            # 供下一批 exists 查询使用。当前 File backend 默认不会进入该分支。
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        # hash_value：最长连续命中的 page hash 列表（长度单位为 page）；
        # storage_query_count：同一命中范围覆盖的 token slot 数。
        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        查询每个预取任务在 L3 中的最长连续命中，并把通过收益检查的任务
        裁剪后交给独立 I/O 线程执行实际的 L3 -> L2 读取。
        TODO: sparse prefill需要看
        本线程负责 hash/exists 查询和任务规划，不直接读取 KV 数据。
        """
        # prefetch_buffer 是“命中查询线程 -> 实际 I/O 线程”的中间队列。
        # 只有确认达到 prefetch_threshold 的 operation 才会进入该队列。
        self.prefetch_buffer = Queue()

        # 启动第二个 daemon 线程。prefetch_io_aux_func() 从 prefetch_buffer
        # 取任务，并通过 _page_transfer() 调用 storage backend 读取 SSD page。
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()

        # 正常运行时持续接收任务；收到停止信号后仍处理完已经进入
        # prefetch_queue 的查询任务，再退出本线程。
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                # 最多阻塞 1 秒，以便队列暂时为空时周期性重新检查停止信号。
                operation = self.prefetch_queue.get(block=True, timeout=1)

                # None 是停止/唤醒线程使用的哨兵，不对应真实请求。
                if operation is None:
                    continue

                # 根据 operation.last_hash 和待查询 token 生成链式 page hash，
                # 再向 L3 backend 查询从第一个 page 开始的最长连续命中：
                # - hash_value：实际连续命中的 page hash 列表；
                # - storage_hit_count：命中的 token slot 数，而非 page 数。
                hash_value, storage_hit_count = self._storage_hit_query(operation)

                # TP/CP 各 rank 分别保存同一 page 的不同 KV 分片。只有所有相关
                # rank 都命中的连续前缀才能复用，因此通过 MIN 取共同命中长度。
                # 当前 TP=1、CP=1 时，该 all-reduce 不会改变数值。
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()

                # 共同命中长度不足收益阈值时撤销任务：
                # - request_id 进入 revoke queue，由主线程清理 ongoing_prefetch
                #   并释放 last_host_node 的保护引用；
                # - 所有预分配的 L2 Host slot 进入延迟释放队列。
                if storage_hit_count < self.prefetch_threshold:
                    self.prefetch_revoke_queue.put(operation.request_id)
                    self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    # storage_hit_count 使用 token 为单位；除以 page_size 得到共同
                    # 命中的 page 数，并将本 rank 的 hash 列表裁剪到该范围。
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]

                    # host_indices 最初为完整候选范围预留。将共同命中范围之后的
                    # L2 slot 送入释放队列，只把真正要读取的前缀保留给 operation。
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )

                    # 交给 I/O 线程执行实际 File/SSD page 读取。该线程完成写入后
                    # 会逐 page 增加 operation.completed_tokens，供 scheduler 轮询。
                    self.prefetch_buffer.put(operation)

            # timeout 只表示这一秒内没有新任务，属于正常空闲状态。
            except Empty:
                continue

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup."""
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`)."""
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)

    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide.
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> None:
        draft_keys = [f"{h}.{PoolName.DRAFT}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            return
        for i, p in enumerate(draft_pages):
            if p is not None:
                self.mem_pool_host_draft.set_from_flat_data_page(
                    host_indices[i * self.page_size], p
                )

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.has_draft:
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                if not self.backup_skip:
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                continue
