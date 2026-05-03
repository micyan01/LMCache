# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, Union
import abc

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector.utils import (
    DiscoverableKVCache,
    LayoutHints,
    assert_is_vllm_flash_attn_or_flash_infer,
    assert_is_vllm_mla_or_flash_attn_or_flash_infer,
    attempt_permute_to_contiguous_view,
    discover_gpu_kv_format,
    get_block_size,
    get_device,
    get_elements_per_layer,
    get_group_data_ptrs,
    get_head_size,
    get_num_blocks,
    get_num_layers,
    get_page_buffer_size,
    get_tokens_per_layer,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class GPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched load the data from a GPU memory into the memory objects.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data from GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Batched store the data from the memory objects to GPU kv cache.
        Sub-classes should define the format of the kwargs.

        For non-layerwise connectors:
        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data to GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.

        For layerwise connectors (generator pattern):
        :param List[int] memory_objs: Actually the starts list
        (positional compatibility)
        :param List[int] starts: Actually the ends list
        (positional compatibility)
        Note: Layerwise connectors receive memory objects
        via generator.send()
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]
            # Ensure contiguity on every call.  HND tensors from vLLM have a
            # non-contiguous logical view (NHD) that must be permuted back to
            # the physical (HND) shape for correct kernel indexing.
            # attempt_permute_to_contiguous_view is a no-op when already contiguous.
            self.kvcaches = attempt_permute_to_contiguous_view(self.kvcaches)


class VLLMPagedMemCPUConnectorV2(GPUConnectorInterface):
    """
    CPU-only version of the connector.
    Assumes both source (vLLM paged KV cache) and destination (LMCache MemoryObj)
    are in CPU memory. No GPU involvement needed.
    """



    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )
    
    
    def __init__(self, hidden_dim_size: int, num_layers: int, **kwargs):
        print("init cpu connector")
        print("=== Call stack ===")
        traceback.print_stack()
        print("=================")

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kvcaches: Optional[List[torch.Tensor]] = None
        
        # 不需要 GPU 指针表，直接用 CPU 地址
        self.kv_cache_pointers = torch.empty(num_layers, dtype=torch.int64, device="cpu")
        
        # 存储格式元数据
        self.layout_hints: LayoutHints = kwargs.get("layout_hints") or {}
        self.use_mla = kwargs.get("use_mla", False)
    
    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:

        print("VLLMPagedMemCPUConnectorV2: _initialize_pointers")
        traceback.print_stack()
        print("=================")
        """初始化 CPU Page Cache 的指针（不需要 GPU）"""
        device = kv_caches[0].device
        assert device.type == "cpu", "kv_caches must be on CPU"
        
        # 确保内存连续
        kv_caches = attempt_permute_to_contiguous_view(kv_caches)
        
        # 直接把 CPU 地址存下来（不需要 copy 到 GPU）
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        
        # 格式发现（复用原有逻辑）
        self.gpu_kv_format = discover_gpu_kv_format(
            kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)
        
        # 返回 CPU 指针表（而不是 GPU 上的）
        return self.kv_cache_pointers
    
    # ==================== 必须实现的接口方法 ====================
    
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)


        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        kv_cache_device = self.kvcaches[0].device
        print("to_gpu: memory_obj.tensor.device:", memory_obj.tensor.device, "kv_cache_device:", kv_cache_device)

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))
       
        print(
            "to_gpu: transfer from CPU to CPU, using CPU fallback copy path."
        )


        print("=== to_gpu CPU fallback ENTER ===", flush=True)

        kv0 = self.kvcaches[0]
        print(f"[chk] is_contiguous={kv0.is_contiguous()}", flush=True)
        print(f"[chk] shape={tuple(kv0.shape)}  stride={kv0.stride()}", flush=True)
        print("memory_obj.tensor:", memory_obj.tensor.shape,
            memory_obj.tensor.device, memory_obj.tensor.dtype, flush=True)
        print("kvcaches[0]:", self.kvcaches[0].shape,
            self.kvcaches[0].device, self.kvcaches[0].dtype, flush=True)

        print("kvcaches[1]:", self.kvcaches[1].shape,
            self.kvcaches[1].device, self.kvcaches[1].dtype, flush=True)
        print("start,end:", start, end, flush=True)
        print("slot_mapping", slot_mapping.shape, slot_mapping.device, slot_mapping.dtype, flush=True)
        print("slot_mapping", slot_mapping)


        sm = slot_mapping[start:end].to("cpu", dtype=torch.long)

        kv0 = self.kvcaches[0]
        _, num_blocks, heads, block_size, head_size = kv0.shape

        hidden_dim = memory_obj.tensor.shape[-1]
        num_kv_heads = hidden_dim // head_size

        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))
        print("memory object", memory_obj.tensor[0, 0, 0, :]) # k=0, l=0 s=0, 
        # CPU vLLM layout: [2, num_blocks, num_kv_heads, block_size, head_size]
        for layer_idx, kv_cache in enumerate(self.kvcaches):
            print("layer_idx", layer_idx)
            print("sm", sm)
            print("sm.tolist()", sm.tolist())
            for local_i, slot in enumerate(sm.tolist()):
                if local_i < skip_prefix_n_tokens or slot < 0:
                    continue
                block_id = slot // block_size
                block_offset = slot % block_size
                print("block_id, block_offset", block_id, block_offset)
                tmp_mem = memory_obj.tensor[:, layer_idx, local_i, :]
                print(f"tmp_mem.shape", tmp_mem.shape)
                token_kv = tmp_mem.reshape(
                    2, num_kv_heads, head_size) # src is 【2， 16， 256， 512】
                print("tmp_mem", tmp_mem)
                kv_cache[0, block_id, :, block_offset, :].copy_(token_kv[0]) # 【8，64】
                kv_cache[1, block_id, :, block_offset, :].copy_(token_kv[1])
            tmp_tensor = self.kvcaches[0][0, sm.tolist()[0] // block_size, :, 0:32, :].cpu()
            tmp_tensor = tmp_tensor.permute(1, 0, 2).contiguous()
            print("layer 0 k cache after copy", tmp_tensor.shape, tmp_tensor)

        print("=== to_gpu CPU fallback EXIT ===", flush=True)
        
        return

    
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        print("using cpu connector from_gpu")

        """
        从 vLLM 的 CPU Page Cache 存储到 LMCache MemoryObj。
        
        传输方向：vLLM Page Cache (CPU) -> MemoryObj (CPU)
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,           # 目标：LMCache MemoryObj (CPU)
            kv_cache_pointers,           # 源：vLLM Page Cache 地址表 (CPU)
            slot_mapping[start:end],
            self.kvcaches[0].device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.D2H,  # D2H 语义：从 Device (vLLM Cache) 到 Host (LMCache)
            self.gpu_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
        )
        
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
    
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        """批量加载（CPU 版本：直接循环，不需要 CUDA stream）"""
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)
    
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        """批量存储（CPU 版本：直接循环）"""
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)
    
    def get_shape(self, num_tokens: int) -> torch.Size:
        """返回 MemoryObj 的形状"""
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])
