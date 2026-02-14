import os
import re
import ftfy
import html
import math
import torch
import torch.nn as nn
from functools import partial
from typing import Any, Optional, Dict, List, Literal, Union

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from diffusers.pipelines import DiffusionPipeline
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.configuration_utils import register_to_config
from diffusers.utils import CONFIG_NAME
from transformers.models import UMT5EncoderModel, T5TokenizerFast
from yunchang.kernels import AttnType

from mova.utils.misc import gpu_timer, track_gpu_mem
from mova.engine.utils.base_pipeline import BasePipeline
from mova.diffusion.models import (
    WanAudioModel, WanModel, sinusoidal_embedding_1d, 
) 

from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge
from mova.diffusion.models.dac_vae import DAC

from mova.registry import MODELS, DIFFUSION_PIPELINES, DIFFUSION_SCHEDULERS

from mova.distributed.functional import (
    _sp_split_tensor, _sp_split_tensor_dim_0, _sp_all_gather_avg
)

from dataclasses import dataclass
from contextlib import nullcontext

# DeepSpeed ZeRO-3 support
try:
    from deepspeed.runtime.zero.partition_parameters import GatheredParameters
    DEEPSPEED_ZERO3_AVAILABLE = True
except ImportError:
    DEEPSPEED_ZERO3_AVAILABLE = False
    GatheredParameters = None

# FP8 CPU Offload support
try:
    from mova.engine.trainer.utils.fp8_cpu_offload import (
        FP8CPUOffloadManager, 
        quantize_to_fp8, 
        dequantize_from_fp8,
        FP8_AVAILABLE,
        FP8_DTYPE
    )
    FP8_OFFLOAD_AVAILABLE = True
except ImportError:
    FP8_OFFLOAD_AVAILABLE = False
    FP8_AVAILABLE = False


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

@dataclass
class TimestepConfig:
    max_timestep_boundary: float = 1.0
    min_timestep_boundary: float = 0.0
    weighting_scheme: str = "uniform"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.0
    independent_timesteps: bool = False
    pair_postprocess: Optional[str] = None
    pair_postprocess_kwargs: Optional[dict] = None


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    min_timestep_boundary: float = 0.0,
    max_timestep_boundary: float = 1.0,
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.zeros(size=(batch_size,), device="cpu")
        a = torch.logit(torch.tensor(min_timestep_boundary))
        b = torch.logit(torch.tensor(max_timestep_boundary))
        torch.nn.init.trunc_normal_(u, mean=logit_mean, std=logit_std, a=a, b=b)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        assert min_timestep_boundary == 0.0 and max_timestep_boundary == 1.0, (
            "mode weighting scheme only supports [0,1] range for now"
        )
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
        u = min_timestep_boundary + (u * (max_timestep_boundary - min_timestep_boundary))
    return u

class MOVATrain(BasePipeline, DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"

    video_vae: AutoencoderKLWan
    audio_vae: DAC
    text_encoder: UMT5EncoderModel
    tokenizer: T5TokenizerFast
    scheduler: Any
    video_dit: WanModel
    video_dit_2: WanModel
    audio_dit: WanAudioModel
    dual_tower_bridge: DualTowerConditionalBridge

    @register_to_config
    def __init__(
        self,
        video_vae: AutoencoderKLWan,
        audio_vae: DAC,
        text_encoder: UMT5EncoderModel,
        tokenizer: T5TokenizerFast,
        scheduler: Any,
        video_dit: WanModel,
        video_dit_2: WanModel,
        audio_dit: WanAudioModel,
        dual_tower_bridge: DualTowerConditionalBridge,
        audio_vae_type: str = "dac", # type: Literal["oobleck", "dac"]
        boundary_ratio: float = 0.9,
        # gradient checkpointing
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        super().__init__()

        self.register_modules(
            video_vae=video_vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            video_dit=video_dit,
            video_dit_2=video_dit_2,
            audio_dit=audio_dit,
            dual_tower_bridge=dual_tower_bridge,
        )

        self.register_to_config(
            audio_vae_type=audio_vae_type,
            boundary_ratio=boundary_ratio,
        )

        self.audio_vae_type = audio_vae_type
        self.boundary_ratio = boundary_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        # build vae
        self.height_division_factor = self.video_vae.spatial_compression_ratio * 2
        self.width_division_factor = self.video_vae.spatial_compression_ratio * 2
        
        # build audio vae
        self.audio_samples_division_factor = int(self.audio_vae.hop_length)
        self.sample_rate = self.audio_vae.sample_rate
        self.audio_latent_dim = self.audio_vae.latent_dim
    
    
    # ============================================================
    # Layer-wise Dynamic CPU Offload with Hooks
    # ============================================================
    
    def setup_layer_offload(
        self,
        device: torch.device,
        target_modules: List[str] = None,
        offload_granularity: str = "all",  # "all", "block", or "layer"
        use_pinned_memory: bool = True,
        exclude_patterns: List[str] = None,
    ):
        """
        Setup layer-wise dynamic CPU offloading using PyTorch hooks.
        
        When a layer starts forward/backward, load its weights to GPU.
        When it finishes, offload weights back to CPU immediately.
        
        Args:
            device: GPU device for computation
            target_modules: Top-level module names to apply offload
            offload_granularity: 
                - "all": All direct children (time_embedding, text_embedding, blocks, head, etc.)
                - "block": Only transformer blocks
                - "layer": Every leaf module with parameters
            use_pinned_memory: Use pinned memory for faster CPU-GPU transfer
            exclude_patterns: Patterns to exclude from offloading (e.g., LoRA layers)
        """
        if target_modules is None:
            target_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge", "text_encoder", "video_vae", "audio_vae"]
        
        if exclude_patterns is None:
            exclude_patterns = ["lora_"]
        
        self._offload_device = device
        self._offload_hooks = []
        self._cpu_weight_storage = {}  # module_id -> {param_name: (cpu_tensor, original_shape)}
        self._use_pinned_memory = use_pinned_memory
        self._exclude_patterns = exclude_patterns
        
        print(f"[LayerOffload] Setting up layer-wise CPU offloading...")
        print(f"[LayerOffload] Target device: {device}")
        print(f"[LayerOffload] Granularity: {offload_granularity}")
        print(f"[LayerOffload] Pinned memory: {use_pinned_memory}")
        
        # Track total CPU memory used for offloading
        self._total_offload_bytes = 0
        
        total_modules = 0
        total_params = 0
        
        for module_name in target_modules:
            if not hasattr(self, module_name):
                continue
            
            top_module = getattr(self, module_name)
            if top_module is None:
                continue
            
            # Decide which sub-modules to hook based on granularity
            if offload_granularity == "all":
                # Hook ALL direct children (time_embedding, text_embedding, blocks, head, etc.)
                modules_to_hook = self._get_all_submodules(top_module, module_name)
            elif offload_granularity == "block":
                # Hook transformer blocks (usually named 'blocks' or similar)
                modules_to_hook = self._get_transformer_blocks(top_module, module_name)
            else:
                # Hook every leaf module with parameters
                modules_to_hook = self._get_leaf_modules(top_module, module_name)
            
            for sub_name, sub_module in modules_to_hook:
                # Skip if matches exclude pattern
                if any(pat in sub_name for pat in exclude_patterns):
                    continue
                
                # Skip modules without parameters (check recursively for containers)
                params = list(sub_module.parameters())
                if not params:
                    continue
                
                # Check if this is part of an inference module (offload ALL params)
                inference_mods = getattr(self, '_inference_modules', set())
                is_inference = any(sub_name.startswith(inf_mod) for inf_mod in inference_mods)
                
                # Move weights to CPU and register hooks
                self._setup_module_offload(sub_module, sub_name, device, offload_all=is_inference)
                total_modules += 1
                total_params += sum(p.numel() for p in params)
        
        self.use_layer_offload = True
        
        print(f"[LayerOffload] Registered offload for {total_modules} modules")
        print(f"[LayerOffload] Total parameters managed: {total_params / 1e9:.2f}B")
        print(f"[LayerOffload] CPU memory used for offload storage: {self._total_offload_bytes / 1e9:.2f} GB")
        
        # Show FP8 vs non-FP8 breakdown
        fp8_bytes = getattr(self, '_fp8_offload_bytes', 0)
        non_fp8_bytes = getattr(self, '_non_fp8_offload_bytes', 0)
        buffer_bytes = getattr(self, '_buffer_offload_bytes', 0)
        print(f"[LayerOffload]   - FP8 quantized params: {fp8_bytes / 1e9:.2f} GB")
        print(f"[LayerOffload]   - Non-FP8 params: {non_fp8_bytes / 1e9:.2f} GB")
        print(f"[LayerOffload]   - Buffers: {buffer_bytes / 1e9:.2f} GB")
        
        # Show top 10 largest buffers
        if hasattr(self, '_buffer_sizes'):
            sorted_buffers = sorted(self._buffer_sizes, key=lambda x: x[1], reverse=True)[:10]
            print(f"[LayerOffload] Top 10 largest buffers:")
            for buf_name, buf_size in sorted_buffers:
                print(f"[LayerOffload]   - {buf_name}: {buf_size / 1e9:.3f} GB")
    
    # Alias for backward compatibility
    def setup_fp8_cpu_offload(
        self,
        device: torch.device,
        target_modules: List[str] = None,
        exclude_patterns: List[str] = None,
        include_inference_modules: bool = True,  # Also offload VAE, text_encoder
        **kwargs,
    ):
        """
        Setup layer-wise CPU offloading for all modules.
        
        Args:
            device: GPU device for computation
            target_modules: Training modules to offload
            exclude_patterns: Patterns to exclude (e.g., LoRA)
            include_inference_modules: Also offload text_encoder, video_vae, audio_vae
        """
        print("[FP8 Offload] Setting up FP8 CPU offloading...")
        
        if target_modules is None:
            target_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        
        # Inference modules - offload ALL parameters (regardless of requires_grad)
        inference_modules = []
        if include_inference_modules:
            inference_modules = ["text_encoder", "video_vae", "audio_vae"]
            print(f"[FP8 Offload] Including inference modules: {inference_modules}")
        
        # Store which modules are inference-only (offload all params)
        self._inference_modules = set(inference_modules)
        
        all_modules = target_modules + inference_modules
        
        print(f"[FP8 Offload] Inference modules (offload all params): {self._inference_modules}")
        print(f"[FP8 Offload] All target modules: {all_modules}")
        
        result = self.setup_layer_offload(
            device=device,
            target_modules=all_modules,
            offload_granularity="all",  # Offload all submodules including time_embedding
            use_pinned_memory=False,  # Disabled by default to save CPU memory
            exclude_patterns=exclude_patterns,
        )
        
        return result
    
    def _get_transformer_blocks(self, module: nn.Module, prefix: str) -> List[tuple]:
        """Get transformer blocks from a module."""
        blocks = []
        
        # Common names for transformer blocks
        block_names = ['blocks', 'layers', 'encoder_layers', 'decoder_layers']
        
        for name in block_names:
            if hasattr(module, name):
                block_list = getattr(module, name)
                if isinstance(block_list, (nn.ModuleList, nn.Sequential)):
                    for i, block in enumerate(block_list):
                        blocks.append((f"{prefix}.{name}.{i}", block))
                    break
        
        # Also add other named children that look like blocks
        for name, child in module.named_children():
            if name not in block_names and 'block' in name.lower():
                blocks.append((f"{prefix}.{name}", child))
        
        # If no blocks found, return the module itself
        if not blocks:
            blocks.append((prefix, module))
        
        return blocks
    
    def _get_all_submodules(self, module: nn.Module, prefix: str) -> List[tuple]:
        """
        Recursively get ALL submodules that have direct parameters.
        This ensures every module with weights gets its own hook.
        """
        submodules = []
        
        def _recurse(mod, path):
            # Check if this module has direct parameters (not from children)
            has_direct_params = len(list(mod.parameters(recurse=False))) > 0
            has_direct_buffers = len(list(mod.named_buffers(recurse=False))) > 0
            
            if has_direct_params or has_direct_buffers:
                # This module has its own parameters, add it
                submodules.append((path, mod))
            
            # Recurse into children
            for name, child in mod.named_children():
                child_path = f"{path}.{name}" if path else name
                
                # Handle different container types
                if isinstance(child, (nn.ModuleList, nn.Sequential)):
                    for i, sub_child in enumerate(child):
                        _recurse(sub_child, f"{child_path}.{i}")
                elif isinstance(child, nn.ModuleDict):
                    for key, sub_child in child.items():
                        _recurse(sub_child, f"{child_path}.{key}")
                else:
                    _recurse(child, child_path)
        
        _recurse(module, prefix)
        return submodules
    
    def _get_leaf_modules(self, module: nn.Module, prefix: str) -> List[tuple]:
        """Get leaf modules (modules without children that have parameters)."""
        leaves = []
        for name, child in module.named_modules():
            # Check if it's a leaf (no children)
            if len(list(child.children())) == 0:
                # Check if it has parameters
                if len(list(child.parameters(recurse=False))) > 0:
                    full_name = f"{prefix}.{name}" if name else prefix
                    leaves.append((full_name, child))
        return leaves
    
    def _setup_module_offload(self, module: nn.Module, name: str, device: torch.device, offload_all: bool = False):
        """Setup offload for a single module (only its DIRECT parameters, not children).
        
        Args:
            module: The module to setup offload for
            name: Name of the module (for logging)
            device: GPU device for computation
            offload_all: If True, offload ALL params regardless of requires_grad
                        (used for inference modules like text_encoder, VAE)
        """
        module_id = id(module)
        
        # Check if this module was already processed (same object with different path)
        if module_id in self._cpu_weight_storage:
            return  # Already has hooks registered
        
        self._cpu_weight_storage[module_id] = {}
        
        offloaded_params = 0
        offloaded_buffers = 0
        skipped_params = 0
        already_empty = 0
        
        # Check if this module uses weight_norm (has weight_g and weight_v instead of weight)
        # weight_norm registers its own forward_pre_hook that runs before ours,
        # causing issues when weights are empty. Solution: remove weight_norm first.
        has_weight_norm = hasattr(module, 'weight_g') and hasattr(module, 'weight_v')
        if has_weight_norm:
            try:
                torch.nn.utils.remove_weight_norm(module)
            except Exception as e:
                print(f"[WARNING] Failed to remove weight_norm from {name}: {e}")
        
        # Only handle DIRECT parameters (recurse=False) - children have their own hooks
        for param_name, param in module.named_parameters(recurse=False):
            if not offload_all and param.requires_grad:
                # Skip trainable params (LoRA weights should stay on GPU)
                # But for inference modules (offload_all=True), offload everything
                skipped_params += 1
                continue
            
            # Skip if already empty (already offloaded) - but track for weight sharing
            if param.numel() == 0:
                already_empty += 1
                # Track this as a shared weight - find the source storage
                if not hasattr(self, '_shared_weight_sources'):
                    self._shared_weight_sources = {}
                # Store that this module.param needs to load from somewhere
                self._shared_weight_sources[(module_id, param_name)] = None  # Will find source later
                continue
            
            # Store original on CPU with optional FP8 quantization
            original_dtype = param.dtype
            original_shape = param.shape
            
            # Try to use FP8 quantization to save CPU memory (BF16 -> FP8 = 50% memory reduction)
            use_fp8 = FP8_OFFLOAD_AVAILABLE and original_dtype in (torch.bfloat16, torch.float16, torch.float32)
            
            #  print FP8 availability once
            if not hasattr(self, '_fp8_debug_printed'):
                self._fp8_debug_printed = True
                print(f"[LayerOffload] FP8_OFFLOAD_AVAILABLE={FP8_OFFLOAD_AVAILABLE}, FP8_AVAILABLE={FP8_AVAILABLE}, FP8_DTYPE={FP8_DTYPE}")
            
            if use_fp8:
                try:
                    # Quantize to FP8 for CPU storage
                    # Pass to_cpu=True to move to CPU first, avoiding GPU memory spike
                    fp8_data, scale = quantize_to_fp8(param.data, to_cpu=True)
                    # fp8_data is already on CPU due to to_cpu=True
                    cpu_data = fp8_data if fp8_data.device.type == 'cpu' else fp8_data.cpu()
                    cpu_scale = scale.cpu() if scale is not None else None
                    
                    self._cpu_weight_storage[module_id][param_name] = {
                        'cpu_data': cpu_data,
                        'scale': cpu_scale,
                        'shape': original_shape,
                        'dtype': original_dtype,
                        'requires_grad': param.requires_grad,
                        'is_fp8': True,
                    }
                    
                    # Track memory usage (FP8 = 1 byte per element)
                    fp8_bytes = cpu_data.numel() * cpu_data.element_size()  # actual bytes
                    self._total_offload_bytes += fp8_bytes
                    if cpu_scale is not None:
                        self._total_offload_bytes += cpu_scale.numel() * cpu_scale.element_size()
                    
                    # Track FP8 vs non-FP8 separately
                    if not hasattr(self, '_fp8_offload_bytes'):
                        self._fp8_offload_bytes = 0
                        self._non_fp8_offload_bytes = 0
                    self._fp8_offload_bytes += fp8_bytes
                except Exception:
                    # Fallback to non-FP8 storage
                    use_fp8 = False
            
            if not use_fp8:
                # Standard storage without FP8
                cpu_data = param.data.cpu()
                
                # Optional: use pinned memory for faster transfer (but uses more memory)
                if self._use_pinned_memory:
                    try:
                        if not cpu_data.is_pinned():
                            cpu_data = cpu_data.pin_memory()
                    except Exception:
                        pass
                
                self._cpu_weight_storage[module_id][param_name] = {
                    'cpu_data': cpu_data,
                    'shape': original_shape,
                    'dtype': original_dtype,
                    'requires_grad': param.requires_grad,
                    'is_fp8': False,
                }
                
                # Track memory usage
                non_fp8_bytes = cpu_data.numel() * cpu_data.element_size()
                self._total_offload_bytes += non_fp8_bytes
                
                # Track FP8 vs non-FP8 separately
                if not hasattr(self, '_fp8_offload_bytes'):
                    self._fp8_offload_bytes = 0
                    self._non_fp8_offload_bytes = 0
                self._non_fp8_offload_bytes += non_fp8_bytes
            
            # Track which tensor this param is, so we can find shared weights later
            if not hasattr(self, '_param_tensor_to_storage'):
                self._param_tensor_to_storage = {}
            self._param_tensor_to_storage[id(param)] = (module_id, param_name)
            
            # Replace with empty placeholder to free GPU memory
            param.data = torch.empty(0, dtype=param.dtype, device='cpu')
            offloaded_params += 1
        
        # Only handle DIRECT buffers (recurse=False)
        # Also apply FP8 quantization to buffers to save memory
        for buf_name, buf in module.named_buffers(recurse=False):
            if buf is None or buf.numel() == 0:
                continue
            
            original_dtype = buf.dtype
            original_shape = buf.shape
            
            # Try FP8 quantization for buffers too
            use_fp8_buf = FP8_OFFLOAD_AVAILABLE and original_dtype in (torch.bfloat16, torch.float16, torch.float32)
            
            if use_fp8_buf:
                try:
                    fp8_data, scale = quantize_to_fp8(buf.data, to_cpu=True)
                    cpu_data = fp8_data if fp8_data.device.type == 'cpu' else fp8_data.cpu()
                    cpu_scale = scale.cpu() if scale is not None else None
                    
                    self._cpu_weight_storage[module_id][f"_buffer_{buf_name}"] = {
                        'cpu_data': cpu_data,
                        'scale': cpu_scale,
                        'shape': original_shape,
                        'dtype': original_dtype,
                        'is_buffer': True,
                        'is_fp8': True,
                    }
                    
                    # Track buffer memory (FP8)
                    buf_bytes = cpu_data.numel() * cpu_data.element_size()
                    self._total_offload_bytes += buf_bytes
                    if cpu_scale is not None:
                        self._total_offload_bytes += cpu_scale.numel() * cpu_scale.element_size()
                    if not hasattr(self, '_buffer_offload_bytes'):
                        self._buffer_offload_bytes = 0
                    self._buffer_offload_bytes += buf_bytes
                    
                except Exception:
                    use_fp8_buf = False
            
            if not use_fp8_buf:
                # Fallback: store without FP8
                cpu_data = buf.detach().cpu()
                if self._use_pinned_memory:
                    try:
                        cpu_data = cpu_data.pin_memory()
                    except Exception:
                        pass
                
                self._cpu_weight_storage[module_id][f"_buffer_{buf_name}"] = {
                    'cpu_data': cpu_data,
                    'shape': original_shape,
                    'dtype': original_dtype,
                    'is_buffer': True,
                    'is_fp8': False,
                }
                
                # Track buffer memory (non-FP8)
                buf_bytes = cpu_data.numel() * cpu_data.element_size()
                self._total_offload_bytes += buf_bytes
                if not hasattr(self, '_buffer_offload_bytes'):
                    self._buffer_offload_bytes = 0
                self._buffer_offload_bytes += buf_bytes
            
            buf.data = torch.empty(0, dtype=buf.dtype, device='cpu')
            offloaded_buffers += 1
        
        # Handle shared weights - if we have empty params, find their source
        shared_param_sources = {}  # param_name -> source_module_id
        if already_empty > 0 and hasattr(self, '_param_tensor_to_storage'):
            for param_name, param in module.named_parameters(recurse=False):
                if param.numel() == 0:
                    # This param was emptied by another module - find its source
                    # We need to load from that source module
                    param_id = id(param)
                    if param_id in self._param_tensor_to_storage:
                        source_module_id, source_param_name = self._param_tensor_to_storage[param_id]
                        shared_param_sources[param_name] = (source_module_id, source_param_name)
        
        # Register hooks if we offloaded something OR have shared params to load
        if offloaded_params == 0 and offloaded_buffers == 0 and not shared_param_sources:
            del self._cpu_weight_storage[module_id]
            return
        
        # Store shared param sources for this module
        if shared_param_sources:
            self._cpu_weight_storage[module_id]['_shared_sources'] = shared_param_sources
        
        # Determine if this is an inference-only module (no backward needed)
        is_inference_module = offload_all  # offload_all=True means it's an inference module
        
        # Register forward pre-hook: load weights to GPU before forward
        def pre_forward_hook(mod, inputs):
            self._load_module_to_gpu(mod)
        
        h1 = module.register_forward_pre_hook(pre_forward_hook)
        self._offload_hooks.append(h1)
        
        if is_inference_module:
            # For inference modules: offload immediately after forward (no backward)
            def post_forward_hook_inference(mod, inputs, outputs):
                self._offload_module_to_cpu(mod)
            
            h2 = module.register_forward_hook(post_forward_hook_inference)
            self._offload_hooks.append(h2)
        else:
            # For training modules: offload immediately after forward, reload before backward
            # This saves GPU memory but increases CPU-GPU transfer overhead
            
            # Check if backward_pre_hook is available (PyTorch >= 1.11)
            has_backward_pre_hook = hasattr(module, 'register_full_backward_pre_hook')
            
            if has_backward_pre_hook:
                # Smart offload strategy compatible with gradient checkpointing:
                #
                # The key insight: with gradient checkpointing, forward is called twice:
                # 1. Normal forward (building computation graph)
                # 2. Recompute forward (during backward, to compute gradients)
                #
                # We use a counter to track forward calls:
                # - First forward: offload after (save memory during forward pass)
                # - Second forward (recompute): DON'T offload (gradient computation needs weights)
                # - After backward: reset counter and offload
                
                # Initialize forward counter
                module._forward_count = 0
                
                def post_forward_hook_training(mod, inputs, outputs):
                    mod._forward_count = getattr(mod, '_forward_count', 0) + 1
                    
                    if mod._forward_count == 1:
                        # First forward (normal): offload to save memory
                        self._offload_module_to_cpu(mod)
                    # else: recompute forward, don't offload (gradient computation needs weights)
                    
                    mod._forward_done = True
                
                def backward_pre_hook(mod, grad_output):
                    # Load weights back to GPU before backward computation
                    # This handles the case where weights were offloaded after first forward
                    self._load_module_to_gpu(mod)
                
                def backward_hook(mod, grad_input, grad_output):
                    # Offload after backward is complete and reset counter
                    self._offload_module_to_cpu(mod)
                    mod._forward_done = False
                    mod._forward_count = 0  # Reset for next iteration
                
                h2 = module.register_forward_hook(post_forward_hook_training)
                h3 = module.register_full_backward_pre_hook(backward_pre_hook)
                h4 = module.register_full_backward_hook(backward_hook)
                self._offload_hooks.extend([h2, h3, h4])
            else:
                # Fallback: keep weights on GPU until backward is done (less memory efficient)
                print(f"[WARNING] register_full_backward_pre_hook not available, "
                      f"using less aggressive offload strategy. Upgrade PyTorch >= 1.11 for better memory efficiency.")
                
                def post_forward_hook(mod, inputs, outputs):
                    mod._forward_done = True
                
                def backward_hook(mod, grad_input, grad_output):
                    self._offload_module_to_cpu(mod)
                    mod._forward_done = False
                
                h2 = module.register_forward_hook(post_forward_hook)
                if hasattr(module, 'register_full_backward_hook'):
                    h3 = module.register_full_backward_hook(backward_hook)
                else:
                    h3 = module.register_backward_hook(backward_hook)
                self._offload_hooks.extend([h2, h3])
        
        module._offload_enabled = True
        module._forward_done = False
    
    def _load_module_to_gpu(self, module: nn.Module):
        """Load a module's DIRECT weights from CPU to GPU."""
        module_id = id(module)
        if module_id not in self._cpu_weight_storage:
            return
        
        device = self._offload_device
        storage = self._cpu_weight_storage[module_id]
        
        # First, handle shared weights - load from source module's storage
        shared_sources = storage.get('_shared_sources', {})
        for param_name, (source_module_id, source_param_name) in shared_sources.items():
            if source_module_id in self._cpu_weight_storage:
                source_storage = self._cpu_weight_storage[source_module_id]
                if source_param_name in source_storage:
                    source_info = source_storage[source_param_name]
                    try:
                        # Handle FP8 for shared weights too
                        is_fp8 = source_info.get('is_fp8', False)
                        if is_fp8 and FP8_OFFLOAD_AVAILABLE:
                            fp8_data = source_info['cpu_data'].to(device, non_blocking=False)
                            scale = source_info.get('scale')
                            if scale is not None:
                                scale = scale.to(device, non_blocking=False)
                            original_dtype = source_info['dtype']
                            gpu_data = dequantize_from_fp8(fp8_data, scale, original_dtype)
                        else:
                            gpu_data = source_info['cpu_data'].to(device, non_blocking=False)
                        
                        if hasattr(module, '_parameters') and param_name in module._parameters:
                            module._parameters[param_name].data = gpu_data
                    except Exception as e:
                        print(f"[LayerOffload] Warning: failed to load shared {param_name}: {e}")
        
        # Load direct parameters and buffers (owned by this module)
        for key, info in storage.items():
            if key.startswith('_') and not key.startswith('_buffer_'):
                continue  # Skip metadata like '_shared_sources'
            
            if key.startswith('_buffer_'):
                attr_name = key[8:]  # Remove '_buffer_' prefix
            else:
                attr_name = key
            
            try:
                # Check if this is FP8 quantized data
                is_fp8 = info.get('is_fp8', False)
                
                if is_fp8 and FP8_OFFLOAD_AVAILABLE:
                    # Dequantize from FP8 to original dtype
                    fp8_data = info['cpu_data'].to(device, non_blocking=False)
                    scale = info.get('scale')
                    if scale is not None:
                        scale = scale.to(device, non_blocking=False)
                    original_dtype = info['dtype']
                    gpu_data = dequantize_from_fp8(fp8_data, scale, original_dtype)
                else:
                    # Standard loading
                    gpu_data = info['cpu_data'].to(device, non_blocking=False)
                
                # Get the parameter/buffer
                if hasattr(module, '_parameters') and attr_name in module._parameters:
                    # Directly set the parameter data
                    param = module._parameters[attr_name]
                    param.data = gpu_data
                elif hasattr(module, '_buffers') and attr_name in module._buffers:
                    buf = module._buffers[attr_name]
                    buf.data = gpu_data
                else:
                    # Fallback to getattr
                    target = getattr(module, attr_name)
                    target.data = gpu_data
            except Exception as e:
                print(f"[LayerOffload] Warning: failed to load {attr_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Note: removed torch.cuda.synchronize() - using non_blocking=False instead might be safer
        # torch.cuda.synchronize()
    
    def _offload_module_to_cpu(self, module: nn.Module):
        """Offload a module's DIRECT weights back to CPU (free GPU memory)."""
        module_id = id(module)
        if module_id not in self._cpu_weight_storage:
            return
        
        storage = self._cpu_weight_storage[module_id]
        
        # Offload parameters - replace with empty tensor to free GPU memory
        # Keep the CPU copy in storage for next forward
        for key, info in storage.items():
            # Skip metadata keys
            if key.startswith('_') and not key.startswith('_buffer_'):
                continue
            
            if key.startswith('_buffer_'):
                attr_name = key[8:]
            else:
                attr_name = key
            
            try:
                target = getattr(module, attr_name)
                if hasattr(info, 'get') and 'dtype' in info:
                    target.data = torch.empty(0, dtype=info['dtype'], device='cpu')
            except Exception:
                pass  # Silently ignore errors during offload

    def replace_attention(self, attn_type: AttnType = AttnType.FA):
        from mova.diffusion.models.wan_video_dit import USPAttention
        # partial
        partial_replace = partial(USPAttention, attn_type=attn_type)
        replaced_cnt = 0
        for block in self.video_dit.blocks:
            block.self_attn.attn = partial_replace(block.self_attn.attn.num_heads)
            replaced_cnt += 1
        if self.video_dit_2 is not None:
            for block in self.video_dit_2.blocks:
                block.self_attn.attn = partial_replace(block.self_attn.attn.num_heads)
                replaced_cnt += 1
        if self.audio_dit is not None:
            for block in self.audio_dit.blocks:
                block.self_attn.attn = partial_replace(block.self_attn.attn.num_heads)
                replaced_cnt += 1
        if self.dual_tower_bridge is not None:
            for conditioner in self.dual_tower_bridge.audio_to_video_conditioners.values():
                conditioner.inner.attn = partial_replace(conditioner.inner.attn.num_heads)
                replaced_cnt += 1
            for conditioner in self.dual_tower_bridge.video_to_audio_conditioners.values():
                conditioner.inner.attn = partial_replace(conditioner.inner.attn.num_heads)
                replaced_cnt += 1
        return replaced_cnt


    def normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Normalize latents with config stats (diffusers Wan convention)
        mean = torch.tensor(self.video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        inv_std = (1.0 / torch.tensor(self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        latents = (latents - mean) * inv_std
        return latents
    

    def denormalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # Denormalize latents with config stats (diffusers Wan convention)
        mean = torch.tensor(self.video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        std = torch.tensor(self.video_vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.video_vae.config.z_dim, 1, 1, 1
        )
        latents = latents * std + mean
        return latents




    def inference_single_step(
        self,
        visual_dit,
        visual_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor],
        y,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: Optional[torch.Tensor],
        video_fps: float,
        cp_mesh=None,
    ):
        """
        Args:
            visual_latents:
                shape=[B, C=16, T // 4 + 1, H // 8, W // 8]
                dtype=bf16
            audio_latents:
                shape=[B, 128, 403]
                dtype=bf16
            y: first frame embedding
                shape=[B, C=20, T // 4 + 1, H // 8, W // 8]
                dtype=bf16
        """
        audio_context = visual_context = context  # [B, 512, C=4096]

        if audio_timestep is None:
            audio_timestep = timestep

        # t: timestep_embeddind
        # NOTE(dhyu): See https://github.com/Wan-Video/Wan2.2/blob/ee56ce852423d780065cd601ac299d0052b0f501/wan/modules/model.py#L462
        # The original Wan implementation ensures timestep computation happens in float32.
        # Always use bf16 for subsequent compute to avoid dtype mismatch when weights are bf16 but inputs are fp32.
        model_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = visual_dit.time_embedding(sinusoidal_embedding_1d(visual_dit.freq_dim, timestep))  # [B, C=5120]
            visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))  # [B, 6, C=5120]

            audio_t = self.audio_dit.time_embedding(sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep))
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))
        
        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)
        
        # prompt embedding
        visual_context_emb = visual_dit.text_embedding(visual_context)  # shape=[B, 512, C=5120]; dtype=bf16
        audio_context_emb = self.audio_dit.text_embedding(audio_context)  # shape=[B, 512, C=1536]; dtype=bf16
        

        visual_x = visual_latents.to(dtype=model_dtype, device=visual_latents.device)
        audio_x = audio_latents.to(dtype=model_dtype, device=audio_latents.device)
        if visual_dit.require_vae_embedding:
            visual_x = torch.cat([visual_x, y], dim=1).to(dtype=model_dtype, device=visual_latents.device)
        # NOTE: Force latents to bf16 to match weight dtype and avoid conv bias dtype mismatch from fp32 inputs.

        visual_x, (t, h, w) = visual_dit.patchify(visual_x)
        # visual_x: [B, L, C=5120]; L = (T // 4 + 1) * (H // 16) * (W // 16)

        grid_size = (t, h, w)
        # grid_size: [T // 4 + 1, H // 16, W // 16]

        # Move to CUDA first; otherwise expand on CPU costs ~25ms.
        visual_freqs = tuple(freq.to(visual_x.device) for freq in visual_dit.freqs)
        
        visual_freqs = torch.cat([
            visual_freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            visual_freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1), 
            visual_freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)
        ], dim=-1).reshape(t * h * w, 1, -1).to(visual_x.device)
        # visual_freqs: [L, 1, 64]

        # TODO(dhyu): Refactor multi-modal freqs generation logic.
        audio_x, (f,) = self.audio_dit.patchify(audio_x, None)
        # audio_x: [1, 403, 1536]
        # f: 403

        audio_freqs = torch.cat(
            [
                self.audio_dit.freqs[0][:f].view(f, -1).expand(f, -1),
                self.audio_dit.freqs[1][:f].view(f, -1).expand(f, -1),
                self.audio_dit.freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1
        ).reshape(f, 1, -1).to(audio_x.device)
        # audio_freqs: [403, 1, 64]

        # visual_dit + audio_dit + bridge forward
        visual_x, audio_x = self.forward_dual_tower_dit(
            visual_dit=visual_dit,
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context_emb,
            audio_context=audio_context_emb,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            visual_freqs=visual_freqs,
            audio_freqs=audio_freqs,
            grid_size=grid_size,
            video_fps=video_fps,
            cp_mesh=cp_mesh,
        )
        
        visual_output = visual_dit.head(visual_x, visual_t)
        visual_output = visual_dit.unpatchify(visual_output, grid_size)  # shape=[B, C=16, T // 4 + 1, H // 8, W // 8]
        
        audio_output = self.audio_dit.head(audio_x, audio_t)
        audio_output = self.audio_dit.unpatchify(audio_output, (f, ))  # [1, 128, 403]

        return visual_output, audio_output


    def _sample_timestep_id(self, timestep_config: TimestepConfig | None = None):
        cfg = timestep_config or TimestepConfig()
        total_timesteps = self.scheduler.num_train_timesteps
        max_timestep_boundary = int(cfg.max_timestep_boundary * total_timesteps)
        min_timestep_boundary = int(cfg.min_timestep_boundary * total_timesteps)
        # Sample timesteps using the SD3 paper's distribution (in [0,1]), then map to [min, max).
        u = compute_density_for_timestep_sampling(
            weighting_scheme=cfg.weighting_scheme,
            batch_size=1,
            logit_mean=cfg.logit_mean,
            logit_std=cfg.logit_std,
            mode_scale=cfg.mode_scale,
            min_timestep_boundary=cfg.min_timestep_boundary,
            max_timestep_boundary=cfg.max_timestep_boundary,
        )
        # Map continuous probabilities to training-step indices and clamp to [min, max).
        timestep_id = torch.floor(u * total_timesteps).to(dtype=torch.long)
        if timestep_id < min_timestep_boundary or timestep_id >= max_timestep_boundary:
            print(f"[WARN] timestep_id {timestep_id} is out of range [{min_timestep_boundary}, {max_timestep_boundary}), clamping to [{min_timestep_boundary}, {max_timestep_boundary - 1})")
            timestep_id = torch.clamp(timestep_id, min=min_timestep_boundary, max=max_timestep_boundary - 1)

        # NOTE(dhyu): Why does DiffSynth cast timestep to bfloat16...?
        return timestep_id

    def sample_timestep_pair(self, timestep_config: TimestepConfig | None = None):
        cfg = timestep_config or TimestepConfig()
        timestep_id = self._sample_timestep_id(cfg)
        base_timestep = self.scheduler.timesteps[timestep_id].to(device=self.device)

        pair_timesteps = getattr(self.scheduler, "pair_timesteps", None)
        if pair_timesteps is None:
            visual_timestep = base_timestep.clone()
            audio_timestep = base_timestep.clone()
            return visual_timestep, audio_timestep

        pair_matrix = self.scheduler.get_pairs("timesteps")
        pair_row = pair_matrix[timestep_id] # [B, 2]
        pair_row = pair_row.to(device=self.device, dtype=base_timestep.dtype)

        visual_timestep = pair_row[:, 0]
        audio_timestep = pair_row[:, 1]

        return visual_timestep, audio_timestep

    def forward_dual_tower_dit(
        self,
        visual_dit,
        visual_x: torch.Tensor,
        audio_x: torch.Tensor,
        visual_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_t_mod: torch.Tensor,
        audio_t_mod: Optional[torch.Tensor],
        visual_freqs: torch.Tensor,
        audio_freqs: torch.Tensor,
        grid_size: tuple[int, int, int],
        video_fps: float,
        condition_scale: Optional[float] = 1.0,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        cp_mesh: Optional[DeviceMesh] = None,
    ):
        min_layers = min(len(visual_dit.blocks), len(self.audio_dit.blocks))
        visual_layers = len(visual_dit.blocks)

        sp_enabled = False
        sp_group = None
        sp_rank = 0
        sp_size = 1
        visual_pad_len = 0
        audio_pad_len = 0

        # Precompute aligned (cos, sin) for cross-modal RoPE (using global sequence length).
        if self.dual_tower_bridge.apply_cross_rope:
            (visual_rope_cos_sin, audio_rope_cos_sin) = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=audio_x.shape[1],
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        if cp_mesh is not None:
            sp_rank = cp_mesh.get_local_rank()
            sp_size = cp_mesh.size()
            sp_group = cp_mesh.get_group()
            visual_x, visual_chunk_len, visual_pad_len, _ = _sp_split_tensor(visual_x, sp_size=sp_size, sp_rank=sp_rank)
            audio_x, audio_chunk_len, audio_pad_len, _ = _sp_split_tensor(audio_x, sp_size=sp_size, sp_rank=sp_rank)
            visual_freqs, _, _, _ = _sp_split_tensor_dim_0(visual_freqs, sp_size=sp_size, sp_rank=sp_rank)
            audio_freqs, _, _, _ = _sp_split_tensor_dim_0(audio_freqs, sp_size=sp_size, sp_rank=sp_rank)
            if visual_rope_cos_sin is not None:
                visual_rope_cos_sin = [
                    _sp_split_tensor(rope_cos_sin, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rope_cos_sin in visual_rope_cos_sin
                ]
            if audio_rope_cos_sin is not None:
                audio_rope_cos_sin = [
                    _sp_split_tensor(rope_cos_sin, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rope_cos_sin in audio_rope_cos_sin
                ]
            # Wan2.2-5B also requires slicing visual_t_mod.
            if len(visual_t_mod.shape) == 4:
                visual_t_mod, _, _, _ = _sp_split_tensor(visual_t_mod, sp_size=sp_size, sp_rank=sp_rank)
            sp_enabled = True
        
        # forward dit blocks and bridges
        def _make_custom_forward(module):
            def _fn(*inputs):
                return module(*inputs)
            return _fn

        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # Audio DiT hidden states condition the visual DiT hidden states.
            if self.dual_tower_bridge.should_interact(layer_idx, 'a2v'):
                if self.use_gradient_checkpointing and self.training:
                    # Create a wrapper that accepts positional args (required by checkpoint)
                    def _bridge_positional(layer_idx_arg, visual_arg, audio_arg):
                        return self.dual_tower_bridge(
                            layer_idx_arg,
                            visual_arg,
                            audio_arg,
                            x_freqs=visual_rope_cos_sin,
                            y_freqs=audio_rope_cos_sin,
                            a2v_condition_scale=a2v_condition_scale,
                            v2a_condition_scale=v2a_condition_scale,
                            condition_scale=condition_scale,
                            video_grid_size=grid_size,
                        )

                    if self.use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            visual_x, audio_x = torch.utils.checkpoint.checkpoint(
                                _bridge_positional,
                                layer_idx,
                                visual_x,
                                audio_x,
                                use_reentrant=False,
                            )
                    else:
                        visual_x, audio_x = torch.utils.checkpoint.checkpoint(
                            _bridge_positional,
                            layer_idx,
                            visual_x,
                            audio_x,
                            use_reentrant=False,
                        )
                else:
                    visual_x, audio_x = self.dual_tower_bridge(
                        layer_idx,
                        visual_x,
                        audio_x,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=audio_rope_cos_sin,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                    )

            # visual block
            if self.use_gradient_checkpointing and self.training:
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        visual_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(visual_block),
                            visual_x, visual_context, visual_t_mod, visual_freqs,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(visual_block),
                        visual_x, visual_context, visual_t_mod, visual_freqs,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)

            # audio block
            if self.use_gradient_checkpointing and self.training:
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        audio_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(audio_block),
                            audio_x, audio_context, audio_t_mod, audio_freqs,
                            use_reentrant=False,
                        )
                else:
                    audio_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(audio_block),
                        audio_x, audio_context, audio_t_mod, audio_freqs,
                        use_reentrant=False,
                    )
            else:
                audio_x = audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)
        
        # forward remaining visual blocks
        assert visual_layers >= min_layers, "visual_layers must be greater than min_layers"
        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_dit.blocks[layer_idx]
            if self.use_gradient_checkpointing and self.training:
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        visual_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(visual_block),
                            visual_x, visual_context, visual_t_mod, visual_freqs,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(visual_block),
                        visual_x, visual_context, visual_t_mod, visual_freqs,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)
        
        if sp_enabled:
            visual_x_full = _sp_all_gather_avg(visual_x, sp_group=sp_group, pad_len=visual_pad_len)
            audio_x_full = _sp_all_gather_avg(audio_x, sp_group=sp_group, pad_len=audio_pad_len)
        else:
            visual_x_full = visual_x
            audio_x_full = audio_x
        
        return visual_x_full, audio_x_full

        
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds


    def forward(self, *args, cp_mesh=None, **kwargs):
        # Ensure FSDP forward hooks take effect.
        return self.training_step(*args, cp_mesh=cp_mesh, **kwargs)
    # ============================================================
    # Training Methods
    # ============================================================
    def training_step(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        first_frame: torch.Tensor,
        caption: list,
        timestep: torch.Tensor = None,
        audio_timestep: torch.Tensor = None,
        video_fps: float = 24.0,
        global_step: int = 0,
        cp_mesh=None,
    ) -> dict:
        """
        Execute one training step, return loss dictionary.
        
        Args:
            video: Video tensor [B, T, C, H, W] or [B, C, T, H, W]
            audio: Audio waveform [B, 1, T_audio]
            first_frame: First frame image [B, C, H, W]
            caption: Text description list
            timestep: Optional fixed time step [B]
            audio_timestep: Optional audio time step [B]
            video_fps: Video frame rate
            
        Returns:
            dict: {"loss": total, "video_loss": ..., "audio_loss": ..., "timestep": ...}
        """
        B = video.shape[0]
        device = video.device
        dtype = self.dtype
        
        # Ensure video format is [B, C, T, H, W]
        if video.dim() == 5 and video.shape[1] != video.shape[2]:
            if video.shape[1] > video.shape[2]:  # [B, T, C, H, W]
                video = video.permute(0, 2, 1, 3, 4)
        
        # Convert data type and device
        video = video.to(dtype=dtype, device=device)
        first_frame = first_frame.to(dtype=dtype, device=device)
        # Audio VAE needs float32
        audio = audio.to(dtype=torch.float32, device=device)
        
        # --------------------------------------------------
        # 1. Encode text
        # --------------------------------------------------
        with torch.no_grad():
            context = self._get_t5_prompt_embeds(caption, device=device)
        
        # --------------------------------------------------
        # 2. Encode video to latent
        # --------------------------------------------------
        with torch.no_grad():
            # Use diffusers VAE API: encode -> latent_dist.mode()
            with torch.autocast("cuda", dtype=dtype):
                # video: [B, C, T, H, W]
                video_latents = self.video_vae.encode(video).latent_dist.mode()
                # Normalize video latents (consistent with inference)
                video_latents = self.normalize_video_latents(video_latents)
        
        # --------------------------------------------------
        # 3. Encode first frame and add mask (consistent with ImageEmbedderVAE)
        # --------------------------------------------------
        with torch.no_grad():
            _, C, num_frames, height, width = video.shape
            H_latent = height // 8
            W_latent = width // 8
            T_latent = video_latents.shape[2]
            
            # Create mask: first frame is 1, others are 0
            # Shape: [B, 4, T_latent, H_latent, W_latent]
            msk = torch.zeros(B, 4, T_latent, H_latent, W_latent, device=device)
            msk[:, :, 0, :, :] = 1  # All channels of first frame are set to 1
            
            # Build first frame input: first frame + zero padding (consistent with ImageEmbedderVAE)
            # first_frame: [B, C, H, W] -> [B, C, T, H, W] with zeros for other frames
            vae_input = torch.concat([
                first_frame.unsqueeze(2),  # [B, C, 1, H, W]
                torch.zeros(B, C, num_frames - 1, height, width, device=device, dtype=dtype)
            ], dim=2)  # [B, C, T, H, W]
            
            # Encode first frame
            with torch.autocast("cuda", dtype=dtype):
                y = self.video_vae.encode(vae_input).latent_dist.mode()
                y = self.normalize_video_latents(y)
            
            # Concat mask and encoded first frame: [B, 4+16, T', H', W'] = [B, 20, T', H', W']
            y = torch.concat([msk.to(dtype=dtype), y], dim=1)  # [B, 20, T', H', W']
        
        # --------------------------------------------------
        # 4. Encode audio to latent
        # --------------------------------------------------
        with torch.no_grad():
            # Audio VAE uses float32
            with torch.autocast("cuda", dtype=torch.float32):
                if self.audio_vae_type == "dac":
                    x_pad = self.audio_vae.preprocess(audio, sample_rate=self.sample_rate)
                    z, codes, latents, commitment_loss, codebook_loss = self.audio_vae.encode(x_pad)
                    audio_latents = z.mode()
                else:
                    audio_latents = self.audio_vae.encode(audio).latent_dist.sample()
            # Convert back to training dtype
            audio_latents = audio_latents.to(dtype=dtype)
        
        # --------------------------------------------------
        # 5. Sample time step
        # --------------------------------------------------
        timestep_config = TimestepConfig(
            max_timestep_boundary=1.0,
            min_timestep_boundary=0.0,
            weighting_scheme="uniform",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.0,
            independent_timesteps=False,
        )
        # Look up the index boundary directly from the scheduler's timestep table.
        # boundary_ratio=0.9 means timestep 900; count how many timesteps >= 900 to get the index fraction.
        boundary = (self.scheduler.timesteps >= self.boundary_ratio * self.scheduler.num_train_timesteps).sum().item() / self.scheduler.num_train_timesteps
        if global_step % 2 == 0:
            timestep_config.max_timestep_boundary = boundary
        else:
            timestep_config.min_timestep_boundary = boundary

        timestep, audio_timestep = self.sample_timestep_pair(timestep_config)
        timestep = timestep.to(device=device)
        audio_timestep = audio_timestep.to(device=device)

        
        # --------------------------------------------------
        # 6. Add noise (Flow Matching)
        # --------------------------------------------------
        video_noise = torch.randn_like(video_latents)
        audio_noise = torch.randn_like(audio_latents)

        noisy_video = self.scheduler.add_noise(video_latents, video_noise, timestep).to(device=device)
        noisy_audio = self.scheduler.add_noise(audio_latents, audio_noise, audio_timestep).to(device=device)
        # --------------------------------------------------
        # 7. Forward pass
        # --------------------------------------------------
        # Select DiT (based on time step to determine high noise or low noise stage)
        if global_step % 2 == 0:
            cur_visual_dit = self.video_dit
        else:
            cur_visual_dit = self.video_dit_2
        
        video_pred, audio_pred = self.inference_single_step(
            visual_dit=cur_visual_dit,
            visual_latents=noisy_video,
            audio_latents=noisy_audio,
            y=y,
            context=context,
            timestep=timestep.unsqueeze(0) if timestep.dim() == 0 else timestep[:1],
            audio_timestep=audio_timestep.unsqueeze(0) if audio_timestep.dim() == 0 else audio_timestep[:1],
            video_fps=video_fps,
            cp_mesh=cp_mesh,
        )
        
        # --------------------------------------------------
        # 8. Calculate loss
        # --------------------------------------------------
        # Flow Matching target: v = noise - sample
        video_target = video_noise - video_latents
        audio_target = audio_noise - audio_latents
        
        # MSE Loss (ensure prediction dtype matches target dtype, avoid mixing fp32/bf16)
        video_loss = torch.nn.functional.mse_loss(video_pred.to(video_target.dtype), video_target)
        audio_loss = torch.nn.functional.mse_loss(audio_pred.to(audio_target.dtype), audio_target)
        
        # Total loss
        total_loss = video_loss + audio_loss
        
        return {
            "loss": total_loss,
            "video_loss": video_loss,
            "audio_loss": audio_loss,
            "timestep": timestep.mean().item(),
        }
    
    def get_trainable_parameters(self, train_modules: list = None):
        """
        Get trainable parameters
        
        Args:
            train_modules: List of module names to train, e.g. ["video_dit", "audio_dit", "dual_tower_bridge"]
                           If None, train all modules
        """
        if train_modules is None:
            train_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        
        params = []
        for name in train_modules:
            if hasattr(self, name):
                module = getattr(self, name)
                params.extend(module.parameters())
                print(f"[Train] {name}: {sum(p.numel() for p in module.parameters())/1e6:.2f}M params")
        
        return params
    
    def freeze_for_training(self, train_modules: list = None):
        """
        Freeze modules that are not needed for training
        """
        if train_modules is None:
            train_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        
        # First freeze all
        for name, module in self.named_children():
            module.eval()
            module.requires_grad_(False)
        
        # Unfreeze modules that need to be trained
        for name in train_modules:
            if hasattr(self, name):
                module = getattr(self, name)
                module.train()
                module.requires_grad_(True)
                print(f"[Unfreeze] {name}")
    
    def save_trainable_weights(self, save_path: str, train_modules: list = None):
        """
        Save weights of trainable modules
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if train_modules is None:
            train_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        
        for name in train_modules:
            if hasattr(self, name):
                module = getattr(self, name)
                module_path = os.path.join(save_path, name)
                os.makedirs(module_path, exist_ok=True)
                
                # Save as HuggingFace format
                if hasattr(module, "save_pretrained"):
                    module.save_pretrained(module_path)
                else:
                    torch.save(module.state_dict(), os.path.join(module_path, "pytorch_model.bin"))
                
                print(f"[Save] {name} -> {module_path}")

def sanitize_hf_device_map(module: torch.nn.Module):
    visited = set()

    def _visit(m):
        if id(m) in visited:
            return
        visited.add(id(m))

        try:
            if hasattr(m, "hf_device_map") and m.hf_device_map is None:
                setattr(m, "hf_device_map", {"":"cpu"})
        except Exception:
            pass

        for child in m.children():
            _visit(child)

    _visit(module)


@DIFFUSION_PIPELINES.register_module()
def MOVATrain_from_pretrained(
    from_pretrained: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    import os
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    
    if use_deepspeed:
        # For DeepSpeed ZeRO-3: try to use empty weights + dispatch to avoid GPU spike
        try:
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            import json
            
            print("[DeepSpeed] Using init_empty_weights to avoid GPU memory spike during loading")
            
            # Load config to get model structure
            config = MOVATrain.load_config(from_pretrained)
            
            # Create empty model (meta tensors, no actual memory)
            with init_empty_weights():
                model = MOVATrain.from_config(config)
            
            # Load weights with CPU offload
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=from_pretrained,
                device_map="cpu",  # Keep everything on CPU for DeepSpeed to manage
                dtype=torch_dtype,
            )
            print("[DeepSpeed] Model loaded with empty weights + dispatch")
        except Exception as e:
            print(f"[DeepSpeed] init_empty_weights failed ({e}), falling back to standard loading")
            model = MOVATrain.from_pretrained(
                from_pretrained, 
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
    elif use_fsdp or device == "cpu":
        # FSDP mode or CPU mode: load on CPU without device_map
        # accelerate does not allow models with device_map for distributed training
        if use_fsdp:
            print("[FSDP] Loading model on CPU without device_map for distributed training compatibility")
        model = MOVATrain.from_pretrained(
            from_pretrained, 
            torch_dtype=torch_dtype,
            # low_cpu_mem_usage=True,
        )
    else:
        model = MOVATrain.from_pretrained(
            from_pretrained, 
            device_map=device, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    
    sanitize_hf_device_map(model)
    # Set gradient checkpointing flags after loading
    model.use_gradient_checkpointing = use_gradient_checkpointing
    model.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
    return model