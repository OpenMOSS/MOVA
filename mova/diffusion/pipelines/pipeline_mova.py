import copy
import gc
import html
import os
import re
from contextlib import nullcontext
from functools import partial
from typing import Any, List, Literal, Optional, Tuple, Union

import ftfy
import torch
import torch.distributed as dist
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from transformers.models import T5TokenizerFast, UMT5EncoderModel
from yunchang.kernels import AttnType

from mova.diffusion.models import (WanAudioModel, WanModel,
                                   sinusoidal_embedding_1d)
from mova.diffusion.models.dac_vae import DAC
from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge
from mova.distributed.functional import (_sp_all_gather_avg, _sp_split_tensor,
                                         _sp_split_tensor_dim_0)
from mova.registry import DIFFUSION_PIPELINES, DIFFUSION_SCHEDULERS, MODELS
from mova.utils.misc import gpu_timer, track_gpu_mem


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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class MOVA(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->video_dit->video_dit_2->video_vae->audio_vae"

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

        # build video vae
        self.vae_scale_factor_spatial = self.video_vae.config.scale_factor_spatial
        self.vae_scale_factor_temporal = self.video_vae.config.scale_factor_temporal
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        
        # build audio vae
        self.audio_vae_scale_factor = int(self.audio_vae.hop_length)
        self.audio_sample_rate = self.audio_vae.sample_rate

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


    def check_inputs(
        self,
        height,
        width,
        num_frames,
    ):
        target_division_factor = self.vae_scale_factor_spatial * 2
        if height % target_division_factor != 0 or width % target_division_factor != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {target_division_factor} but are {height} and {width}.")
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            raise ValueError(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal} but is {num_frames - 1}."
            )
    
    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)  # [batch_size, channels, 1, height, width]

        if last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.video_vae.dtype)

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.video_vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.video_vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = self.normalize_video_latents(latent_condition)

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)
    
    def prepare_audio_latents(
        self,
        audio: Optional[torch.Tensor],
        batch_size: int,
        num_channels: int,
        num_samples: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        latent_t = (num_samples - 1) // self.audio_vae_scale_factor + 1
        shape = (batch_size, num_channels, latent_t)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        return latents
    
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

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        image,
        negative_prompt="",
        seed=42,
        height=360,
        width=640,
        num_frames=193,
        video_fps=24.0,
        visual_shift=5.0,
        audio_shift=5.0,
        # scheduler
        num_inference_steps=50,
        sigma_shift=5.0,
        # cfg
        cfg_scale=5.0,
        cp_mesh=None,
        remove_video_dit=False,
    ):

        # 1. check inputs. Raise ValueError if inputs are invalid.
        self.check_inputs(height, width, num_frames)

        denoising_strength = 1.0
        cfg_mode = "text"
        cfg_merge = False
        audio_num_samples = int(self.audio_sample_rate * num_frames / video_fps)

        device = self._execution_device
        # Generate on CPU first so seeded noise is reproducible across execution devices.
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
            device=device,
        )
        scheduler_support_audio = hasattr(self.scheduler, "get_pairs")
        if scheduler_support_audio:
            audio_scheduler = self.scheduler
            paired_timesteps = self.scheduler.get_pairs()
        else:
            audio_scheduler = copy.deepcopy(self.scheduler)
            paired_timesteps = torch.stack([self.scheduler.timesteps, self.scheduler.timesteps], dim=1)

        # 5. Prepare latent variables
        num_channels_latents = self.video_vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        latents, condition = self.prepare_latents(
            image,
            1,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator=generator,
            latents=None,
            last_image=None,
        )
        audio_latents = self.prepare_audio_latents(
            None,
            1,
            self.audio_vae.latent_dim,
            audio_num_samples,
            torch.float32,
            device,
            generator=generator,
            latents=None,
        )

        prompt_embeds = self._get_t5_prompt_embeds(prompt)
        negative_prompt_embeds = self._get_t5_prompt_embeds(negative_prompt)

        # --------------------------------------------------
        # diffusion steps
        # --------------------------------------------------
        cur_visual_dit = self.video_dit
        total_steps = paired_timesteps.shape[0]
        switched = False
        boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps

        for idx_step in tqdm(range(total_steps), disable=dist.get_rank() != 0):
            timestep, audio_timestep = paired_timesteps[idx_step]

            # print(f"timestep: {timestep}, audio_timestep: {audio_timestep}")

            # switch to low-noise DiT
            if not switched and timestep.item() < boundary_timestep:
                cur_visual_dit = self.video_dit_2
                if remove_video_dit:
                    self.video_dit = None
                    gc.collect()
                switched = True

            latent_model_input = torch.cat([latents, condition], dim=1) # TODO: to dtype
            # timestep
            timestep = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_timestep = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)

            # Inference a single step
            # with gpu_timer("inference_single_step_posi"):
            with nullcontext():
                noise_pred_posi = self.inference_single_step(
                    visual_dit=cur_visual_dit,
                    visual_latents=latent_model_input,  # video noise
                    audio_latents=audio_latents,  # audio noise
                    context=prompt_embeds,  # prompt embedding
                    timestep=timestep,
                    audio_timestep=audio_timestep,
                    video_fps=video_fps,
                    cp_mesh=cp_mesh,
                )
            if cfg_scale == 1.0 and "dual" not in cfg_mode:
                visual_noise_pred = noise_pred_posi[0].float()
                audio_noise_pred = noise_pred_posi[1].float()
            elif "dual" not in cfg_mode:
                if cfg_merge:
                    visual_noise_pred_posi, visual_noise_pred_nega = noise_pred_posi[0].float().chunk(2, dim=0)
                    audio_noise_pred_posi, audio_noise_pred_nega = noise_pred_posi[1].float().chunk(2, dim=0)
                else:
                    noise_pred_nega = self.inference_single_step(
                        visual_dit=cur_visual_dit,
                        visual_latents=latent_model_input,
                        audio_latents=audio_latents,
                        context=negative_prompt_embeds,
                        timestep=timestep, 
                        audio_timestep=audio_timestep,
                        video_fps=video_fps,
                        cp_mesh=cp_mesh,
                    )
                    visual_noise_pred_nega, audio_noise_pred_nega = noise_pred_nega[0].float(), noise_pred_nega[1].float()
                    visual_noise_pred_posi, audio_noise_pred_posi = noise_pred_posi[0].float(), noise_pred_posi[1].float()
                visual_noise_pred = visual_noise_pred_nega + cfg_scale * (visual_noise_pred_posi - visual_noise_pred_nega)
                audio_noise_pred = audio_noise_pred_nega + cfg_scale * (audio_noise_pred_posi - audio_noise_pred_nega)
            else:
                raise NotImplementedError

            # move a step
            if scheduler_support_audio:
                next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
                next_audio_timestep = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
                latents = self.scheduler.step_from_to(
                    visual_noise_pred,
                    timestep,
                    next_timestep,
                    latents,
                )
                audio_latents = audio_scheduler.step_from_to(
                    audio_noise_pred,
                    audio_timestep,
                    next_audio_timestep,
                    audio_latents,
                )
            else:
                latents = self.scheduler.step(visual_noise_pred, timestep, latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(audio_noise_pred, audio_timestep, audio_latents, return_dict=False)[0]
        
        # decode video
        video_latents = self.denormalize_video_latents(latents)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = self.video_vae.decode(video_latents).sample
        video = self.video_processor.postprocess_video(video, output_type="pil")

        # decode audio
        with torch.autocast("cuda", dtype=torch.float32):
            audio = self.audio_vae.decode(audio_latents)  # [B, 1, T]

        return video, audio

    def _pre_forward(self, model):
        if hasattr(model, "_hf_hook") and hasattr(model._hf_hook, "pre_forward"):
            model._hf_hook.pre_forward(model)

    def inference_single_step(
        self,
        visual_dit,
        visual_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor],
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

        self._pre_forward(visual_dit)
        self._pre_forward(self.audio_dit)
        self._pre_forward(self.dual_tower_bridge)

        visual_x = visual_latents
        audio_x = audio_latents
        audio_context = visual_context = context  # [B, 512, C=4096]

        if audio_timestep is None:
            audio_timestep = timestep

        # t: timestep_embedding
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = visual_dit.time_embedding(sinusoidal_embedding_1d(visual_dit.freq_dim, timestep))  # [B, C=5120]
            visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))  # [B, 6, C=5120]

            audio_t = self.audio_dit.time_embedding(sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep))
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))
        
        model_dtype = visual_dit.dtype  # bf16
        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)
        
        # prompt embedding
        visual_context_emb = visual_dit.text_embedding(visual_context)  # shape=[B, 512, C=5120]; dtype=bf16
        audio_context_emb = self.audio_dit.text_embedding(audio_context)  # shape=[B, 512, C=1536]; dtype=bf16
        
        visual_x = visual_latents.to(model_dtype)
        audio_x = audio_latents.to(model_dtype)

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
        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # Audio DiT hidden states condition the visual DiT hidden states.
            if self.dual_tower_bridge.should_interact(layer_idx, 'a2v'):
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

            visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)
            audio_x = audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)
        
        # forward remaining visual blocks
        assert visual_layers >= min_layers, "visual_layers must be greater than min_layers"
        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_dit.blocks[layer_idx]
            visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)
        
        if sp_enabled:
            visual_x_full = _sp_all_gather_avg(visual_x, sp_group=sp_group, pad_len=visual_pad_len)
            audio_x_full = _sp_all_gather_avg(audio_x, sp_group=sp_group, pad_len=audio_pad_len)
        else:
            visual_x_full = visual_x
            audio_x_full = audio_x
        
        return visual_x_full, audio_x_full
