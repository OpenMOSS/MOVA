"""
Accelerate + FSDP Trainer
Supports:   
- Automatic mixed precision (AMP)
- Fully sharded data parallel (FSDP)
- Gradient checkpointing
- DeepSpeed
"""

import os, re
import torch
from tqdm import tqdm

try:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("[Warning] accelerate not installed. Run: pip install accelerate")

from mova.registry import OPTIMIZERS
from mova.engine.trainer.utils.logger import build_logger


class AccelerateTrainer:
    """
    Trainer based on HuggingFace Accelerate
    
    Features:
    - Automatic handling of distributed training
    - Supports FSDP / DeepSpeed
    - Mixed precision training
    - Gradient accumulation
    - Gradient checkpointing
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        optimizer_cfg: dict,
        # Training params
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float = 1.0,
        # Mixed precision
        mixed_precision: str = "bf16",  # "no", "fp16", "bf16"
        # FSDP config
        use_fsdp: bool = False,
        fsdp_config: dict = None,
        # Logging
        log_interval: int = 10,
        logger_type: str = "wandb",
        logger_kwargs: dict = None,
        # Checkpointing
        save_interval: int = 1000,
        save_path: str = "./checkpoints",
        resume_from: str = None,
        # LR Scheduler
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
        # Training modules
        train_modules: list = None,
        # LoRA
        use_lora: bool = False,
        lora_config: dict = None,
        # Context Parallel
        enable_cp: bool = False,
    ):
        if not ACCELERATE_AVAILABLE:
            raise RuntimeError("Please install accelerate: pip install accelerate")
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer_cfg = optimizer_cfg
        
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        
        self.mixed_precision = mixed_precision
        self.use_fsdp = use_fsdp
        self.fsdp_config = fsdp_config
        
        self.log_interval = log_interval
        self.logger_type = logger_type
        self.logger_kwargs = logger_kwargs or {}
        
        self.save_interval = save_interval
        self.save_path = save_path
        self.resume_from = resume_from
        
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
        self.train_modules = train_modules or ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        
        self.use_lora = use_lora
        self.lora_config = lora_config
        self.enable_cp = enable_cp
        
        # Initialize
        self.global_step = 0
        self.epoch = 0
        self._setup()
        
    def _find_latest_checkpoint(self):
        """
        Find the latest step-* checkpoint in self.save_path
        Return: checkpoint path or None
        """
        if not os.path.isdir(self.save_path):
            return None

        step_pattern = re.compile(r"^step-(\d+)$")

        candidates = []

        for name in os.listdir(self.save_path):
            match = step_pattern.match(name)
            if match:
                step = int(match.group(1))
                full_path = os.path.join(self.save_path, name)
                if os.path.isdir(full_path):
                    candidates.append((step, full_path))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    
    def _setup(self):
        """Initialize training components"""
        ignored_modules = []
        # LoRA configuration
        if self.use_lora:
            self._setup_lora()
        else:
            self.model.freeze_for_training(self.train_modules)
        
        fsdp_plugin = None
        if self.use_fsdp:
            from accelerate import FullyShardedDataParallelPlugin
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                ShardedOptimStateDictConfig,
                    ShardedStateDictConfig,
                )
            
            for attr_name in ['text_encoder', 'prompter', 'video_vae', 'audio_vae']:
                if hasattr(self.model, attr_name):
                    module = getattr(self.model, attr_name)
                    if isinstance(module, torch.nn.Module):
                        ignored_modules.append(module)
            if self.use_lora and hasattr(self, "lora_layers"):
                for lora_module in self.lora_layers.values():
                    for sub in lora_module.modules():
                        ignored_modules.append(sub)
            
            self._ignored_modules = ignored_modules if ignored_modules else None
            if self.fsdp_config is not None:
                fsdp_config = dict(self.fsdp_config)
                if ignored_modules:
                    fsdp_config['ignored_modules'] = ignored_modules
                # FSDP2 requires reshard_after_forward to be a bool, not None
                if 'reshard_after_forward' not in fsdp_config or fsdp_config['reshard_after_forward'] is None:
                    fsdp_config['reshard_after_forward'] = True
                
                fsdp_plugin = FullyShardedDataParallelPlugin(
                    state_dict_config=ShardedStateDictConfig(
                        offload_to_cpu=True,
                    ),
                    optim_state_dict_config=ShardedOptimStateDictConfig(
                        offload_to_cpu=True,
                    ),
                    **fsdp_config
                )
        # Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=None,  
            fsdp_plugin=fsdp_plugin,
        )
        from accelerate.utils import set_seed
        set_seed(42)
        
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        
        if self.is_main_process and ignored_modules:
            print(f"[FSDP] Excluding {len(ignored_modules)} modules from FSDP wrapping: "
                    f"{[m.__class__.__name__ for m in ignored_modules]}")
        # Context Parallel configuration
        self.cp_mesh = None
        if self.enable_cp:
            from yunchang import set_seq_parallel_pg
            import torch.distributed as dist
            self.cp_mesh = self.accelerator.torch_device_mesh['cp']
            cp_size = self.cp_mesh.size()
            # FIXME(dhyu): Due to WanAudio-1.3B, the maximum value is 4
            MAX_ULYSSES_DEGREE = 4
            sp_ulysses_degree = min(MAX_ULYSSES_DEGREE, cp_size)
            sp_ring_degree = cp_size // sp_ulysses_degree
            assert sp_ring_degree * sp_ulysses_degree == cp_size, (
                f"sp_ring_degree * sp_ulysses_degree != cp_size: {sp_ring_degree} * {sp_ulysses_degree} != {cp_size}"
            )
            set_seq_parallel_pg(
                sp_ulysses_degree,
                sp_ring_degree,
                self.accelerator.process_index,
                self.accelerator.num_processes,
                use_ulysses_low=True,
            )
            print(f"{dist.get_backend(self.cp_mesh.get_group()) = }")
            print(f"sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}")
            print(f"Replaced {self.model.replace_attention()} AttentionModules.")
        
        if self.use_fsdp:
            if self.is_main_process:
                print("[FSDP] Moving model to CPU before FSDP wrapping to reduce memory usage")
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
        
        if self.use_lora:
            trainable_params = self._get_lora_params()
        else:
            trainable_params = self.model.get_trainable_parameters(self.train_modules)
        
        self.optimizer = OPTIMIZERS.build(
            self.optimizer_cfg,
            default_args={"params": trainable_params}
        )
        
        self.lr_scheduler = self._build_lr_scheduler()
        
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        
        if self.use_fsdp and hasattr(self, '_ignored_modules') and self._ignored_modules:
            cpu_offload = self.accelerator.state.fsdp_plugin.cpu_offload
            target_device = torch.device('cpu') if cpu_offload else self.device
            for module in self._ignored_modules:
                module.to(target_device)
                if self.is_main_process:
                    print(f"[FSDP] Moved ignored module '{module.__class__.__name__}' to device {target_device}")
        
        self.logger = build_logger(
            logger_type=self.logger_type,
            is_main_process=self.is_main_process,
            **self.logger_kwargs
        )
        
        resume_path = None

        if self.resume_from is not None:
            resume_path = self.resume_from
        else:
            resume_path = self._find_latest_checkpoint()

        if resume_path is not None:
            self.accelerator.print(
                f"[Resume] Loading checkpoint from: {resume_path}"
            )
            self._resume_checkpoint(resume_path)
        else:
            self.accelerator.print("[Resume] No checkpoint found, training from scratch.")
        
        if self.is_main_process:
            os.makedirs(self.save_path, exist_ok=True)
        
        # torch.cuda.memory._record_memory_history(enabled='all')
    
    def _setup_lora(self):
        """Configure LoRA"""
        from mova.engine.trainer.accelerate.lora_utils import inject_lora_to_model
        
        lora_config = self.lora_config or {}
        lora_rank = lora_config.get("rank", 16)
        lora_alpha = lora_config.get("alpha", 16)
        lora_dropout = lora_config.get("dropout", 0.0)
        target_modules = lora_config.get("target_modules", ["q", "k", "v", "o"])
        
        for module_name in self.train_modules:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                inject_lora_to_model(
                    module,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_modules=target_modules,
                )
                print(f"[LoRA] Injected to {module_name}")
        
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
    
    def _get_lora_params(self):
        """Get LoRA parameters"""
        params = []
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                params.append(param)
        
        if self.is_main_process:
            total = sum(p.numel() for p in params)
            print(f"[LoRA] Trainable params: {total / 1e6:.2f}M")
        
        return params

    def _build_lr_scheduler(self):
        """Build learning rate scheduler"""
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
        )
        
        if self.lr_scheduler_type == "constant":
            return ConstantLR(self.optimizer, factor=1.0)
        
        elif self.lr_scheduler_type == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps - self.warmup_steps,
                eta_min=self.min_lr
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        
        elif self.lr_scheduler_type == "linear":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.min_lr / self.optimizer_cfg.get("lr", 1e-4),
                total_iters=self.max_steps - self.warmup_steps
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.warmup_steps]
            )
        
        else:
            raise ValueError(f"Unknown lr_scheduler_type: {self.lr_scheduler_type}")
    
    def train(self):
        """Main training loop"""
        self.model.train()
        if self.is_main_process:
            print(f"Starting training from step {self.global_step}")
            print(f"Max steps: {self.max_steps}")
            print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"Mixed precision: {self.mixed_precision}")
            print(f"FSDP: {self.use_fsdp}")
            print(f"LoRA: {self.use_lora}")
        
        data_iter = iter(self.train_dataloader)
        
        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            disable=not self.is_main_process
        )
        
        accumulated_loss = 0.0
        accumulated_video_loss = 0.0
        accumulated_audio_loss = 0.0
        log_count = 0
        
        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            with self.accelerator.accumulate(self.model):
                loss_dict = self.model(
                    video=batch["video"],
                    audio=batch["audio"],
                    first_frame=batch["first_frame"],
                    caption=batch["caption"],
                    global_step=self.global_step,
                    cp_mesh=self.cp_mesh
                )
                
                loss = loss_dict["loss"]
                
                self.accelerator.backward(loss)
                
                if self.gradient_clip_norm > 0:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm
                        )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            accumulated_loss += loss_dict["loss"].item()
            accumulated_video_loss += loss_dict["video_loss"].item()
            accumulated_audio_loss += loss_dict["audio_loss"].item()
            log_count += 1
            
            self.global_step += 1
            
            if self.global_step % self.log_interval == 0:
                avg_loss = accumulated_loss / log_count
                avg_video_loss = accumulated_video_loss / log_count
                avg_audio_loss = accumulated_audio_loss / log_count
                lr = self.optimizer.param_groups[0]["lr"]
                
                metrics = {
                    "train/loss": avg_loss,
                    "train/video_loss": avg_video_loss,
                    "train/audio_loss": avg_audio_loss,
                    "train/lr": lr,
                    "train/epoch": self.epoch,
                }
                
                self.logger.log(metrics, step=self.global_step)
                
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "v_loss": f"{avg_video_loss:.4f}",
                    "a_loss": f"{avg_audio_loss:.4f}",
                    "lr": f"{lr:.2e}",
                })
                
                accumulated_loss = 0.0
                accumulated_video_loss = 0.0
                accumulated_audio_loss = 0.0
                log_count = 0
            
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint()
            
            pbar.update(1)
        
        self._save_checkpoint(final=True)
        pbar.close()
        self.logger.finish()
        
        if self.is_main_process:
            print(f"Training completed at step {self.global_step}")
    
    def _save_checkpoint(self, final: bool = False):
        """Save checkpoint"""
        self.accelerator.wait_for_everyone()
        
        step_dir = os.path.join(
            self.save_path,
            "final" if final else f"step-{self.global_step}"
        )
        
        if self.is_main_process:
            os.makedirs(step_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if self.use_lora:
            from mova.engine.trainer.accelerate.lora_utils import save_lora_weights_fsdp
            
            if self.use_fsdp:
                full_state_dict = self.accelerator.get_state_dict(self.model)
                save_lora_weights_fsdp(
                    full_state_dict, 
                    step_dir, 
                    self.train_modules,
                    is_main_process=self.is_main_process
                )
            else:
                from mova.engine.trainer.accelerate.lora_utils import save_lora_weights
                save_lora_weights(unwrapped_model, step_dir, self.train_modules)
        else:
            unwrapped_model.save_trainable_weights(step_dir, self.train_modules)
        
        if self.is_main_process:
            state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
            }
            torch.save(state, os.path.join(step_dir, "trainer_state.pt"))
            print(f"[Checkpoint] Saved to {step_dir}")
            os.makedirs(os.path.join(step_dir, "accelerator"), exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(os.path.join(step_dir, "accelerator"))
        if self.use_fsdp:
            rank = self.accelerator.process_index
            accel_dir = os.path.join(step_dir, "accelerator")
            os.makedirs(accel_dir, exist_ok=True)
            torch.save(self.optimizer.state_dict(),
               os.path.join(accel_dir, f"optimizer_{rank}.bin"))
            torch.save(self.lr_scheduler.state_dict(),
               os.path.join(accel_dir, f"scheduler_{rank}.bin"))
        else:
            self.accelerator.save_state(os.path.join(step_dir, "accelerator"))

    
    def _resume_checkpoint(self, checkpoint_path: str):
        """Resume checkpoint"""
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
        
        accelerator_path = os.path.join(checkpoint_path, "accelerator")
        if os.path.exists(accelerator_path):
            if self.use_fsdp:
                rank = self.accelerator.process_index
                optim_path = os.path.join(accelerator_path, f"optimizer_{rank}.bin")
                sched_path = os.path.join(accelerator_path, f"scheduler_{rank}.bin")
                if os.path.exists(optim_path):
                    self.optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
                if os.path.exists(sched_path):
                    self.lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
            else:
                self.accelerator.load_state(accelerator_path)


        if self.use_lora:
            from mova.engine.trainer.accelerate.lora_utils import load_lora_weights
            load_lora_weights(self.model, checkpoint_path)
        
        if self.is_main_process:
            print(f"[Resume] Loaded from {checkpoint_path}, step={self.global_step}")

