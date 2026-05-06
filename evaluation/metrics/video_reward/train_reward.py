"""
Model creation utilities for VideoReward evaluation.

Provides `create_model_and_processor` to load a Qwen2-VL model with LoRA
adapters and a reward model head (rm_head), along with the matching processor.
"""

import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

from utils import ModelConfig, PEFTLoraConfig, TrainingConfig

# Reward special tokens
REWARD_SPECIAL_TOKENS = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    """Find the target linear modules for LoRA, matching the original training code's logic."""
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def create_model_and_processor(
    model_config: ModelConfig,
    peft_lora_config: PEFTLoraConfig,
    training_args: TrainingConfig = None,
):
    """Create the Qwen2-VL reward model with LoRA adapters and the processor.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration (includes model_name_or_path, output_dim, etc.)
    peft_lora_config : PEFTLoraConfig
        LoRA configuration (lora_r, lora_alpha, target modules, etc.)
    training_args : TrainingConfig, optional
        Training arguments (used for dtype, device, and flash attention settings).

    Returns
    -------
    model : PeftModel
        The model with LoRA adapters and rm_head applied.
    processor : AutoProcessor
        The matching processor/tokenizer.
    peft_config : LoraConfig
        The PEFT LoRA configuration used.
    """
    from peft import LoraConfig, get_peft_model

    model_name_or_path = model_config.model_name_or_path

    # Resolve local model path:
    # 1. If QWEN2VL_LOCAL_PATH env var is set and points to a dir, use that
    # 2. If model_name_or_path is already a local dir, use as-is
    # 3. Otherwise, keep the HF hub name (requires offline cache or network)
    local_path = os.environ.get("QWEN2VL_LOCAL_PATH", "")
    if local_path and os.path.isdir(local_path):
        model_name_or_path = local_path
        print(f"[INFO] Using local Qwen2-VL from QWEN2VL_LOCAL_PATH: {local_path}")
    elif not os.path.isdir(model_name_or_path):
        print(f"[INFO] model_name_or_path '{model_name_or_path}' is not a local directory; "
              f"will try HF cache or set QWEN2VL_LOCAL_PATH env var")

    # Determine dtype
    if model_config.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    # Determine attn_implementation: use flash_attention_2 unless explicitly disabled
    use_flash_attn = not (training_args and training_args.disable_flash_attn2)
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    # Allow model_config.attn_implementation to override if explicitly set
    if model_config.attn_implementation is not None:
        attn_impl = model_config.attn_implementation

    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Load processor from local checkpoint if available, otherwise from model path
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    # Add special reward tokens
    if model_config.use_special_tokens:
        tokenizer = processor.tokenizer
        num_added = tokenizer.add_tokens(REWARD_SPECIAL_TOKENS, special_tokens=True)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"[INFO] Added {num_added} reward special tokens, vocab size: {len(tokenizer)}")

    # Add reward model head (rm_head) — one scalar per reward dimension
    output_dim = model_config.output_dim  # typically 1
    hidden_size = model.config.hidden_size
    rm_head = torch.nn.Linear(hidden_size, output_dim, bias=False)
    model.rm_head = rm_head

    # Convert to target dtype before applying LoRA
    if training_args and training_args.bf16:
        model.to(torch.bfloat16)
    if training_args and training_args.fp16:
        model.to(torch.float16)

    # Configure LoRA: use find_target_linear_names to match original training code
    lora_target_modules = peft_lora_config.lora_target_modules
    if isinstance(lora_target_modules, str):
        lora_target_modules = [lora_target_modules]

    lora_namespan_exclude = peft_lora_config.lora_namespan_exclude
    if isinstance(lora_namespan_exclude, str):
        lora_namespan_exclude = [lora_namespan_exclude]

    # When lora_target_modules is None, discover all linear/embedding modules
    # and exclude those matching lora_namespan_exclude — matching original training code
    if lora_target_modules is None:
        lora_target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=lora_namespan_exclude or [],
        )

    peft_config = LoraConfig(
        r=peft_lora_config.lora_r,
        lora_alpha=peft_lora_config.lora_alpha,
        lora_dropout=peft_lora_config.lora_dropout,
        target_modules=lora_target_modules,
        modules_to_save=peft_lora_config.lora_modules_to_save,
        task_type=peft_lora_config.lora_task_type,
        use_rslora=peft_lora_config.use_rslora,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Freeze/unfreeze components as configured
    if model_config.freeze_vision_tower:
        for param in model.base_model.model.visual.parameters():
            param.requires_grad = False

    if model_config.freeze_llm:
        for param in model.base_model.model.model.parameters():
            param.requires_grad = False

    if not model_config.tune_merger:
        for param in model.base_model.model.visual.merger.parameters():
            param.requires_grad = False

    # Print trainable params
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[INFO] Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    return model, processor, peft_config
