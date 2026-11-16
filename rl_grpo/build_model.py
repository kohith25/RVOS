from __future__ import annotations
#!/usr/bin/env python3
"""
rl_grpo/build_model.py - Model Builder for Ref-Thinker Policy and Reference Models

Implements the exact initialization and weight-loading sequence from
`scripts/finetune/inference_hyper_lora.py` lines 85–238:
1. Base config via LlamaConfig or Qwen2Config (NOT generic AutoConfig).
2. Standard peft LoRA when use_hyper_lora=False (all 7 linear target modules),
   or peft_hyper when use_hyper_lora=True.
3. Multimodal modules initialized with full kwargs matching inference_hyper_lora.py.
4. Tokenizer initialized and multimodal special tokens registered.
5. Loads finetune_weights.bin from ckpt_dir and non_lora_trainables.bin from avs_ckpt_dir.
6. Key validation: asserts unexpected_keys is empty and missing_keys contains zero
   trainable/adapter/projector/qformer keys.
7. Reference model: supports deepcopy or peft_disable mode.
8. Mock mode: ONLY runs when use_mock=True. When use_mock=False and weights/files are
   missing, raises clear FileNotFoundError (never silently falls back to mock).
"""

import os
import copy
from typing import Tuple, Optional, Any, List
import contextlib

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def validate_checkpoint_keys(
    missing_keys: List[str],
    unexpected_keys: List[str],
    loaded_keys: Optional[set] = None,
    model_keys: Optional[set] = None,
) -> None:
    """
    Validates checkpoint keys once on the union of loaded keys from
    finetune_weights.bin and non_lora_trainables.bin.
    - Raises if critical trainable keys are missing.
    - Raises if expected trainable keys are missing from loaded weights.
    - Warns if unexpected non-critical keys appear; raises on critical unexpected keys.
    """
    critical_substrings = [
        "lora",
        "projector",
        "qformer",
        "visual_query",
        "audio_query",
        "query_generator",
        "mask_decoder",
    ]

    bad_missing = [
        k for k in missing_keys
        if any(crit in k.lower() for crit in critical_substrings)
    ]
    if bad_missing:
        raise RuntimeError(
            f"State dict mismatch: critical trainable keys are missing ({len(bad_missing)} keys): "
            f"{bad_missing[:5]}"
        )

    if model_keys is not None and loaded_keys is not None:
        expected_trainable = {
            k for k in model_keys
            if any(crit in k.lower() for crit in critical_substrings)
        }
        missing_from_loaded = expected_trainable - loaded_keys
        if missing_from_loaded:
            raise RuntimeError(
                f"State dict mismatch: expected trainable keys missing from loaded weights "
                f"({len(missing_from_loaded)} keys): {list(missing_from_loaded)[:5]}"
            )

    if unexpected_keys:
        raise RuntimeError(
            f"State dict mismatch: unexpected_keys found ({len(unexpected_keys)} keys): "
            f"{unexpected_keys[:5]}"
        )


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.unk_token_id = 0
        self.vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}

    def __len__(self):
        return 32000

    def encode(self, text, add_special_tokens=True):
        tokens = [1] if add_special_tokens else []
        for word in text.split():
            tokens.append(abs(hash(word)) % 30000 + 10)
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        return "mock decoded response <f_object> dog </f_object> <s_object> barking dog </s_object>"

    def batch_encode_plus(self, texts, return_tensors="pt", padding=True):
        if not HAS_TORCH:
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
            "attention_mask": torch.tensor([[1, 1, 1]] * len(texts)),
        }


class MockInnerModel:
    def __init__(self):
        self.pad_token_id = 0
        self.multimodal_initialized = False

    def init_multimodal_modules(self, *args, **kwargs):
        self.multimodal_initialized = True
        self.mm_kwargs = kwargs


class MockThinkerModel:
    def __init__(self, d_model=4096):
        self.d_model = d_model
        self.inner_model = MockInnerModel()
        self.active_adapters = ["default"]
        self._disabled_adapter = False
        if HAS_TORCH:
            self.dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    def get_model(self):
        return self.inner_model

    def initialize_MM_tokenizer(self, tokenizer, use_vqgan=False):
        pass

    @contextlib.contextmanager
    def disable_adapter(self):
        self._disabled_adapter = True
        try:
            yield
        finally:
            self._disabled_adapter = False

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def parameters(self):
        if HAS_TORCH:
            yield self.dummy_param
        else:
            return []

    def named_parameters(self):
        if HAS_TORCH:
            yield "dummy_param", self.dummy_param
        else:
            return []

    def forward(self, *args, **kwargs):
        if not HAS_TORCH:
            return None
        loss = self.dummy_param.sum() * 0.0
        return type("Output", (), {
            "loss": loss,
            "logits": torch.zeros((1, 10, 32000), requires_grad=True)
        })()

    def generate(self, *args, **kwargs):
        if not HAS_TORCH:
            return None
        return type("GenOutput", (), {
            "sequences": torch.tensor([[1, 101, 102, 2]]),
            "scores": [torch.zeros((1, 32000))] * 3
        })()

    def load_state_dict(self, state_dict, strict=False):
        missing = [k for k in ["dummy_base_key"] if k not in state_dict]
        unexpected = [k for k in state_dict if k.startswith("unexpected_")]
        if not strict:
            validate_checkpoint_keys(missing, unexpected)
        type_res = type("IncompatibleKeys", (), {"missing_keys": missing, "unexpected_keys": unexpected})()
        return type_res

    def to(self, device):
        return self


def build_tgs_model(
    model_name_or_path: str = "./pretrained_weights/Llama-2-7b-chat-hf",
    llm_name: str = "llama",
    ckpt_dir: Optional[str] = None,
    avs_ckpt_dir: Optional[str] = None,
    device: Optional[str] = None,
    use_mock: bool = False,
    use_hyper_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
    freeze_backbone: bool = True,
    ref_mode: str = "peft_disable",
    bf16: bool = True,
    fp16: bool = False,
    visual_branch: bool = True,
    audio_branch: bool = True,
    seg_branch: bool = False,
    vit_ckpt_path: str = "./pretrained_weights/clip-vit-large-patch14",
    select_layer_list: Any = None,
    select_feature: str = "patch",
    image_size: int = 224,
    patch_size: int = 14,
    visual_query_token_nums: int = 32,
    audio_query_token_nums: int = 32,
    BEATs_ckpt_path: str = "./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    prompt_embed_dim: int = 256,
    mask_decoder_transformer_depth: int = 2,
    low_res_mask_size: int = 112,
    avs_query_num: int = 300,
    num_classes: int = 1,
    query_generator_num_layers: int = 2,
    dice_loss_weight: float = 0.5,
    bce_loss_weight: float = 2.0,
) -> Tuple[Any, Any, Any]:
    """
    Builds the Ref-Thinker policy model, reference model, and tokenizer.
    Returns:
        (policy_model, ref_model, tokenizer)
    """
    # ------------------------------------------------------------------ #
    # Mock / Test Path (ONLY when use_mock is True)                       #
    # ------------------------------------------------------------------ #
    if use_mock:
        tokenizer = MockTokenizer()
        policy_model = MockThinkerModel()
        if ref_mode == "peft_disable":
            ref_model = policy_model
        else:
            ref_model = MockThinkerModel()
        return policy_model, ref_model, tokenizer

    # ------------------------------------------------------------------ #
    # Strict validation of files: RAISE if missing, never silent mock   #
    # ------------------------------------------------------------------ #
    if not os.path.exists(model_name_or_path):
        raise FileNotFoundError(
            f"Pretrained LLM backbone not found at: {model_name_or_path}. "
            "To run tests or pipeline in mock mode without pretrained weights, pass use_mock=True or --mock."
        )

    if ckpt_dir:
        ft_path = os.path.join(ckpt_dir, "finetune_weights.bin")
        if not os.path.exists(ft_path):
            raise FileNotFoundError(
                f"SFT LoRA checkpoint not found at: {ft_path}. "
                "Pass valid --ckpt_dir or use --mock."
            )

    if avs_ckpt_dir:
        non_lora_path = os.path.join(avs_ckpt_dir, "non_lora_trainables.bin")
        if not os.path.exists(non_lora_path):
            raise FileNotFoundError(
                f"Non-lora trainables checkpoint not found at: {non_lora_path}. "
                "Pass valid --avs_ckpt_dir or use --mock."
            )

    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for non-mock model building.")

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # ------------------------------------------------------------------ #
    # Real Model Loading Sequence from inference_hyper_lora.py lines 85-238
    # ------------------------------------------------------------------ #
    if llm_name == "llama":
        d_model = 4096
    elif llm_name == "qwen":
        d_model = 3584
    else:
        raise ValueError(f"Unsupported llm_name: {llm_name}. Choose 'llama' or 'qwen'.")

    compute_dtype = torch.float32
    if fp16:
        compute_dtype = torch.float16
    elif bf16:
        compute_dtype = torch.bfloat16

    attn_implementation = "sdpa" if torch.cuda.is_available() else "eager"

    # Base config: LlamaConfig or Qwen2Config (NOT generic AutoConfig)
    if llm_name == "llama":
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=compute_dtype,
        )
    elif llm_name == "qwen":
        from models.unified_qwen import UnifiedForCausalLM
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained(model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=compute_dtype,
        )

    model.config.use_cache = True

    if freeze_backbone:
        model.model.requires_grad_(False)

    # LoRA setup matching inference_hyper_lora.py
    target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj".split(",")
    if use_hyper_lora:
        from peft_hyper import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_nums=3,
        )
        model = get_peft_model(model, peft_config)
    else:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Tokenizer
    if llm_name == "qwen":
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
    elif llm_name == "llama":
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    model.get_model().pad_token_id = tokenizer.pad_token_id

    # Multimodal modules: exact call matching inference_hyper_lora.py lines 189-208
    model.get_model().init_multimodal_modules(
        visual_branch=visual_branch,
        audio_branch=audio_branch,
        segment_branch=seg_branch,
        d_model=d_model,
        vit_ckpt_path=vit_ckpt_path,
        select_layer_list=select_layer_list,
        select_feature=select_feature,
        image_size=image_size,
        patch_size=patch_size,
        visual_query_token_nums=visual_query_token_nums,
        audio_query_token_nums=audio_query_token_nums,
        BEATs_ckpt_path=BEATs_ckpt_path,
        prompt_embed_dim=prompt_embed_dim,
        mask_decoder_transformer_depth=mask_decoder_transformer_depth,
        low_res_mask_size=low_res_mask_size,
        avs_query_num=avs_query_num,
        num_classes=num_classes,
        query_generator_num_layers=query_generator_num_layers,
        dice_loss_weight=dice_loss_weight,
        bce_loss_weight=bce_loss_weight,
        use_vqgan=False,
    )

    model.initialize_MM_tokenizer(tokenizer, use_vqgan=False)

    # Checkpoint loading & strict key validation across UNION of both checkpoint files
    combined_state_dict = {}
    loaded_keys = set()

    if ckpt_dir:
        ckpt_path = os.path.join(ckpt_dir, "finetune_weights.bin")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            combined_state_dict.update(ckpt)
            loaded_keys.update(ckpt.keys())

    if avs_ckpt_dir:
        nolora_ckpt_path = os.path.join(avs_ckpt_dir, "non_lora_trainables.bin")
        if os.path.exists(nolora_ckpt_path):
            nolora_ckpt = torch.load(nolora_ckpt_path, map_location="cpu")
            combined_state_dict.update(nolora_ckpt)
            loaded_keys.update(nolora_ckpt.keys())

    if combined_state_dict:
        incomp = model.load_state_dict(combined_state_dict, strict=False)
        validate_checkpoint_keys(
            missing_keys=incomp.missing_keys,
            unexpected_keys=incomp.unexpected_keys,
            loaded_keys=loaded_keys,
            model_keys=set(model.state_dict().keys())
        )

    model = model.to(target_device)

    # Reference Model: copy or peft_disable
    if ref_mode == "peft_disable":
        ref_model = model
    else:
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    return model, ref_model, tokenizer
