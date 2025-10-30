from dataclasses import dataclass, field
from typing import Dict, Optional, List
from transformers import (
    Seq2SeqTrainingArguments,
)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use Flash Attention."})
    model_type: str = 'auto'


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    do_shuffle: bool = True

    train_dialects_num_map: str = field(default=None)
    eval_dialects_num_map: str = field(default=None)


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    expr_id: str = '20250606'
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    resume: bool = False
    output_router_logits: bool = False
    flash_attn: bool = False
    router_aux_loss_coef: float = 0.001
    predict_with_generate: bool = False
    include_inputs_for_metrics: bool = False
    # moe lora config
    use_moe_lora: bool = False
    unfreeze_layer_norms: bool = True
    lora_route_type: str = 'sentence'
    moe_lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    # moe config
    enable_dialect_router: bool = False
    dialect_router_loss_coef: float = 0.01
    dialect_num: int = 4
    enable_label_smooth: bool = False
    smooth_factor: float = 0.1
    add_moe_fusion: bool = False
    use_in_group_balance: bool = False
    hard_dialect_router: bool = False

    use_moe_expert: bool = False
    moe_intermediate_size: int = 64
    num_experts: int = 32
    num_experts_per_tok: int = 6
    share_expert_num: int = 1



@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 16 
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False