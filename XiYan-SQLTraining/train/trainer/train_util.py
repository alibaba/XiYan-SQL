import torch
# import time
import torch.nn as nn
from .argument import TrainingArguments
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForTokenClassification
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .moe_momq import momq_init

class DataCollatorForGeneration(DataCollatorForTokenClassification):
    """Collate examples for training."""
    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            # pad_labels = []
            # for label in labels:
            #     if sequence_length - len(label) > 0:
            #         pad_labels.append([to_list(label)[0]] + [self.label_pad_token_id] * (sequence_length - len(label) - 1) + to_list(label))
            #     else:
            #         pad_labels.append([self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label))
            # batch[label_name] = pad_labels
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch


def modify_model_with_lora(model: nn.Module, training_args: TrainingArguments, lora_args: LoraConfig):

    if training_args.use_lora:
        print('Using Single Lora')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        # Print trainable params
        for n, p in model.named_parameters():
            if p.requires_grad and n.find('.1.') >= 0:
                print(f"{n}, {p.shape}, {p.dtype}")

    return model

def load_tokenizer_and_model(model_args, training_args: TrainingArguments, lora_args: LoraConfig):
    # model config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        # use_cache=False,
        trust_remote_code=True,
    )
    config.use_cache = False
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Here you can customize the tokenizer according to the model

    # Load model
    if 'auto' in model_args.model_type:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if model_args.use_flash_attention else None
        )
        if training_args.use_lora:
            print('Using Single Lora')
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if lora_args.q_lora:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )
            model = get_peft_model(model, lora_config)
            # Print peft trainable params
            model.print_trainable_parameters()
            if training_args.gradient_checkpointing:
                model.enable_input_require_grads()
            # Print trainable params
            for n, p in model.named_parameters():
                if p.requires_grad and n.find('.1.') >= 0:
                    print(f"{n}, {p.shape}, {p.dtype}")

    # Note: Start fine-tuning the XiYan multi-dialect MOE model in MOMQ mode.
    elif 'momq' in model_args.model_type:
        # init moe lora config
        model = momq_init(config, training_args, lora_args)

    else:
        raise ValueError(f"model_type {model_args.model_type} not supported")

    model = modify_model_with_lora(model, training_args, lora_args)
    return tokenizer, model
