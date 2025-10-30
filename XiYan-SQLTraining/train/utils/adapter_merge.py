import torch
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.utils import GenerationConfig

 
def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    # base.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


def args_paser():
    parser = argparse.ArgumentParser(description='Argument Parser Example')
    parser.add_argument('--base_model_path', '-base_model_path', type=str, default="./model/Qwen/Qwen2.5-Coder-3B-Instruct", help='Path to model')
    parser.add_argument('--lora_adapter_path', '-lora_adapter_path', type=str, default="./output/sft_dense/XiYanSQL_qwen_3b/xxx", help='Path to adapter folder')
    parser.add_argument('--output_path', '-output_path', type=str, default="./output/sft_dense/XiYanSQL_qwen_3b/xxx_merged", help='Path to output folder')
    return parser.parse_args()



if __name__ == "__main__":
    args = args_paser()
    print(args)
    # Merge the base model and lora adapter
    apply_lora(args.base_model_path, args.output_path, args.lora_adapter_path)
    print("Merger Completed！")





