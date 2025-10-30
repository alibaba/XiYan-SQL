"""
sql_infer.py

大模型在测试数据上进行推理，获得结果，支持多种并发方式。
"""

import os
import sys
sys.path.append('../')
import datetime
import argparse
import torch
import torch.utils
import torch.utils.data

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
# from model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from eval_utils.sql_utils import read_json, write_json

from tqdm import tqdm
from peft import PeftModel

from accelerate import Accelerator
from accelerate.utils import gather_object
accelerator = Accelerator()


class Evaluator:

    def __init__(self, model_name_or_path, lora_path, test_data_path, expr_version, prompt_version='', batch_size=4, device='auto'):

        self.model_path = model_name_or_path
        self.test_data_path = test_data_path
        self.lora_path = lora_path

        self.batch_size = batch_size
        self.device = device

        result_dir = os.path.join(test_data_path.split('/')[0], 'output', expr_version)
        if not os.path.exists(result_dir):
            try:
                os.makedirs(result_dir)
                print(f"{result_dir} -File created!!!")
            except:
                print(f"{result_dir} -File exists!!!")

        today = datetime.date.today().strftime('%Y%m%d')
        self.save_path = os.path.join(result_dir, f'{expr_version}_{today}_results.json')


    def model_init(self, use_flash_attention):
        config = AutoConfig.from_pretrained(
            self.model_path,
            use_cache=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        print("loading model")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
            device_map=self.device
        )

        if len(self.lora_path) > 0:
            print("loading lora adpter...")
            model = PeftModel.from_pretrained(model, self.lora_path, torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2" if use_flash_attention else None, device_map='auto')
            model = model.merge_and_unload()
        self.model = model

    def inference_batch(self, save_interval=100, **infer_paras):
        final_result = []
        if os.path.exists(self.save_path):
            final_result = read_json(self.save_path)

        print("------Model inference-------")
        eval_json = read_json(self.test_data_path)
        print(f"{len(eval_json)} samples need to be processed...")

        texts, item_temps = [], []
        for idx, item in tqdm(enumerate(eval_json)):
            if idx < len(final_result) and len(final_result[idx]['pred_sql']) > 0:
                continue

            conversations = item['conversations'][:1]
            text = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                # chat_template=TEMPLATE,
                add_generation_prompt=True,
                enable_thinking=False
            )
            texts.append(text)
            item_temps.append((idx, item))
            if (len(texts) % self.batch_size == 0) or idx == len(eval_json) - 1:
                model_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
                    self.model.device)
                generated_ids = self.model.generate(
                    **model_inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=10,
                    temperature=infer_paras.get('temperature', None),
                    num_beams=1,
                    # top_p=infer_paras.get('top_p', 0.8),
                    do_sample=True,
                )
                torch.cuda.empty_cache()
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for i in range(len(response)):
                    cur_idx, item_temp = item_temps[i]
                    item_temp['pred_sql'] = response[i]
                    item_temp['sql'] = item_temp['conversations'][1]['content']
                    if cur_idx < len(final_result):
                        final_result[cur_idx] = item_temp
                    else:
                        final_result.append(item_temp)
                texts = []
                item_temps = []

                if (idx + 1) % save_interval == 0:
                    write_json(self.save_path, final_result)
        write_json(self.save_path, final_result)

    def inference_accelerator(self, **infer_paras):

        print("------Model Running-------")
        eval_json = read_json(self.test_data_path)[:40]
        print(f"{len(eval_json)} samples need to be processed...")

        def chunk_data(data, chunk_size):
            for i in range(0, len(data), chunk_size):
                yield data[i: min(i + chunk_size, len(data))]

        # add the original order indices
        for i, item in enumerate(eval_json):
            item['idx'] = i

        data_chunks = list(chunk_data(eval_json, self.batch_size))

        final_result = []
        for idx, t in tqdm(enumerate(data_chunks)):
            with accelerator.split_between_processes(t, ) as batch:
                texts = []
                for row in batch:
                    # prompt = gen_train_prompt(idx, row, self.sql_dialect)
                    conversations = row['conversations'][:1]
                    text = self.tokenizer.apply_chat_template(
                        conversations,
                        tokenize=False,
                        # chat_template=TEMPLATE,
                        add_generation_prompt=True
                    )
                    texts.append(text)

                model_inputs = self.tokenizer(texts, return_tensors="pt").to(accelerator.device)
                generated_ids = self.model.generate(
                    **model_inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=10,
                    temperature=infer_paras.get('temperature', None),
                    num_beams=1,
                    top_p=None,
                    do_sample=False,
                )
                torch.cuda.empty_cache()
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # calc index
                # num_samples_per_process, num_extras = divmod(len(t), accelerator.num_processes)
                # start_index = accelerator.process_index * num_samples_per_process + min(accelerator.process_index, num_extras)

                for i in range(len(response)):
                    batch[i]['pred_sql'] = response[i]
                    batch[i]['sql'] = batch[i]['conversations'][1]['content']
                    final_result.append(batch[i])

        final_result_gathered = gather_object(final_result)
        final_result_reordered = sorted(final_result_gathered, key=lambda x: x['idx'])
        write_json(self.save_path, final_result_reordered)
        # write_json(self.result_dir + f'/res_{accelerator.process_index}.json', final_result)


def args_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='/train/model/output/xiyansql_model/',
                        help='Path to trained model or base model')
    parser.add_argument('--lora_path', type=str, default='',
                        help='If model_path is the base model, you need to provide lora adapter for online merge')

    parser.add_argument('--expr_version', type=str, default='xiyan_date', help='Experimental version as the saved name')

    parser.add_argument('--test_set_path', type=str, default='bird_evaluation/eval_set/bird_dev_mschema_0926_short.json', help='Path to test set')

    parser.add_argument('--batch_size', type=int, default=2, help='The size of parallel processing')
    parser.add_argument('--use_flash_attention', type=bool, default=True, help='Whether to use Flash Attention.')
    parser.add_argument('--prompt_type', type=str, default='PostgreSQL', help='The type of prompt')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_paser()

    evaluator = Evaluator(args.model_name_or_path, args.lora_path, args.test_set_path, args.expr_version,
                          batch_size=args.batch_size, device='auto')
    evaluator.model_init(args.use_flash_attention)
    # evaluator.inference_batch(save_interval=args.batch_size*10, temperature=0.01)
    evaluator.inference_accelerator(temperature=0.01)







