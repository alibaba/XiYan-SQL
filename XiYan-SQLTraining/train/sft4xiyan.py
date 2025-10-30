import os
import sys
sys.path.append('../')
import json
import numpy as np
import transformers
from transformers import (
    Trainer,
    GenerationConfig,
    set_seed
)
from trainer.trainer import DeepCustomTrainer
from trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from trainer.train_util import load_tokenizer_and_model, DataCollatorForGeneration
from datasets import load_dataset, concatenate_datasets
from datasets import set_caching_enabled
set_caching_enabled(True)
from utils.common_utils import read_json, write_json

# Use swanlab to visualize training records
import swanlab
from swanlab.integration.transformers import SwanLabCallback

IGNORE_TOKEN_ID = -100
# Global variables
call_counter = 0


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    # set seed
    set_seed(training_args.seed)
    tokenizer, model = load_tokenizer_and_model(model_args, training_args, lora_args)

    def preprocess_data(example):
        conversations = example['conversations']
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            # chat_template=TEMPLATE,
            add_generation_prompt=False
        )
        encodings = tokenizer(text, truncation=True)
        target = conversations[1]['content']
        idx = text.find(target)
        target_idx = encodings.char_to_token(idx)
        labels = encodings['input_ids'].copy()
        # prompt_part = tokenizer.apply_chat_template(
        #     conversations[:1],
        #     # chat_template=TEMPLATE,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # encodings = tokenizer(text, truncation=True)
        # target_idx = len(tokenizer(prompt_part)['input_ids'])
        # labels = encodings['input_ids'].copy()

        if target_idx:
            labels[:target_idx] = [IGNORE_TOKEN_ID] * target_idx

        assert len(labels) == len(encodings['input_ids'])
        # Route different dialects under momq, ignoring in normal mode.
        if training_args.enable_dialect_router:
            sql_type = example.get("sql_type", "sqlite").lower()
            if sql_type == 'postgresql':
                labels[0] = min(0, training_args.dialect_num - 1)
            elif sql_type == 'mysql':
                labels[0] = min(1, training_args.dialect_num - 1)
            elif sql_type == 'sqlite':
                labels[0] = min(2, training_args.dialect_num - 1)
            elif sql_type == 'cypher':
                labels[0] = min(3, training_args.dialect_num - 1)
            elif sql_type == 'ngql':
                labels[0] = min(4, training_args.dialect_num - 1)
        encodings['labels'] = labels
        return encodings

    def preprocess_eval_data(example):
        conversations = example['conversations'][:1]
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            # chat_template=TEMPLATE,
            add_generation_prompt=True
        )
        encodings = tokenizer(text, truncation=True)
        db_name = example.get('db_name', '')
        sql_type = example.get("sql_type", "postgresql")
        target = example['conversations'][1]['content']
        labels = tokenizer(db_name + '[sep]' + sql_type + '[sep]' + target, truncation=True)['input_ids']
        encodings['labels'] = labels
        return encodings

    # load data
    train_data = load_dataset('json', data_files=data_args.data_path)['train']
    train_data = train_data.map(preprocess_data, remove_columns=train_data.column_names, num_proc=16, load_from_cache_file=True)
    eval_data = None
    if data_args.eval_data_path:
        eval_data_raw = load_dataset('json', data_files=data_args.eval_data_path)['train']
        eval_data = eval_data_raw.map(preprocess_eval_data, remove_columns=eval_data_raw.column_names, num_proc=16, load_from_cache_file=True)
    data_collator = DataCollatorForGeneration(tokenizer)

    def save_and_evaluate(pred_sqls, label_sqls):
        global call_counter
        call_counter += 1
        eval_data_result = []
        for idx in range(len(label_sqls)):
            pred_sql = pred_sqls[idx]
            new_item = {
                "idx": idx,
                "pred_sql": pred_sql,
                "sql": label_sqls[idx]
            }
            eval_data_result.append(new_item)
        # The calculation of metrics is omitted here.
        metrics = {}

        # save results
        save_path = os.path.join(
            training_args.output_dir,
            training_args.expr_id,
            f"{training_args.expr_id}_{int(call_counter * training_args.eval_steps):05d}.json"
        )
        write_json(save_path, eval_data_result)
        return metrics

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        gt_sqls = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_sqls = [
            pred[len(prompt):] for prompt, pred in zip(decoded_inputs, decoded_preds)
        ]
        metrics = save_and_evaluate(pred_sqls, gt_sqls)
        return metrics

    # Only used for evaluation during training
    training_args.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        temperature=None,
        num_beams=1,
        top_p=None,
        do_sample=False,
        use_cache=True
    )
    # Add extra loss for momq-moe
    extra_losses = []
    if training_args.enable_dialect_router:
        extra_losses.append('dialect_loss')
    if training_args.output_router_logits:
        extra_losses.append('aux_loss')

    # SwanLabCallback
    swanlab_callback = SwanLabCallback(
        project="SQLTrainer",
        experiment_name=training_args.expr_id
    )
    # Start via Deeply customized trainer
    trainer = DeepCustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        extra_losses=extra_losses,
        callbacks=[swanlab_callback],
    )

    # start via hf native trainer
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=eval_data,
    #     data_collator=data_collator,
    #     callbacks = [swanlab_callback],
    # )

    trainer.train(resume_from_checkpoint=training_args.resume)
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    train()

