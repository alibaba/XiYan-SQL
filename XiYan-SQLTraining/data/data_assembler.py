"""
data_assembler.py

这个模块负责加载、预处理和整合来自多个数据源的数据。

"""
import random
import argparse
from data_utils.common_utils import read_json, write_json
from data_utils.aug_ops.augment import AugCompose
from data_utils.aug_ops.schema_aug import SchemaShuffle, SchemaFilter, SchemaPermute, SQLTranslate
from data_utils.prompt_utils import gen_train_prompt
from transformers import AutoTokenizer

def main(args):
    
    # Loading data source configuration
    dataset_config = read_json(args.dataset_config_path)
    token_sample = False

    # Data augmentation configuration
    # Make sure the schema field is db_schema, the sql field is sql, and the db name field is db_name...
    augment_compose = AugCompose([
        SchemaShuffle(tab_rand_p=0.5, col_rand_p=0.3),
        SchemaFilter(tab_rand_p=0.8, col_rand_p=0.7),
        SchemaPermute(),
        SQLTranslate()
    ])
    # Data Assembly
    global_cn = 0
    tokenizer = None
    train_data_merge = []
    for data_name, data_config in dataset_config.items():

        data_path = data_config['data_path']
        sample_num = data_config["sample_num"]
        print(f"Loading {data_name} data from {data_path}")
        data_list = read_json(data_path)
        sample_data_list = random.sample(data_list, sample_num) if sample_num > 0 else data_list

        task_type = data_config["task_name"]
        data_aug = data_config["data_aug"]

        for idx, item in enumerate(sample_data_list):

            if data_aug:
                item = augment_compose(item)
            train_item = gen_train_prompt(global_cn, item, task_type)
            if token_sample:
                # You can sample by token as needed. The following is just an example of truncation
                LONG_MAX_LEN = 1024 * 12
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained('/path/to/tokenizer')
                text = tokenizer.apply_chat_template(
                    train_item['conversations'],
                    tokenize=False,
                    add_generation_prompt=False
                )
                input_ids = tokenizer.encode(text)
                if len(input_ids) >= LONG_MAX_LEN:
                    continue

            train_data_merge.append(train_item)
            global_cn += 1
        print(f"{data_name} data loaded, total {len(sample_data_list)} samples")

    datasets_save_path = args.save_path
    write_json(datasets_save_path, train_data_merge)
    print(f"Total {len(train_data_merge)} samples save to {datasets_save_path}")


def args_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config_path', type=str, default='/path/to/configs/dataset_example.json',
                        help='Path to dataset config file')
    parser.add_argument('--save_path', type=str, default='../train/datasets/dataset_xxxx.json',
                        help='Path to save processed dataset')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_paser()
    main(args)



