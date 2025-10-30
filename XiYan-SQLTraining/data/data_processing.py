"""
data_processing.py

对于原始数据进行预处理，生成可以待装配的数据文件：
-

"""
import os
import argparse
from data_utils.common_utils import read_json, write_json, save_raw_text
from data_utils.schema_engine import get_db_engine, SchemaEngine


def gen_nl2sql_task(raw_data_path, db_conn_config, processed_data_dir, use_llm_description=False, save_schema_path=''):
    # m-schema dict for each db
    raw_data = read_json(raw_data_path)
    db_conn_config = read_json(db_conn_config)
    db_schema_dict = {}

    new_item_list = []
    for idx, item in enumerate(raw_data):

        db_name = item['db_name'] if 'db_name' in item else item['db_id']
        # Read the schema directly from the raw data
        db_schema_dict[db_name] = item.get('db_schema', db_schema_dict.get(db_name, None))
        
        # generate mschema
        if db_schema_dict[db_name] is None:
            print(f"[INFO] Generate mschema for db: {db_name}")
            engine = get_db_engine(db_name, db_conn_config)

            if not use_llm_description:
                schema_engine = SchemaEngine(engine=engine, db_name=db_name)
            else:
                #  use llm
                from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
                dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key='YOUR API KEY HERE.')
                schema_engine = SchemaEngine(engine=engine, db_name=db_name, llm=dashscope_llm, comment_mode='generation')
                schema_engine.fields_category()
                schema_engine.table_and_column_desc_generation()

            m_schema = schema_engine.mschema
            db_schema_dict[db_name] = m_schema.to_mschema()

        # Contains the required fields
        new_item = {
            "idx": idx,
            'db_name': db_name,
            # user question
            'question': item.get('question', ''),
            # external knowledge
            'evidence': item.get('evidence', ''),
            # db schema
            'db_schema': db_schema_dict[db_name],
            # gt sql
            'sql': item.get('SQL', ''),
        }
        new_item_list.append(new_item)

    # save mschema
    if len(save_schema_path) > 0:
        if not os.path.exists(save_schema_path):
            os.makedirs(save_schema_path)
        for db, schema_info in db_schema_dict.items():
            save_scm_path = os.path.join(save_schema_path, f"{db}.txt")
            save_raw_text(save_scm_path, schema_info)

    # save to file
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    raw_name, _ = os.path.splitext(os.path.basename(raw_data_path))
    save_path = os.path.join(processed_data_dir, f"{raw_name}_nl2{db_conn_config['dialect'].lower()}.json")
    write_json(save_path, new_item_list)

    # task info
    data_name = "_".join(os.path.splitext(save_path)[0].split('/')[1:])
    sum_num = len(new_item_list)
    task_name = f"nl2{db_conn_config['dialect'].lower()}"

    return data_name, {"data_path": save_path, "sample_num": -1, "sum_num": sum_num, "task_name": task_name}


def main_process(args):
    # s1: read raw data
    raw_data_path = args.raw_data_path
    db_conn_config = args.db_conn_config
    processed_data_dir = args.processed_data_dir
    save_mschema_path = args.save_mschema_dir
    print(f"[INFO] Read raw data from {args.raw_data_path}")
    # s2: generate processed data for different sql tasks
    data_name, task_info = gen_nl2sql_task(raw_data_path, db_conn_config, processed_data_dir,
                                           use_llm_description=args.use_llm_description, save_schema_path=save_mschema_path)
    print(f"[INFO] Generate processed data for task: {data_name}")
    # ...
    # s3: save task info to configs
    datasets_configs = read_json(args.save_to_configs)
    datasets_configs[data_name] = {**task_info, "data_aug": True}
    write_json(args.save_to_configs, datasets_configs)
    print(f"[INFO] Save task info to {args.save_to_configs}")
    

def args_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, required=True, default='data_warehouse/bird_train/raw_data/train.json',
                        help='Path to raw data file')
    parser.add_argument('--db_conn_config', type=str, required=True, default='data_warehouse/bird_train/db_conn.json',
                        help='Path to database connection config file')
    parser.add_argument('--use_llm_description', type=bool, required=False, default=False,
                        help='Whether to use llm to generate schema description')
    parser.add_argument('--processed_data_dir', type=str, required=True, default='data_warehouse/bird_train/processed_data/',
                        help='Path to processed data file')
    parser.add_argument('--save_mschema_dir', type=str, default='', help='schema folder path, whether to save db mschema as a separate file')
    parser.add_argument('--save_to_configs', type=str, required=True, default='configs/datasets_all.json',
                        help='Path to save task info file')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = args_paser()
    main_process(args)
