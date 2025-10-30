"""
sql_eval.py

对模型推理结果进行评测，当前支持ex、bird_ex、exec执行语法通过三种评测指标。
"""
import argparse
from eval_utils.sql_utils import read_json, write_json
from eval_utils.sql_calc_metric import sql_evaluator_all

def merge_data(pred_data, gt_data, exec_data, pred_field='pred_sql'):
    assert len(pred_data) == len(gt_data) == len(exec_data), "The length of the three data sets is not equal"

    merge_item_list = []
    for idx, (pred_item, gt_item, exec_item) in enumerate(zip(pred_data, gt_data, exec_data)):

        merge_item = {
            "idx": idx,
            "db_name": gt_item.get("db_name", ""),
            "question": gt_item.get("question", ""),
            "pred_sql": pred_item[pred_field],
            "sql": gt_item["sql"],
            **exec_item,
        }
        merge_item_list.append(merge_item)
    return merge_item_list

def args_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_sql_path', type=str, required=True, default='/train/model/output/xiyansql_model/',
                        help='The file path containing the predicted sql')
    parser.add_argument('--test_sql_path', type=str, required=True, default='bird_evaluation/eval_set/bird_dev_mschema_0926_short.json',
                        help='The file path containing the ground-truth sql')
    parser.add_argument('--db_conn_config', type=str, required=True, default='bird_evaluation/db_conn.json',
                        help='Path to database connection config file')
    parser.add_argument('--save_eval_path', type=str, default='', help='The file path to save the evaluation result')

    parser.add_argument('--pred_field', type=str, default='pred_sql', help='Select the predicted sql field')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_paser()
    metric_dict, exec_result = sql_evaluator_all(args.pred_sql_path, args.test_sql_path, args.db_conn_config, pred_field=args.pred_field)
    # print(metric_dict)
    print("*********Evaluation Results*********")
    for key, value in metric_dict.items():
        print("{} : {:<20.2f}".format(key, value))
    if len(args.save_eval_path) > 0:
        write_json(args.save_eval_path, merge_data(pred_data=read_json(args.pred_sql_path), gt_data=read_json(args.test_sql_path), exec_data=exec_result))








