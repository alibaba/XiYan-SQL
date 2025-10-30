import os
import sys
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from eval_utils.sql_utils import (
    execute_sql,
    package_sqls,
    sort_results,
)
from eval_utils.value_match import result_eq

exec_result = []
def result_callback(result):
    exec_result.append(result)


def calculate_ex(predicted_res, ground_truth_res):
    '''
    Calculate various execution accuracy
    '''
    ex_eq = 0
    ex_bird = 0
    if result_eq(ground_truth_res, predicted_res, False):
        ex_eq = 1
    if set(predicted_res) == set(ground_truth_res):
        ex_bird = 1
    return [ex_eq, ex_bird]


def execute_model(
    predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(idx, predicted_sql, ground_truth, db_place, calculate_ex),
        )
        ex_eq, ex_bird = res
        executable = 1
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        ex_eq = 0
        ex_bird = 0
        executable = 1
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        ex_eq = 0
        ex_bird = 0
        executable = 0
    result = {"sql_idx": idx, "ex_eq": ex_eq, 'ex_bird': ex_bird, "executable": executable}
    return result


def run_sqls_parallel(
    sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                meta_time_out,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()


def compute_acc_by_diff(exec_results, metric_type='ex_eq'):
    num_queries = len(exec_results)
    results = [res[metric_type] for res in exec_results]
    all_acc = sum(results) / num_queries
    count_lists = [
        num_queries
    ]
    return (
        all_acc * 100,
        count_lists,
    )


def sql_evaluator_all(src_path, test_data, db_conn_config_path, pred_field="pred_sql", num_cpus=16, meta_time_out=30.0):
    print("Starting SQL evaluation...")
    global exec_result
    pred_queries, db_conns = package_sqls(src_path, db_conn_config_path, 'pred', pred_field=pred_field)
    # generate ground truth sqls:
    gt_queries, _ = package_sqls(test_data, db_conn_config_path, 'gt')

    assert len(gt_queries) == len(pred_queries)
    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(
        query_pairs,
        db_places=db_conns,
        num_cpus=num_cpus,
        meta_time_out=meta_time_out,
    )
    exec_result = sort_results(exec_result)
    print("Starting calculate metric...")
    metric_return_dict = {}
    for metric_type in ['ex_eq', 'ex_bird', 'executable']:
        ex_acc, _ = compute_acc_by_diff(
            exec_result, metric_type)
        metric_return_dict[metric_type] = ex_acc
    temp_exec = exec_result
    exec_result = []
    return metric_return_dict, temp_exec
