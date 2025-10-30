
PRED_SQL_PATH=${1}
TEST_SQL_PATH=${2}
DB_CONN_CONFIG=${3}
SAVE_EVAL_PATH=${4}

PRED_SQL_PATH=${PRED_SQL_PATH:-"bird_evaluation/output/eval_test/bird_eval_test.json"}
TEST_SQL_PATH=${TEST_SQL_PATH:-"bird_evaluation/eval_set/bird_dev_mschema_0926_short.json"}
DB_CONN_CONFIG=${DB_CONN_CONFIG:-"bird_evaluation/db_conn.json"}
SAVE_EVAL_PATH=${SAVE_EVAL_PATH:-"bird_evaluation/output/eval_test/sql_res_eval_0611.json"}

python sql_eval.py --pred_sql_path ${PRED_SQL_PATH} --test_sql_path ${TEST_SQL_PATH} --db_conn_config ${DB_CONN_CONFIG} --save_eval_path ${SAVE_EVAL_PATH}
