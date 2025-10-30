RAW_DATA_PATH=${1}
DB_CONN_CONFIG=${2}
PROCESSED_DATA_DIR=${3}
SAVE_MSCHEMA_DIR=${4}
SAVE_TO_CONFIGS=${5}

RAW_DATA_PATH=${RAW_DATA_PATH:-"data_warehouse/bird_train/raw_data/train.json"}
DB_CONN_CONFIG=${DB_CONN_CONFIG:-"data_warehouse/bird_train/db_conn.json"}
PROCESSED_DATA_DIR=${PROCESSED_DATA_DIR:-"data_warehouse/bird_train/processed_data/"}
SAVE_MSCHEMA_DIR=${SAVE_MSCHEMA_DIR:-"data_warehouse/bird_train/mschema/"}
SAVE_TO_CONFIGS=${SAVE_TO_CONFIGS:-"configs/datasets_all.json"}
python data_processing.py --raw_data_path  ${RAW_DATA_PATH} --db_conn_config ${DB_CONN_CONFIG} --processed_data_dir ${PROCESSED_DATA_DIR} --save_mschema_dir ${SAVE_MSCHEMA_DIR} --save_to_configs ${SAVE_TO_CONFIGS}


