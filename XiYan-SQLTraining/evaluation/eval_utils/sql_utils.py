import os
import re
import json
import psycopg2
import mysql.connector
import sqlite3
def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def connect_postgresql(db_conn):
    db = psycopg2.connect(
        dbname=db_conn['db_name'],
        user=db_conn['db_user'],
        password=db_conn['db_password'],
        host=db_conn['db_host'],
        port=db_conn['db_port']
    )
    return db


def connect_mysql(db_conn):
    db = mysql.connector.connect(
        database=db_conn['db_name'],
        user=db_conn['db_user'],
        password=db_conn['db_password'],
        host=db_conn['db_host'],
        port=db_conn['db_port']
    )
    return db


def connect_db(db_conn):

    sql_dialect = db_conn.get('dialect', 'sqlite').lower()

    if sql_dialect == "sqlite":
        db_name = db_conn['db_name']
        db_path = f"{db_conn['db_host']}/{db_name}/{db_name}.sqlite"
        if not os.path.exists(db_path):
            print(f"Database file {db_path} does not exist.")
            raise FileNotFoundError(f"Database file {db_path} does not exist.")
        # print(db_path)
        conn = sqlite3.connect(db_path)
    elif sql_dialect == "mysql":
        conn = connect_mysql(db_conn)
    elif sql_dialect == "postgresql":
        conn = connect_postgresql(db_conn)
    else:
        raise ValueError("Unsupported SQL dialect")
    return conn


def execute_sql(idx, predicted_sql, ground_truth, db_conn, calculate_func):

    conn = connect_db(db_conn)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    conn.close()
    res = calculate_func(predicted_res, ground_truth_res)
    return res

def extract_sql_from_strs(model_result) -> str:
    sql = model_result
    pattern = r"```sql(.*?)```"

    sql_code_snippets = re.findall(pattern, model_result, re.DOTALL)

    if len(sql_code_snippets) > 0:
        sql = sql_code_snippets[-1].strip()
    if "```" in sql[:5]:
        sql = sql.split("```")[-1]
    if "```" in sql[-5:]:
        sql = sql.split("```")[0]

    return sql


def package_sqls(data_path, db_conn_config_path, mode='gt', pred_field='pred_sql', post_process=False):
    clean_sqls = []
    db_conn_list = []
    sql_data = read_json(data_path)
    db_conn = read_json(db_conn_config_path)
    for item in sql_data:
        db_name = item.get('db_id', '')
        if db_name == '':
            db_name = item.get('db_name', '')

        db_conn_list.append({
            'db_name': db_name,
            **db_conn
        })
        if mode == 'pred':
            sql = item[pred_field]
            if post_process:
                sql = extract_sql_from_strs(sql)
        else:
            try:
                sql = item['sql']
            except:
                sql = item['SQL']
        clean_sqls.append(sql)

    return clean_sqls, db_conn_list


def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x["sql_idx"])

