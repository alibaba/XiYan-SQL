NL2SQL_TEMPLATE = """你是一名SQL专家，现在需要阅读并理解下面的【数据库schema】描述，可能用到的【参考信息】，并运用数据库知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

NL2SQLITE_TEMPLATE = """你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

NL2MYSQL_TEMPLATE = """你是一名MySQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用MySQL知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

NL2PGSQL_TEMPLATE = """你是一名PostgreSQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用PostgreSQL知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""


NL2CYPHER_TEMPLATE = """你是一名Neo4j专家，现在需要阅读并理解下面的【图数据库schema】描述，以及可能用到的【参考信息】，并运用Cypher知识生成Cypher Query语句回答【用户问题】。
【用户问题】
{question}

【图数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```cypher"""

NL2NGQL_TEMPLATE = """你是一名NebulaGraph专家，现在需要阅读并理解下面的【图数据库schema】描述，以及可能用到的【参考信息】，并运用nGQL知识生成Graph Query语句回答【用户问题】。
【用户问题】
{question}

【图数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```ngql"""



SQLITE_SELF_REFINE_TEMPLATE = """你是一名SQLite专家，之前回复用户问题的【SQL】查询未能产生正确的结果，你需要根据提供的【数据库schema】描述，可能用到的【参考信息】和不正确SQL的【执行结果】来进行纠正，请提供一个能够正确回复【用户问题】的更正SQL。
【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

【SQL】
{error_sql}

【执行结果】
{error_info}

【更正SQL】
```sql"""


CANDIDATE_TEMPLATE = """候选{id}
【SQL】
{sql}
【执行结果】
{exec_info}"""

SQL2SELECT_TEMPLATE = """你是一名SQLite专家，针对【用户问题】，下面有{num}条 候选【SQL】及该sql在数据库上的【执行结果】（展示前10行）；你需要比较这些候选，分析不同的候选【SQL】之间的差异。基于给出的【数据库schema】、【参考信息】和【用户问题】选择一个正确合理的结果。
【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

==========

{candidates}

请输出所选择的候选是{items}"""



NL2PGSQL_TEMPLATE_EN = """You are a PostgreSQL expert. You need to read and understand the following【Database Schema】description and the possible provided【Evidence】, and use valid PostgreSQL knowledge to generate SQL for answering the【Question】.
【Question】
{question}

【Database Schema】
{db_schema}

【Evidence】
{evidence}

【Question】
{question}

```sql"""


NL2SQLITE_TEMPLATE_EN = """You are a SQLite expert. You need to read and understand the following【Database Schema】description and the possible provided【Evidence】, and use valid SQLite knowledge to generate SQL for answering the【Question】.
【Question】
{question}

【Database Schema】
{db_schema}

【Evidence】
{evidence}

【Question】
{question}

```sql"""



def gen_train_prompt(idx: int, data_item: dict, task_type: str) -> dict:
    """
    generate train samples
    """
    question = data_item["question"]
    evidence = data_item.get("evidence", "")
    db_schema = data_item["db_schema"]
    task_type = task_type.lower()

    if task_type == "nl2sqlite":
        prompt = NL2SQLITE_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif task_type == "nl2postgresql":
        prompt = NL2PGSQL_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif task_type == "nl2sqlite":
        prompt = NL2MYSQL_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    elif task_type == "self_refine":
        error_sql = data_item["pred_sql_res"][0]
        error_info = data_item["pred_sql_res"][1]
        prompt = SQLITE_SELF_REFINE_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence,
                                                    error_sql=error_sql, error_info=error_info)
    elif task_type == "cypher":
        prompt = NL2CYPHER_TEMPLATE.format(db_schema=db_schema.strip(), question=question, evidence=evidence)
    else:
        # for more task type, you can add more template here
        raise ValueError(f"Unsupported sql_type: {task_type}")

    output = data_item["sql"]
    conversation = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": output
        }
    ]
    train_item = {
        "id": idx,
        "conversations": conversation,
        "sql_type": task_type
    }
    return train_item

