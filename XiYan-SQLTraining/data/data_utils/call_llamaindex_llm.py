import time
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence


DEFAULT_IS_DATE_TIME_FIELD_TMPL = """你现在是一名数据分析师，给你数据表中某一列的相关信息，请你分析该列是否属于时间日期类型，仅回答"是"或"否"。
时间日期类型指的是由年、月、日、时、分、秒中的一种或几种组合而成的，要求月份必须在1-12之间，日期在1-31之间，小时在0-23之间，分钟和秒在0-59之间。

{field_info_str}
"""

DEFAULT_IS_DATE_TIME_FIELD_PROMPT = PromptTemplate(
    DEFAULT_IS_DATE_TIME_FIELD_TMPL,
    prompt_type=PromptType.CUSTOM,
)

# 时间日期类字段的最小颗粒度
DEFAULT_DATE_TIME_MIN_GRAN_TMPL = """你现在是一名数据分析师，给你数据表中的一个字段，已知该字段表示的含义与时间日期有关，请你根据该字段的组成格式和数据样例，推测该字段的最小颗粒度是什么。
说明：时间日期字段的最小颗粒度是指该字段能够精确到的最小时间单位。

以下是常见的时间单位：
YEAR: 最小时间单位是一年，例如，2024
MONTH: 某年的第几个月，一年有12个月，Month的取值在1-12之间，例如，2024-12
DAY: 某月的第几天，一个月最多有31天，因此Day取值在1-31之间，例如，2024-12-31
WEEK: 自然周，一般为一年中的第几周，一年包含52周多几天，Week通常在0-53之间，例如，2024-34
QUARTER: 某年的第几个季度，一年有四个季度，Quarter通常在1-4取值
HOUR: 某天的第几个小时，一天有24个小时，Hour在0-23之间
MINUTE: 某小时的第几分钟，一小时有60分钟，Minute在0-59之间
SECOND: 某分钟的第几秒，一分钟有60秒，Second在0-59之间
MILLISECOND: 毫秒
MICROSECOND: 微秒
OTHER: 其他不属于以上的时间单位，比如半年、一刻钟等

直接给出最小时间单位的名称。

以下样例供你参考：
【字段信息】
字段名称: dt
数据类型: DOUBLE
Value Examples: [202412.0, 202301.0 202411.0, 202201.0, 202308.0, 202110.0, 202211.0]
最小时间单位: MONTH

【字段信息】
字段名称: dt
数据类型: TEXT
Value Examples: ['2022-12', '2022-14', '2021-40', '2021-37', '2021-01', '2021-32', '2023-04', '2023-37']
最小时间单位: WEEK

【字段信息】
字段名称: dt
数据类型: TEXT
Value Examples: ['12:30:30', '23:45:23', '01:23:12', '12:12:12', '14:34:31', '18:43:01', '22:13:21']
最小时间单位: SECOND

请你参考以上样例，推测下面字段的最小时间单位，直接给出最小时间单位的名称。
【字段信息】
{field_info_str}
最小时间单位: """

DEFAULT_DATE_TIME_MIN_GRAN_PROMPT = PromptTemplate(
    DEFAULT_DATE_TIME_MIN_GRAN_TMPL,
    prompt_type=PromptType.CUSTOM,
)


DEFAULT_STRING_CATEGORY_FIELD_TMPL = '''你现在是一名数据分析师，给你数据表中某一列的相关信息，请你分析该列是enum类型、code类型、还是text类型，仅回答"enum"、"code"或"text"。

enum：具有枚举的特征：字段取值相对固定，集中在一个预定义的有限集合内，通常长度较短，组成模式相对固定，一般用于状态、类型等字段；
code：有特定意义的编码，code的组成通常存在一定的规律或标准，比如用户id、身份证号等；
text：自由文本，通常用于描述或说明，不受长度和形式的限制，内容可以是任何形式的文本。

{field_info_str}
'''

DEFAULT_STRING_CATEGORY_FIELD_PROMPT = PromptTemplate(
    DEFAULT_STRING_CATEGORY_FIELD_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_NUMBER_CATEGORY_FIELD_TMPL = """你现在是一名数据分析师，给你数据表中某一列的相关信息，请你分析该列是enum类型、code类型、还是measure类型，仅回答"enum"、"code"或"measure"。

enum：枚举类型，取值局限于一个预定义的有限集合，通常长度较短，一般用于状态、类型等字段；
code：有特定意义的编码，code的组成通常存在一定的规律或标准，比如用户id、身份证号等；
measure：指标、度量，可以用来做进行计算和聚合，比如求平均、最大值等。

{field_info_str}
"""

DEFAULT_NUMBER_CATEGORY_FIELD_PROMPT = PromptTemplate(
    DEFAULT_NUMBER_CATEGORY_FIELD_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_UNKNOWN_CATEGORY_FIELD_TMPL = """你现在是一名数据分析师，给你数据表中某一列的相关信息，请你分析该列是enum类型、measure类型、code类型、还是text类型，仅回答"enum"、"measure"、"code"或"text"。

enum：枚举类型，取值局限于一个预定义的有限集合，通常长度较短，一般用于状态、类型等字段；
code：有特定意义的编码，code的组成通常存在一定的规律或标准，比如用户id、身份证号等；
text：自由文本，通常用于描述或说明，不受长度限制，内容可以是任何形式的文本；
measure：指标、度量，可以用来做进行计算和聚合，比如求平均、最大值等。

{field_info_str}
"""

DEFAULT_UNKNOWN_FIELD_PROMPT = PromptTemplate(
    DEFAULT_UNKNOWN_CATEGORY_FIELD_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_COLUMN_DESC_GEN_CHINESE_TMPL = '''你现在是一名数据分析师，给你一张数据表的字段信息和一些数据样例如下：

{table_mschema}

【SQL】
{sql}
【Examples】
{sql_res}

下面是该表中字段"{field_name}"的详细信息：
{field_info_str}

以下信息可供你参考：
{supp_info}

现在请你仔细阅读并理解上述内容和数据，为字段"{field_name}"添加中文名称，要求如下：
1、中文名称尽可能简洁清晰，准确描述该字段所表示的业务语义，不要偏离原有的字段描述；
2、字段中文名的长度不要超过20个字；
3、按json格式输出：
```json
{"chinese_name": ""}
```
'''

DEFAULT_COLUMN_DESC_GEN_CHINESE_PROMPT = PromptTemplate(
    DEFAULT_COLUMN_DESC_GEN_CHINESE_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_COLUMN_DESC_GEN_ENGLISH_TMPL = '''你现在是一名数据分析师，给你一张数据表的字段信息和一些数据样例如下：

{table_mschema}

【SQL】
{sql}
【Examples】
{sql_res}

下面是该表中字段"{field_name}"的详细信息：
{field_info_str}

以下信息可供你参考：
{supp_info}

现在请你仔细阅读并理解上述内容和数据，为字段"{field_name}"添加英文描述，要求如下：
1、英文描述要尽可能简洁清晰，准确描述该字段所表示的业务语义，不要偏离原有的字段描述；
2、总输出长度不要超过20个单词；
3、按json格式输出：
```json
{"english_desc": ""}
```
'''

DEFAULT_COLUMN_DESC_GEN_ENGLISH_PROMPT = PromptTemplate(
    DEFAULT_COLUMN_DESC_GEN_ENGLISH_TMPL,
    prompt_type=PromptType.CUSTOM,
)


DEFAULT_UNDERSTAND_DATABASE_TMPL = '''你现在是一名数据分析师，给你一个数据库的Schema如下：

{db_mschema}

请你仔细阅读以上信息，在database的层面上分析，该数据库主要存储的是什么领域的什么数据，给出总结即可，不需要针对每张表分析。
'''

DEFAULT_UNDERSTAND_DATABASE_PROMPT = PromptTemplate(
    DEFAULT_UNDERSTAND_DATABASE_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_GET_DOMAIN_KNOWLEDGE_TMPL = '''有这样一个数据库，基本信息如下：
{db_info}

结合你所学习到的知识分析，在该领域，人们通常关心的维度和指标有哪些？
'''

DEFAULT_GET_DOMAIN_KNOWLEDGE_PROMPT = PromptTemplate(
    DEFAULT_GET_DOMAIN_KNOWLEDGE_TMPL,
    prompt_type=PromptType.CUSTOM,
)

# 按照category，理解各个字段之间的区别与联系
DEFAULT_UNDERSTAND_FIELDS_BY_CATEGORY_TMPL = '''你现在是一名数据分析师，给你一个数据的基本信息：

【数据库信息】
{db_info}

其中数据表"{table_name}"的字段信息和数据样例如下：
{table_mschema}

【SQL】
{sql}
【Examples】
{sql_res}

请你仔细阅读并理解该数据表，已知表中的{fields}字段均为 {category} 字段，请你分析这几个字段之间的关系和区别是什么？
'''

DEFAULT_UNDERSTAND_FIELDS_BY_CATEGORY_PROMPT = PromptTemplate(
    DEFAULT_UNDERSTAND_FIELDS_BY_CATEGORY_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_TABLE_DESC_GEN_CHINESE_TMPL = '''你现在是一名数据分析师，给你一张数据表的字段信息如下：

{table_mschema}

以下是一些数据样例：
【SQL】
{sql}
【Examples】
{sql_res}

现在请你仔细阅读并理解上述内容和数据，为该数据表生成一段中文的表描述，要求：
1、说明该表在何种维度（包括时间维度和其他维度）上存储了什么指标数据；
2、字数控制在100字以内。
3、回答以json格式输出。

```json
{"table_desc": ""}
```
'''

DEFAULT_TABLE_DESC_GEN_CHINESE_PROMPT = PromptTemplate(
    DEFAULT_TABLE_DESC_GEN_CHINESE_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_TABLE_DESC_GEN_ENGLISH_TMPL = '''你现在是一名数据分析师，给你一张数据表的字段信息如下：

{table_mschema}

以下是一些数据样例：
【SQL】
{sql}
【Examples】
{sql_res}

现在请你仔细阅读并理解上述内容和数据，为该数据表生成一段英文的表描述，要求：
1、说明该表在何种维度（包括时间维度和其他维度）上存储了什么指标数据；
2、长度不要超过100个单词。
3、回答以json格式输出。

```json
{"table_desc": ""}
```
'''

DEFAULT_TABLE_DESC_GEN_ENGLISH_PROMPT = PromptTemplate(
    DEFAULT_TABLE_DESC_GEN_ENGLISH_TMPL,
    prompt_type=PromptType.CUSTOM,
)

DEFAULT_SQL_GEN_TMPL = '''你现在是一名{dialect}数据分析师，给你一个数据库的Schema信息如下：

【数据库Schema】
{db_mschema}

【用户问题】
{question}
【参考信息】
{evidence}

请你仔细阅读并理解该数据库，根据用户问题和参考信息的提示，生成一句可执行的SQL来回答用户问题，生成的SQL用```sql 和```保护起来。
'''

DEFAULT_SQL_GEN_PROMPT = PromptTemplate(
    DEFAULT_SQL_GEN_TMPL,
    prompt_type=PromptType.CUSTOM,
)



def call_llm(prompt: BasePromptTemplate, llm: Optional[LLM] = None, max_try=5, sleep=10, **prompt_args)->str:
    for try_idx in range(max_try):
        try:
            res = llm.predict(prompt, **prompt_args)
            return res
        except:
            time.sleep(sleep)
    return ''


def call_llm_message(messages: Sequence[ChatMessage], llm: Optional[LLM] = None, max_try=5, sleep=10, **kwargs)->str:
    for try_idx in range(max_try):
        try:
            res = llm.chat(messages, **kwargs)
            return res.message.content
        except:
            time.sleep(sleep)
    return ''

