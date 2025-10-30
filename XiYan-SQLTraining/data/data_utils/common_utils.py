import re
import json
import decimal
import datetime



def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(filename)->str:
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            data.append(line)
    return data

def read_raw_text(filename)->str:

    # 使用with语句来确保文件正确关闭
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()  # 读取整个文件内容
    except:
        content = ''
    return content

def save_raw_text(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def read_list(file):
    ls = []
    with open(file,'r', encoding='utf-8') as f:
        ls = f.readlines()

    ls  = [l.strip().replace('\n','') for l in ls]
    return ls

def read_dict_list(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            data.append(json.loads(line))
    return data

def save_list(file,ls):
    with open(file,'w', encoding='utf-8') as f:
        f.write('\n'.join(ls))


def read_map_file(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            data[line[0]] = line[1].split('、')
            data[line[0]].append(line[0])
    return data



def is_email(string):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False

def examples_to_str(examples: list) -> list[str]:
    """
    from examples to a list of str
    """
    values = examples
    for i in range(len(values)):
        if isinstance(values[i], datetime.date):
            values = [values[i]]
            break
        elif isinstance(values[i], datetime.datetime):
            values = [values[i]]
            break
        elif isinstance(values[i], decimal.Decimal):
            values[i] = str(float(values[i]))
        elif is_email(str(values[i])):
            values = []
            break
        elif 'http://' in str(values[i]) or 'https://' in str(values[i]):
            values = []
            break
        elif values[i] is not None and not isinstance(values[i], str):
            pass
        elif values[i] is not None and '.com' in values[i]:
            pass

    return [str(v) for v in values if v is not None and len(str(v)) > 0]


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

def extract_simple_json_from_strs(model_result) -> dict:
    model_result=model_result.replace('\n', '')
    pattern = r"```json(.*?)```"

    # 使用re.DOTALL标志来使得点号(.)可以匹配包括换行符在内的任意字符
    sql_code_snippets = re.findall(pattern, model_result, re.DOTALL)
    data={}
    if len(sql_code_snippets) > 0:
        data = sql_code_snippets[-1].strip()
        try:
            data = eval(data)
        except:
            find = re.findall('错误信息\':\'(.*)\'', data)
            try:
                if len(find)>0:
                    find_out = find[0].replace('\'','"')
                    data=data.replace(find[0],find_out)
                    data = eval(data)
                else:

                    #re.findall('错误信息\':\'(.*)\'', data)[0].replace('\'', '"')
                    if "]}" in data:
                        data = data.replace(']}', '}]')
                        data = eval(data)
                    if 'false' in data or 'true' in data:
                        data = data.replace('false','False').replace('true','True')
                        data = eval(data)
                    else:
                        print("en error happened on eval")
                    data={}
            except:
                data={}
    return data