from data_utils.common_utils import examples_to_str, read_json, write_json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from data_utils.type_engine import TypeEngine


class MSchema:
    def __init__(self, db_id: str = 'Anonymous', type_engine: Optional[TypeEngine] = None,
                 schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.foreign_keys = []
        self.type_engine = type_engine

    def add_table(self, name, fields={}, comment=None):
        self.tables[name] = {"fields": fields.copy(), 'examples': [], 'comment': comment}

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
            primary_key: bool = False, nullable: bool = True, default: Any = None,
            autoincrement: bool = False, unique: bool = False,
            comment: str = "",
            examples: list = [], category: str = '', dim_or_meas: Optional[str] = '', **kwargs):
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else f'{default}',
            "autoincrement": autoincrement,
            "unique": unique,
            "comment": comment,
            "examples": examples.copy(),
            "category": category,
            "dim_or_meas": dim_or_meas,
            **kwargs}

    def add_foreign_key(self, table_name, field_name, ref_schema, ref_table_name, ref_field_name):
        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_abbr_field_type(self, field_type, simple_mode=True)->str:
        if not simple_mode:
            return field_type
        else:
            return field_type.split("(")[0]

    def erase_all_table_comment(self):
        """clear all table descriptions."""
        for table_name in self.tables.keys():
            self.tables[table_name]['comment'] = ''

    def erase_all_column_comment(self):
        """clear all column descriptions."""
        for table_name in self.tables.keys():
            fields = self.tables[table_name]['fields']
            for field_name, field_info in fields.items():
                self.tables[table_name]['fields'][field_name]['comment'] = ''

    def has_table(self, table_name: str) -> bool:
        """check if given table_name exists in M-Schema"""
        if table_name in self.tables.keys():
            return True
        else:
            return False

    def has_column(self, table_name: str, field_name: str) -> bool:
        if self.has_table(table_name):
            if field_name in self.tables[table_name]["fields"].keys():
                return True
            else:
                return False
        else:
            return False

    def set_table_property(self, table_name: str, key: str, value: Any):
        if not self.has_table(table_name):
            print("The table name {} does not exist in M-Schema.".format(table_name))
        else:
            self.tables[table_name][key] = value

    def set_column_property(self, table_name: str, field_name: str, key: str, value: Any):
        if not self.has_column(table_name, field_name):
            print("The table name {} or column name {} does not exist in M-Schema.".format(table_name, field_name))
        else:
            self.tables[table_name]['fields'][field_name][key] = value

    def get_field_info(self, table_name: str, field_name: str) -> Dict:
        try:
            return self.tables[table_name]['fields'][field_name]
        except:
            return {}

    def get_category_fields(self, category: str, table_name: str) -> List:
        """
        给定table_name和category，获取当前table下所有category类型的字段名称
        category: 从type_engine.field_category_all_labels中取值
        """
        assert category in self.type_engine.field_category_all_labels, \
                        'Invalid category {}'.format(category)
        if self.has_table(table_name):
            res = []
            fields = self.tables[table_name]['fields']
            for field_name, field_info in fields.items():
                _ = field_info.get('category', '')
                if _ == category:
                    res.append(field_name)
            return res
        else:
            return []

    def get_dim_or_meas_fields(self, dim_or_meas: str, table_name: str) -> List:
        assert dim_or_meas in self.type_engine.dim_measure_labels, 'Invalid dim_or_meas {}'.format(dim_or_meas)
        if self.has_table(table_name):
            res = []
            fields = self.tables[table_name]['fields']
            for field_name, field_info in fields.items():
                _ = field_info.get('dim_or_meas', '')
                if _ == dim_or_meas:
                    res.append(field_name)
            return res
        else:
            return []


    def single_table_mschema(self, table_name: str, example_num=3, each_example_max_len=30,
                             show_type_detail=False) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get('comment', '')
        if table_comment is not None and table_comment != 'None' and len(table_comment) > 0:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}, {table_comment}")
        else:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}")
            else:
                output.append(f"# Table: {table_name}")

        field_lines = []
        # 处理表中的每一个字段
        for field_name, field_info in table_info['fields'].items():
            raw_type = self.get_abbr_field_type(field_info['type'], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()}"
            if field_info['comment'] != '':
                field_line += f", {field_info['comment'].strip()}"
            else:
                pass

            ## 加上value mapping
            if "value_mapping" in field_info and len(field_info["value_mapping"]) > 0:
                value_mapping_str_list = []
                for k, v in field_info["value_mapping"].items():
                    value_mapping_str_list.append(f"{k}表示{v}")
                value_mapping_str = ",".join(value_mapping_str_list)
                field_line += f", {value_mapping_str}"
            else:
                pass

            ## 打上主键标识
            is_primary_key = field_info.get('primary_key', False)
            if is_primary_key:
                field_line += f", Primary Key"

            """这两个标识暂不启用"""
            # # 是否允许非空
            # nullable = field_info.get('nullable', True)
            # if nullable and not is_primary_key:
            #     field_line += ', NULLABLE'
            # else:
            #     field_line += ', Not NULL'
            #
            # # 维度 or 度量
            # dim_or_meas = field_info.get('dim_or_meas', '')
            # if dim_or_meas in self.type_engine.dim_measure_labels:
            #     field_line += ', ' + dim_or_meas

            # 其他：外键依赖、Unique、索引等

            # 如果有示例，添加上
            if len(field_info.get('examples', [])) > 0 and example_num > 0:
                examples = field_info['examples']
                examples = [s for s in examples if s is not None]
                examples = examples_to_str(examples)
                if len(examples) > example_num:
                    examples = examples[:example_num]

                if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]]
                elif len(examples) > 0 and max([len(s) for s in examples]) > each_example_max_len:
                    examples = [example for example in examples if len(example) <= each_example_max_len]

                    if sum([len(s) for s in examples]) > 100:
                        examples = [examples[0]]
                else:
                    pass
                if len(examples) > 0:
                    example_str = ', '.join([str(example) for example in examples])
                    field_line += f", Examples: [{example_str}]"
                else:
                    pass
            else:
                field_line += ""
            field_line += ")"

            field_lines.append(field_line)
        output.append('[')
        output.append(',\n'.join(field_lines))
        output.append(']')

        return '\n'.join(output)

    def to_mschema(self, selected_tables: List = None, example_num=3, each_example_max_len=30, show_type_detail=False,
                   table_type=None) -> str:
        """
        convert to a MSchema string.
        selected_tables: 默认为None，表示选择所有的表
        type: 表类型，['table', 'view']，默认为None，表示不区分表类型
        """
        output = []

        # 添加 DB_ID
        output.append(f"【DB_ID】 {self.db_id}")
        # output.append("【DB_ID】")
        output.append("【Schema】")

        if selected_tables is not None:
            selected_tables = [s.lower() for s in selected_tables]

        # 依次处理每一个表
        temp_tables = []
        for table_name, table_info in self.tables.items():
            if selected_tables is None or table_name.lower() in selected_tables:
                cur_table_type = table_info.get('type', 'table')
                if table_type is None or cur_table_type == table_type:
                    temp_tables.append(
                        self.single_table_mschema(table_name, example_num, each_example_max_len, show_type_detail))
        # output.extend(sorted(temp_tables, key=len))
        output.extend(temp_tables)
        # 添加外键信息
        if self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                ref_schema = fk[2]
                table1, column1, _, table2, column2 = fk
                if selected_tables is None or \
                        (table1.lower() in selected_tables and table2.lower() in selected_tables):
                    if ref_schema == self.schema:
                        output.append(f"{fk[0]}.{fk[1]}={fk[3]}.{fk[4]}")

        return '\n'.join(output)

    def single_table_mschema_with_selected(self, table_name: str, select_col_list: List = None, select_values_dict: Dict = None, example_num=3, show_type_detail=False) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get('comment', '')
        if table_comment is not None and table_comment != 'None' and len(table_comment) > 0:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}, {table_comment}")
        else:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}")
            else:
                output.append(f"# Table: {table_name}")

        field_lines = []
        # 处理表中的每一个字段
        for field_name, field_info in table_info['fields'].items():
            # 列没有选到 跳过
            tab_col_name = table_name.lower() + "." + field_name.lower()
            # print(tab_col_name)
            if select_col_list is not None and (tab_col_name not in select_col_list):
                continue
            # print(tab_col_name)

            raw_type = self.get_abbr_field_type(field_info['type'], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()}"
            if field_info['comment'] != '':
                field_line += f", {field_info['comment'].strip()}"
            else:
                pass

            ## 打上主键标识
            is_primary_key = field_info.get('primary_key', False)
            if is_primary_key:
                field_line += f", Primary Key"

            # 如果有示例，添加上
            if len(field_info.get('examples', [])) > 0 and example_num > 0:

                examples = field_info['examples']
                # print(table_name, field_name, examples)
                examples = [s for s in examples if s is not None]
                examples = examples_to_str(examples)
                if len(examples) > example_num:
                    examples = examples[:example_num]
                # print(examples)
                if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]]
                elif len(examples) > 0 and max([len(s) for s in examples]) > 100:
                    examples = [example for example in examples if len(example) <= 100]

                    if sum([len(s) for s in examples]) > 100:
                        examples = [examples[0]]
                else:
                    pass
                # if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                #     examples = [examples[0]]
                # elif len(examples) > 0 and max([len(s) for s in examples]) > 20:
                #     if max([len(s) for s in examples]) > 50:
                #         examples = []
                #     else:
                #         examples = [examples[0]]
                # else:
                #     pass
                examples = [str(example) for example in examples]
                # print(examples)
                # 添加上指定的值
                if select_values_dict is not None and tab_col_name in select_values_dict:
                    selected_vaules_list = select_values_dict[tab_col_name]
                    # print(selected_vaules_list)
                    # print(examples)

                    examples = selected_vaules_list + examples
                    # if len(selected_vaules_list) <= 3:
                    #     examples = selected_vaules_list + examples
                    # else:
                    #     examples = selected_vaules_list

                    # examples = selected_vaules_list
                    examples = list(dict.fromkeys(examples))

                if len(examples) > 0:
                    example_str = ', '.join(examples)
                    field_line += f", Examples: [{example_str}]"

            else:
                field_line += ""


            field_line += ")"
            field_lines.append(field_line)

        output.append('[')
        output.append(',\n'.join(field_lines))
        output.append(']')

        return '\n'.join(output)

    def to_mschema_with_selected(self, selected_tables: List = None, select_col_list: List = None, select_values_dict: Dict = None, example_num=3, show_type_detail=False) -> str:
        """
        convert to a MSchema string.
        """
        output = []

        # 添加 DB_ID
        output.append(f"【DB_ID】 {self.db_id}")
        # output.append("【DB_ID】")
        output.append("【Schema】")

        if selected_tables is not None:
            selected_tables = [s.lower() for s in selected_tables]
        if select_col_list is not None:
            select_col_list = [s.lower() for s in select_col_list]
        if select_col_list is not None:
            selected_tables = list(dict.fromkeys([s.lower().split(".")[0] for s in select_col_list]))
        if select_values_dict is not None:
            select_values_dict = {key.lower(): value for key, value in select_values_dict.items()}
        # print(selected_tables)

        # 依次处理每一个表
        # having_tab_cn = 0
        temp_tables = []
        for table_name, table_info in self.tables.items():
            if selected_tables is None or table_name.lower() in selected_tables:
                temp_tables.append(self.single_table_mschema_with_selected(table_name, select_col_list, select_values_dict, example_num, show_type_detail))
                # having_tab_cn += 1

        output.extend(sorted(temp_tables, key=len))
        # 根据表的长度进行排序
        # output.sort(key=len)

        # 添加外键信息
        if self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                ref_schema = fk[2]
                table1, column1, _, table2, column2 = fk
                if selected_tables is None or \
                        (table1.lower() in selected_tables and table2.lower() in selected_tables):
                    if ref_schema == self.schema:
                        output.append(f"{fk[0]}.{fk[1]}={fk[3]}.{fk[4]}")

        return '\n'.join(output)

    def dump(self):
        schema_dict = {
            "db_id": self.db_id,
            "schema": self.schema,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys
        }
        return schema_dict

    def save(self, file_path: str):
        schema_dict = self.dump()
        write_json(file_path, schema_dict)

    def load(self, file_path: str):
        data = read_json(file_path)
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])

import re
import random
mschema_shape_template = """【DB_ID】 {db_id}
【Schema】
{tables_info}
【Foreign keys】
{fks_info}"""


def scm_text2dict(schema_text: str):
    """
    Standard scm text to dict
    args: schema_text: m-schema-like text
    returns:
    db_schema: {"tab_name":{"col_name":"col_line" }, ...}
    fk_list
    """
    # 最后无换行或数据会少解析一个table
    if "【Foreign keys】" not in schema_text:
        schema_text += "\n"
    schema_info = {}
    tables_pattern = re.compile(r'# Table: (.+?)\n\[([\s\S]+?)\]\n', re.MULTILINE)
    matches = tables_pattern.findall(schema_text)
    # print(matches)
    # 提取表结构
    for match in matches:
        table_name, table_contents = match
        schema_info[table_name] = {}
        # 过滤Value examples行
        row_items = table_contents.strip().split(",\n")
        for item in row_items:
            left_s = item.find('(') if item.find('(') > -1 else 0
            right_s = item.rfind(')') if item.rfind(')') > -1 else len(item)
            item = item[left_s + 1:right_s]
            col_name_str = item.strip().split(",")[0]
            col_name = col_name_str.split(":")[0].strip()
            schema_info[table_name][col_name] = "(" + item + ")"

    # 提取外键候选主键，无法处理所有格式
    fk_list = []
    if "【Foreign keys】" not in schema_text:
        return schema_info, fk_list
    fk_list_str = schema_text.strip().split("【Foreign keys】")[-1].strip().split("\n")
    for fks in fk_list_str:
        if len(fks) <= 0:
            continue
        fk1, fk2 = fks.strip().split("=")
        tab1, col1 = fk1.strip().split(".")
        tab2, col2 = fk2.strip().split(".")
        if "`" in col1 and "`" in col2:
            fk_list.append([tab1 + "." + col1[1:-1], tab2 + "." + col2[1:-1]])
        else:
            fk_list.append([tab1 + "." + col1, tab2 + "." + col2])

    return schema_info, fk_list


def scm_augdict2text(db_id: str, db_dict: dict, fk_item_list: List, space_rand_p: float=0) -> str:
    """
    aug dict转为标准的scm text
    args: db_id, db_dict:{"tab_name":{"col_name":"col_line" }, ...}, fk:[[], []]
    returns: db schema : mac sql text
    """
    pre_fix = "" if random.random() < space_rand_p else "  "
    table_info_list = []
    for tab_name, tab_col_content in db_dict.items():
        if len(tab_col_content) <= 0:
            continue
        tab_cols_list = []
        for col, col_line in tab_col_content.items():
            tab_cols_list.append(pre_fix + col_line)
        table_cols = ",\n".join(tab_cols_list)
        table_strs = f"# Table: {tab_name}\n[\n{table_cols}\n]"
        table_info_list.append(table_strs)

    fk_list = []
    for fks in fk_item_list:
        if len(fks) <= 0:
            continue
        fk1, fk2 = fks
        fk_list.append(fk1 + " = " + fk2)
    return mschema_shape_template.format(db_id=db_id, tables_info="\n".join(table_info_list), fks_info="\n".join(fk_list))

def scm_fk_filter(db_dict: dict, fk_list: List):
    """
    过滤掉不属于db_dict的fk
    args: db_dict:{"tab_name":{"col_name":"col_line" }, ...},  fk:[[], []]
    returns: new fk_list
    """
    new_fk_list = []
    for fks in fk_list:
        if len(fks) <= 0:
            continue
        fk1, fk2 = fks
        fk1_temp = fk1.strip().split('.')
        t1, c1 = fk1_temp[0], fk1_temp[1]
        fk2_temp = fk2.strip().split('.')
        t2, c2 = fk2_temp[0], fk2_temp[1]
        if t1 in db_dict and t2 in db_dict:
            if c1 in db_dict[t1] and c2 in db_dict[t2]:
                new_fk_list.append([fk1, fk2])
    return new_fk_list

