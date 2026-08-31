import json
import copy
from typing import Any, Dict, List, Optional

from schema_engine.paas_utils import examples_to_str


class Schema:
    def __init__(self, db_id: str = 'Anonymous', schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.topo = []
        self.foreign_keys = []
        self.db_info = ''

    def add_table(self, name, fields={}, comment=None, ddl=None, table_type='table'):
        self.tables[name] = {
            "fields": fields.copy(),
            "examples": [],
            "comment": comment,
            "ddl": ddl,
            "type": table_type,
        }

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
                  primary_key: bool = False, nullable: bool = True, default: Any = None,
                  autoincrement: bool = False, unique: bool = False, constrained: bool = False,
                  comment: str = "", examples: list = [], value_desc: str = "",
                  value_mapping: dict = {}, category: str = '', dim_or_meas: Optional[str] = '',
                  **kwargs):
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else f'{default}',
            "autoincrement": autoincrement,
            "unique": unique,
            "constrained": constrained,
            "comment": comment,
            "examples": examples.copy(),
            "value_desc": value_desc,
            "value_mapping": value_mapping,
            "category": category,
            "dim_or_meas": dim_or_meas,
            **kwargs,
        }

    def add_foreign_key(self, table_name, field_name, ref_schema, ref_table_name, ref_field_name):
        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_field_type(self, field_type, simple_mode=True) -> str:
        if not simple_mode:
            return field_type
        return field_type.split("(")[0]

    def is_unique_pk_cons(self, table_name, field_name):
        try:
            item = self.tables[table_name]["fields"][field_name]
            return item.get('unique', False) or item.get('primary_key', False) or item.get('constrained', False)
        except Exception:
            return False

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def has_column(self, table_name: str, field_name: str) -> bool:
        return self.has_table(table_name) and field_name in self.tables[table_name]["fields"]

    def set_table_property(self, table_name: str, key: str, value: Any):
        if self.has_table(table_name):
            self.tables[table_name][key] = value

    def set_column_property(self, table_name: str, field_name: str, key: str, value: Any):
        if self.has_column(table_name, field_name):
            self.tables[table_name]['fields'][field_name][key] = value

    def get_field_info(self, table_name: str, field_name: str) -> Dict:
        try:
            return self.tables[table_name]['fields'][field_name]
        except Exception:
            return {}

    def erase_all_table_comment(self):
        for table_name in self.tables:
            self.tables[table_name]['comment'] = ''

    def erase_all_column_comment(self):
        for table_name in self.tables:
            for field_name in self.tables[table_name]['fields']:
                self.tables[table_name]['fields'][field_name]['comment'] = ''

    def single_table_mschema(self, table_name: str, selected_columns: List = None,
                             selected_values: Dict = None, example_num=3,
                             show_type_detail=False) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get('comment', '')

        header = f"# Table: "
        if self.schema and len(self.schema) > 0:
            header += f"{self.schema}."
        header += table_name
        if table_comment and table_comment != 'None' and len(table_comment) > 0:
            header += f", {table_comment}"
        output.append(header)

        field_lines = []
        if not selected_columns:
            selected_columns = list(table_info['fields'].keys())

        for field_name in selected_columns:
            if not self.has_column(table_name, field_name):
                continue
            field_info = self.get_field_info(table_name, field_name)
            raw_type = self.get_field_type(field_info['type'], not show_type_detail)
            field_line = f"({field_name}:{raw_type.upper()}"

            if field_info.get('comment', ''):
                field_line += f", {field_info['comment'].strip()}"

            if field_info.get('primary_key', False):
                field_line += ", Primary Key"

            if (len(field_info.get('examples', [])) > 0 and example_num > 0) or \
               (selected_values and f'{table_name}.{field_name}' in selected_values):

                if selected_values and f'{table_name}.{field_name}' in selected_values and \
                   len(selected_values[f'{table_name}.{field_name}']) > 0:
                    examples = list(selected_values[f'{table_name}.{field_name}'])
                    for e in field_info.get('examples', []):
                        if e not in examples:
                            examples.append(e)
                else:
                    examples = field_info.get('examples', [])

                examples = [s for s in examples if s is not None]
                examples = examples_to_str(examples)
                if len(examples) > example_num:
                    examples = examples[:example_num]

                if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]] if examples else []
                elif len(examples) > 0 and max(len(s) for s in examples) > 50:
                    if max(len(s) for s in examples) > 100:
                        examples = []
                    else:
                        examples = [examples[0]]

                if examples:
                    example_str = ', '.join(str(e) for e in examples)
                    field_line += f", Examples: [{example_str}]"

            if len(field_info.get('value_desc', '')) > 0:
                field_line += f", {field_info['value_desc'].replace(chr(10), '')}"

            if len(field_info.get('value_mapping', {})) > 0:
                for k, v in field_info['value_mapping'].items():
                    field_line += f", {k} means {v}"

            field_line += ")"
            field_lines.append(field_line)

        output.append('[')
        output.append(',\n'.join(field_lines))
        output.append(']')
        return '\n'.join(output)

    def to_mschema(self, selected_tables: List = None, selected_columns: List = None,
                   selected_values: Dict = None, example_num=3, show_type_detail=False,
                   table_type=None) -> str:
        output = []
        output.append(f"【DB_ID】 {self.db_id}")
        output.append("【Schema】")

        if selected_columns is not None and selected_tables is None:
            selected_tables = []
            for s in selected_columns:
                if s.split('.')[0] not in selected_tables:
                    selected_tables.append(s.split('.')[0])

        if selected_tables is None:
            selected_tables = list(self.tables.keys())

        for table_name in selected_tables:
            table_info = self.tables.get(table_name, {})
            if selected_columns is not None:
                cur_selected_columns = [
                    c.split('.')[1] for c in selected_columns if c.split('.')[0] == table_name
                ]
            else:
                cur_selected_columns = None

            if table_type is None or table_info.get('type', 'table') == table_type:
                output.append(self.single_table_mschema(
                    table_name, cur_selected_columns, selected_values, example_num, show_type_detail
                ))

        if (table_type is None or table_type == 'table') and self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                table1, column1, ref_schema, table2, column2 = fk
                if selected_tables is None or (table1 in selected_tables and table2 in selected_tables):
                    if ref_schema == self.schema:
                        output.append(f"{table1}.{column1}={table2}.{column2}")

        return '\n'.join(output)

    def single_table_omnischema(self, table_name: str, example_num: int = 2) -> str:
        """Generate DDL-based schema string with column comments and examples."""
        table_info = self.tables.get(table_name, {})
        ddl = table_info.get("ddl", "")
        if not ddl:
            return ""

        fields_dict = copy.deepcopy(table_info.get("fields", {}))
        new_ddl = []
        for line in ddl.split('\n'):
            line_parts = line.strip().split()
            if not line_parts:
                new_ddl.append(line)
                continue

            if line_parts[0].startswith(('"', '`')):
                quote_char = line_parts[0][0]
                end_idx = line.find(quote_char, line.find(quote_char) + 1)
                initial_word = line[:end_idx + 1].strip().strip('"').strip('`')
            else:
                initial_word = line_parts[0].strip().strip('"').strip('`')

            if initial_word not in fields_dict:
                new_ddl.append(line)
                continue

            info = fields_dict[initial_word]
            new_line = line
            comment = info.get("comment", "")
            examples = info.get("examples", [])[:example_num]

            if comment and examples:
                new_line += f" -- {comment}, example: {examples}"
            elif comment:
                new_line += f" -- {comment}"
            elif examples:
                new_line += f" -- example: {examples}"

            new_ddl.append(new_line)
            del fields_dict[initial_word]

        return "\n".join(new_ddl) + ";\n\n"

    def dump(self):
        return {
            "db_id": self.db_id,
            "schema": self.schema,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys,
            "topo": self.topo,
            "db_info": self.db_info,
        }

    def save(self, file_path: str):
        schema_dict = self.dump()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, ensure_ascii=False, indent=4, default=str)

    def load(self, file_path: str):
        with open(file_path) as f:
            data = json.load(f)
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])
        self.topo = data.get("topo", [])
        self.db_info = data.get("db_info", '')
