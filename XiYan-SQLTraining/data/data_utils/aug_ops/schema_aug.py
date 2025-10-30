import random
from data_utils.m_schema import *

class SchemaShuffle(object):
    """
    Convert standard m-schema format text to shuffled scm text, including tab sf and col sf.
    Can be controlled by sf probability factor

    """
    def __init__(self, tab_rand_p=0, col_rand_p=0):
        super().__init__()
        # tab_rand_p 进行 table shuffle
        self.tab_rand_p = tab_rand_p
        # col_rand_p 进行 column shuffle
        self.col_rand_p = col_rand_p
        # fk sf
        self.fk_rand_p = 0.2

    def __call__(self, data: dict):
        """
        Args: data: json dict
        Returns: augmented data
        """
        try:
            db_name = data["db_name"]
            schema = data["db_schema"]
        except:
            raise ValueError("data must contain db_name and db_schema")
        scm_dict, fk_item_list = scm_text2dict(schema)
        # tab sf
        tab_list = list(scm_dict.items())
        if random.random() < self.tab_rand_p:
            tab_list = random.sample(tab_list, len(tab_list))
        shuffled_scm_dict = dict(tab_list)
        # col sf
        for tab_name, tab_info in shuffled_scm_dict.items():
            tab_info_list = list(tab_info.items())
            if random.random() < self.col_rand_p:
                tab_info_list = random.sample(tab_info_list, len(tab_info_list))
            shuffled_tab_dict = dict(tab_info_list)
            shuffled_scm_dict[tab_name] = shuffled_tab_dict
        # fk sf
        if random.random() < self.fk_rand_p:
            fk_item_list = random.sample(fk_item_list, len(fk_item_list))

        sf_schema = scm_augdict2text(db_name, shuffled_scm_dict, fk_item_list)
        data["db_schema"] = sf_schema
        return data


class SchemaFilter(object):
    """
    对标准的scm 进行 随机过滤操作
    """
    def __init__(self, tab_rand_p=0, col_rand_p=0):
        super().__init__()
        # The probability of choosing a table that does not appear
        self.tab_rand_p = tab_rand_p
        # Select the probability of a column not appearing
        self.col_rand_p = col_rand_p

    def __call__(self, data: dict):

        try:
            db_name = data["db_name"]
            schema = data["db_schema"]
            sql = data["sql"]
        except:
            raise ValueError("data must contain db_name and db_schema")
        scm_dict, fk_item_list = scm_text2dict(schema)

        scm_new_dict = {}
        for tab_name, tab_info in scm_dict.items():
            # 这里是应对一些匹配到表名和表描述 作为表名的情况，例如yiqi
            pattern_tab_name = tab_name.strip().split(",")[0]
            pattern = r'\b' + re.escape(pattern_tab_name) + r'\b'
            # 在sql里面，必选
            if re.search(pattern, sql, re.IGNORECASE):
                scm_new_dict[tab_name] = {}
            else:
                # 否则以tab_rand_p概率选择
                if random.random() < self.tab_rand_p:
                    scm_new_dict[tab_name] = {}
            # 如果选到
            if tab_name in scm_new_dict:
                for col_name, col_line in tab_info.items():
                    pattern = r'\b' + re.escape(col_name) + r'\b'
                    # 在sql里面，必选
                    if re.search(pattern, sql, re.IGNORECASE):
                        scm_new_dict[tab_name][col_name] = col_line
                    else:
                        # 否则以col_rand_p概率选择
                        if random.random() < self.col_rand_p:
                            scm_new_dict[tab_name][col_name] = col_line
        new_fk_item_list = scm_fk_filter(scm_new_dict, fk_item_list)
        filter_schema = scm_augdict2text(db_name, scm_new_dict, new_fk_item_list)
        data["db_schema"] = filter_schema

        return data

class SchemaPermute(object):
    """
    Replace some content of the standard scm
    """
    def __init__(self):
        super().__init__()
        self.default_enum_pre = "Value examples:"
        self.default_enum_pre_list = ["Value examples:", "Examples:"]

    def __call__(self, data: dict):
        try:
            db_name = data["db_name"]
            schema = data["db_schema"]
        except:
            raise ValueError("data must contain db_name and db_schema")
        scm_dict, fk_item_list = scm_text2dict(schema)

        # Some content replacement
        selected_weight = [10, 83, 3, 3, 1]
        selected_enum_pre = random.choices(["Value examples:", "Examples:", "示例值:", "values:", "--"], weights=selected_weight, k=1)[0]

        scm_new_dict = {}
        for tab_name, tab_info in scm_dict.items():
            scm_new_dict[tab_name] = {}
            for col_name, col_line in tab_info.items():

                for enum_pre in self.default_enum_pre_list:
                    if enum_pre in col_line:
                        col_pre, col_values = col_line.split(enum_pre)
                        if selected_enum_pre == "--":
                            new_col_line = col_pre + ")" + " -- Value examples:" + col_values[0:-1]
                        else:
                            new_col_line = col_pre + selected_enum_pre + col_values
                        scm_new_dict[tab_name][col_name] = new_col_line
                        break
                else:
                    new_col_line = col_line
                    scm_new_dict[tab_name][col_name] = new_col_line

        new_schema = scm_augdict2text(db_name, scm_new_dict, fk_item_list, space_rand_p=0.5)
        data["db_schema"] = new_schema
        return data


class SchemaReplace(object):
    """
    Table name replacement
    todo: column replacement
    """
    def __init__(self, tab_rand_p=0):
        super().__init__()
        self.tab_rand_p = tab_rand_p

    def __call__(self, data: dict):

        try:
            db_name = data["db_name"]
            schema = data["db_schema"]
            sql = data["sql"]
        except:
            raise ValueError("data must contain db_name and db_schema")
        scm_dict, fk_item_list = scm_text2dict(schema)
        # print(scm_dict)
        replace_list = []
        for tab_name, tab_info in scm_dict.items():
            # 这里是应对一些匹配到表名和表描述 作为表名的情况，例如yiqi
            pattern_tab_name = tab_name.strip().split(",")[0]
            pattern = r'\b' + re.escape(pattern_tab_name) + r'\b'
            # in sql
            if re.search(pattern, sql, re.IGNORECASE):
                if random.random() < self.tab_rand_p:
                    tab_new_name = db_name + "." + pattern_tab_name
                    replace_list.append([pattern_tab_name, tab_new_name])
        for item in replace_list:
            ori_tab_name = item[0]
            new_tab_name = item[1]
            schema = schema.replace(ori_tab_name, new_tab_name)
            sql = sql.replace(ori_tab_name, new_tab_name)

        data["db_schema"] = schema
        data["sql"] = sql

        return data


from data_utils.common_utils import extract_sql_from_strs
class SQLTranslate(object):
    """
    Replace some content of the standard sql
    """

    def __call__(self, data: dict):
        try:
            sql = data["sql"]
        except:
            raise ValueError("data must contain db_name and sql")

        if sql.count('--') >= 1:
            sql = re.sub(r'--.*?(\n|$)', '', sql)
        if sql.count("\n") >= 3:
            sql = re.sub(r'\s+', ' ', sql).strip()
        new_sql = extract_sql_from_strs(sql)

        data["sql"] = new_sql
        return data










