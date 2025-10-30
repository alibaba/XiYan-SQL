
from typing import Tuple, Any, List, Set, Hashable, Dict
from itertools import product
from collections import defaultdict

import random
import sqlglot
import textdistance
from sqlglot.optimizer import optimize
from sqlglot import parse_one, diff
from sqlglot.diff import Keep


def sql_compare(sql_gt, sql_pred, dialect):
    '''
    The two SQL queries are exactly the same
    '''
    dialect = dialect.lower()
    if dialect == 'postgresql':
        dialect = 'postgres'
    try:
        sql_gt = sqlglot.transpile(sql_gt, read=dialect,
                                   write=dialect, identify=True, pretty=False)[0]
        sql_pred = \
            sqlglot.transpile(sql_pred, read=dialect,
                              write=dialect, identify=True, pretty=False)[0]
    except:
        pass

    try:
        optimized_source = optimize(parse_one(sql_gt, dialect=dialect))
        optimized_pred = optimize(parse_one(sql_pred, dialect=dialect))
        edit_script = diff(optimized_source, optimized_pred)
        _ = sum(0 if isinstance(e, Keep) else 1 for e in edit_script)
        if _ == 0:
            exact_score = True
        else:
            exact_score = False
    except:
        exact_score = False

    return exact_score

def contains_none(res: list[tuple])->bool:
    if res is None:
        return True
    if len(res) == 0:
        return True
    else:
        for res_tuple in res: # for each tuple
            if len(res_tuple) > 0 and None in res_tuple:
                return True
    return False

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False




def rowsim(setL: Set, setR: Set) -> float:
    """
    Calculate the similarity between two sets using Jaccard index formula.
    """
    return len(setL.intersection(setR)) / len(setL.union(setR))


def floatify(v: Any) -> Any:
    """
    Attempts to convert a value to a float if it is a string and represents a
    number, or recursively apply the conversion to elements within a list or dict.
    """
    if isinstance(v, str):
        return v
    try:
        f = float(v)
        return f
    except:
        pass
    if isinstance(v, list):
        return [floatify(x) for x in v]
    if isinstance(v, dict):
        return {k: floatify(u) for k, u in v.items()}
    return v


def make_hashable(v: Any) -> Hashable:
    """
    Convert a value to a hashable type (needed for set operations).
    """
    float_v = floatify(v)
    if not isinstance(float_v, Hashable):
        return str(float_v)
    else:
        return float_v


def make_alignment(dictL: List[Dict], dictR: List[Dict]) -> Tuple[List[Set], List[Set]]:
    """
    Align rows from two lists of dictionaries based on their similarity.
    """
    swap = len(dictL) > len(dictR)

    # Forming set views from the list of dictionaries.
    setViewsL = [{make_hashable(v) for k, v in row.items()} for row in dictL]
    setViewsR = [{make_hashable(v) for k, v in row.items()} for row in dictR]
    if swap:
        setViewsL, setViewsR = setViewsR, setViewsL

    for i in range(len(setViewsL)):
        max_sim = -1
        max_j = -1
        for j in range(i, len(setViewsR)):
            sim = rowsim(setViewsL[i], setViewsR[j])
            if sim > max_sim:
                max_j = j
                max_sim = sim
        tmp = setViewsR[i]
        setViewsR[i] = setViewsR[max_j]
        setViewsR[max_j] = tmp
    if swap:
        setViewsL, setViewsR = setViewsR, setViewsL
    return setViewsL, setViewsR


def df_sim(dictL: List[Dict], dictR: List[Dict], list_view: bool) -> float:
    """
    Calculate the data frame similarity based on either the original row order or an alignment.
    """
    if list_view:
        # Original row order for lists of dictionaries
        view_L = [row.values() for row in dictL]
        view_R = [row.values() for row in dictR]
    else:
        view_L, view_R = make_alignment(dictL, dictR)

    totalSetL = set()
    for i, s in enumerate(view_L):
        for elem in s:
            totalSetL.add((i, make_hashable(elem)))
    totalSetR = set()
    for i, s in enumerate(view_R):
        for elem in s:
            totalSetR.add((i, make_hashable(elem)))
    intersection = totalSetL.intersection(totalSetR)
    union = totalSetL.union(totalSetR)

    if len(union) == 0 and len(intersection) == 0:
        return 1.0
    elif len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def df_sim_pair(pair_L, pair_R):
    """
    Compute the Jaccard similarity of two data frames (lists of dictionaries),
    taking into account the order of rows if indicated by the involved Cypher queries.
    """
    cypher_L, dict_L = pair_L
    cypher_R, dict_R = pair_R

    return df_sim(dict_L, dict_R, "order by" in f"{cypher_L} {cypher_R}".lower())


def get_jw_distance(string1: str, string2: str) -> float:
    """
    Calculate the Jaro-Winkler distance between two strings.

    The Jaro-Winkler distance is a measure of similarity between two strings.
    The score is normalized such that 0 equates to no similarity and
    1 is an exact match.
    """
    # Call the jaro_winkler function from the textdistance library.
    return textdistance.jaro_winkler(string1, string2)