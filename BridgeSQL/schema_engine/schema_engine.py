import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.engine import Engine

from schema_engine.mschema import Schema
from schema_engine.paas_utils import examples_to_str


class SchemaEngine:
    """Lightweight schema extraction engine for SQLite databases."""

    def __init__(self, engine: Engine, db_name: str = '',
                 schema: Optional[str] = None, consider_topo: bool = True):
        self._engine = engine
        self._db_name = db_name
        self._schema = schema
        self._dialect = engine.dialect.name

        metadata = MetaData()
        metadata.reflect(bind=engine, schema=schema)
        self._metadata = metadata
        from sqlalchemy import inspect
        self._inspector = inspect(engine)

        self._usable_tables = [
            t for t in self._inspector.get_table_names(schema=schema)
            if self._inspector.has_table(t, schema)
        ]

        self._mschema = Schema(db_id=db_name, schema=schema)
        self._init_mschema(consider_topo)

    @property
    def mschema(self) -> Schema:
        return self._mschema

    def _get_protected_field_name(self, field_name: str) -> str:
        return f'`{field_name}`'

    def _get_protected_table_name(self, table_name: str) -> str:
        return f'`{table_name}`'

    def _fetch(self, sql_query: str):
        if not sql_query.strip().endswith(';'):
            sql_query += ';'
        with self._engine.connect() as conn:
            try:
                cursor = conn.execute(text(sql_query))
                return cursor.fetchall()
            except Exception:
                return None

    def _get_table_ddl(self, table_name: str) -> str:
        query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        result = self._fetch(query)
        if result and result[0][0]:
            ddl = result[0][0]
            ddl = re.sub(r'/\*.*?\*/', '', ddl, flags=re.DOTALL)
            return ddl
        return ""

    def _validate_topo(self) -> Tuple[bool, Optional[List[str]]]:
        edges = []
        for fk in self._mschema.foreign_keys:
            table_name, field_name, ref_schema, ref_table_name, ref_field_name = fk
            ref_table_info = self._mschema.tables.get(ref_table_name)
            if ref_table_info:
                ref_field_info = ref_table_info['fields'].get(ref_field_name, {})
                is_pk = ref_field_info.get('primary_key', False)
                is_unique = ref_field_info.get('unique', False)
                if not (is_pk or is_unique):
                    return False, None
            if table_name != ref_table_name:
                edges.append((ref_table_name, table_name))

        # Kahn's algorithm for topological sort
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        all_nodes = set(self._mschema.tables.keys())

        for src, dst in edges:
            if src in all_nodes and dst in all_nodes:
                adj[src].append(dst)
                in_degree[dst] += 1

        queue = deque([n for n in all_nodes if in_degree[n] == 0])
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) < len(all_nodes):
            if set(topo_order).issubset(all_nodes):
                topo_order += list(all_nodes - set(topo_order))
                return True, topo_order
            return False, None

        return True, topo_order

    def _init_mschema(self, consider_topo: bool):
        for table_name in self._usable_tables:
            table_ddl = self._get_table_ddl(table_name)
            self._mschema.add_table(table_name, fields={}, ddl=table_ddl, comment='')

            pks = self._inspector.get_pk_constraint(table_name, self._schema)
            pks = pks.get('constrained_columns', []) if pks else []

            # Unique constraints
            unique_keys = []
            try:
                for uc in self._inspector.get_unique_constraints(table_name, self._schema):
                    unique_keys.append(uc['column_names'])
            except Exception:
                pass
            self._mschema.tables[table_name]['unique_keys'] = unique_keys

            # Indexes
            keys = []
            try:
                for idx in self._inspector.get_indexes(table_name, self._schema):
                    keys.append(idx['column_names'])
            except Exception:
                pass
            self._mschema.tables[table_name]['keys'] = keys

            # Foreign keys
            fks = self._inspector.get_foreign_keys(table_name, self._schema)
            constrained_columns = []
            current_cols = {
                col['name'].lower(): col['name']
                for col in self._inspector.get_columns(table_name, schema=self._schema)
            }

            for fk in fks:
                referred_schema = fk.get('referred_schema')
                referred_table = next(
                    (t for t in self._usable_tables if t.lower() == fk['referred_table'].lower()),
                    fk['referred_table'],
                )
                referred_cols = {}
                try:
                    referred_cols = {
                        col['name'].lower(): col['name']
                        for col in self._inspector.get_columns(referred_table, schema=referred_schema)
                    }
                except Exception:
                    pass

                for c, r in zip(fk['constrained_columns'], fk['referred_columns']):
                    c = current_cols.get(c.lower(), c)
                    r = referred_cols.get(r.lower(), r)
                    self._mschema.add_foreign_key(table_name, c, referred_schema, referred_table, r)
                    constrained_columns.append(c)

            # Columns
            fields = self._inspector.get_columns(table_name, schema=self._schema)
            for field in fields:
                field_name = field['name']
                field_type = f"{field['type']!s}"
                primary_key = field_name in pks
                is_unique = (primary_key and len(pks) == 1) or ([field_name] in unique_keys)

                default = field.get('default')
                if default is not None:
                    default = f'{default}'

                constrained = field_name in constrained_columns

                # Fetch example values
                examples = []
                try:
                    sql = (
                        f"SELECT DISTINCT {self._get_protected_field_name(field_name)} "
                        f"FROM {self._get_protected_table_name(table_name)} "
                        f"WHERE {self._get_protected_field_name(field_name)} IS NOT NULL LIMIT 5;"
                    )
                    rows = self._fetch(sql)
                    if rows:
                        examples = [r[0] for r in rows]
                except Exception:
                    pass

                examples = examples_to_str(examples)
                examples = [e for e in examples if e is not None and e != '']

                self._mschema.add_field(
                    table_name, field_name, field_type=field_type,
                    primary_key=primary_key, nullable=field.get('nullable', True),
                    default=default, autoincrement=field.get('autoincrement', False),
                    unique=is_unique, constrained=constrained, comment='',
                    examples=examples,
                )

        if consider_topo:
            valid, topo_order = self._validate_topo()
            if not valid:
                raise ValueError("Invalid schema: cycles detected in foreign key constraints")
            self._mschema.topo = topo_order
        else:
            self._mschema.topo = None

    def dispose(self):
        self._engine.dispose()


def build_schema_engine(db_path: str, db_name: str) -> SchemaEngine:
    """Create a SchemaEngine from a SQLite database file."""
    engine = create_engine(f"sqlite:///{db_path}")
    return SchemaEngine(engine=engine, db_name=db_name)
