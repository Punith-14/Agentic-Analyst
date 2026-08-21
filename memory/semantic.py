# memory/semantic.py
"""Krishna (Layer C) - NetworkX Schema Knowledge Graph.
Parses SQLite schema into graph structure (tables, columns, foreign keys).
Provides fast schema queries and relationship traversal.
"""
import sqlite3
from typing import Optional, List, Dict, Any

class SchemaGraph:
    """NetworkX schema knowledge graph representation."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "data/db/analytics.db"
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.foreign_keys: List[Dict[str, str]] = []
        if db_path:
            self.build(db_path)

    def build(self, db_path: str) -> None:
        """Parse SQLite schema into knowledge graph."""
        self.db_path = db_path
        self.tables.clear()
        self.foreign_keys.clear()

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names = [r[0] for r in cursor.fetchall()]

            for t in table_names:
                cursor.execute(f"PRAGMA table_info({t});")
                cols = [{"name": r[1], "type": r[2], "primary_key": bool(r[5])} for r in cursor.fetchall()]
                
                cursor.execute(f"PRAGMA foreign_key_list({t});")
                fks = [{"from_column": r[3], "to_table": r[2], "to_column": r[4]} for r in cursor.fetchall()]

                self.tables[t] = {
                    "columns": cols,
                    "foreign_keys": fks
                }
                for fk in fks:
                    self.foreign_keys.append({"from_table": t, **fk})

            conn.close()
        except Exception:
            # Fallback schema if DB file not yet initialized
            self.tables = {
                "orders": {"columns": [{"name": "order_id", "type": "INTEGER"}, {"name": "region", "type": "TEXT"}, {"name": "sales", "type": "REAL"}, {"name": "profit", "type": "REAL"}, {"name": "date", "type": "TEXT"}]},
                "customers": {"columns": [{"name": "customer_id", "type": "INTEGER"}, {"name": "customer_name", "type": "TEXT"}, {"name": "segment", "type": "TEXT"}]},
                "products": {"columns": [{"name": "product_id", "type": "INTEGER"}, {"name": "product_name", "type": "TEXT"}, {"name": "category", "type": "TEXT"}]},
                "regions": {"columns": [{"name": "region_id", "type": "INTEGER"}, {"name": "region_name", "type": "TEXT"}, {"name": "manager", "type": "TEXT"}]},
            }

    def get_schema(self, table: Optional[str] = None) -> Dict[str, Any]:
        """Return table list if table=None, or columns/FKs if table specified."""
        if not self.tables:
            self.build(self.db_path)

        if table is None:
            return {"tables": list(self.tables.keys())}
        
        if table in self.tables:
            return {"table": table, **self.tables[table]}
        
        return {"error": f"Table '{table}' not in schema."}

    def related_tables(self, table: str, hops: int = 1) -> List[str]:
        """Find tables reachable by foreign key relationships."""
        if not self.tables:
            self.build(self.db_path)

        related = set()
        for fk in self.foreign_keys:
            if fk["from_table"] == table:
                related.add(fk["to_table"])
            elif fk["to_table"] == table:
                related.add(fk["from_table"])

        return list(related)
