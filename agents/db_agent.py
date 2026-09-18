"""
Benchmark Agent 3: SQL Database Analyst Agent.
Translates user queries to SQL, validates against schema, and executes.
"""
from typing import Dict, Any, List
import sqlite3
from injector.proxy import RuntimeFaultProxy

class DatabaseAgent:
    def __init__(self, proxy: RuntimeFaultProxy):
        self.proxy = proxy
        self.conn = sqlite3.connect(":memory:")
        self.context: List[Dict[str, Any]] = []
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total_amount REAL, order_date TEXT)")
        cursor.execute("INSERT INTO orders VALUES (1, 101, 249.50, '2026-03-01')")
        cursor.execute("INSERT INTO orders VALUES (2, 102, 89.00, '2026-03-02')")
        cursor.execute("INSERT INTO orders VALUES (3, 101, 140.00, '2026-03-05')")
        self.conn.commit()

    def execute_task(self, query: str, task_prompt: str) -> Dict[str, Any]:
        self.context.append({"role": "user", "content": task_prompt})

        def _run_sql():
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            return {"rows": rows, "count": len(rows)}

        res, telemetry = self.proxy.intercept_tool_call(
            agent_id="db_agent",
            tool_name="sql_query",
            tool_args={"query": query},
            execute_fn=_run_sql
        )

        if telemetry.raw_error or res is None:
            self.context.append({"role": "assistant", "content": f"SQL execution error: {telemetry.raw_error}"})
            return {"success": False, "result": None, "telemetry": telemetry}

        self.context.append({"role": "assistant", "content": f"Query result: {res}"})
        return {"success": True, "result": res, "telemetry": telemetry}
