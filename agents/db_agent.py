"""
Benchmark Agent 3: SQL Database Analyst Agent.
Translates user queries to SQL, validates against schema, and executes.

L1 Fix: Optionally uses GroqLLMClient to generate SQL from natural language,
proving DRAC is model-agnostic. Set GROQ_API_KEY env var to enable.
"""
from typing import Dict, Any, List, Optional
import sqlite3
from injector.proxy import RuntimeFaultProxy
try:
    from drac.llm_client import GroqLLMClient
except ImportError:
    GroqLLMClient = None  # type: ignore

class DatabaseAgent:
    def __init__(self, proxy: RuntimeFaultProxy, llm_client=None):
        self.proxy = proxy
        self.conn = sqlite3.connect(":memory:")
        self.context: List[Dict[str, Any]] = []
        self._init_db()
        self.llm = llm_client
        # WHY optional llm_client: when set, Groq generates the SQL query from
        # natural language. DRAC intercepts the sql_query tool call -- proving
        # the framework is model-agnostic (works with real LLM decisions).

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total_amount REAL, order_date TEXT)")
        cursor.execute("INSERT INTO orders VALUES (1, 101, 249.50, '2026-03-01')")
        cursor.execute("INSERT INTO orders VALUES (2, 102, 89.00, '2026-03-02')")
        cursor.execute("INSERT INTO orders VALUES (3, 101, 140.00, '2026-03-05')")
        self.conn.commit()

    def execute_task(self, query: str, task_prompt: str) -> Dict[str, Any]:
        self.context.append({"role": "user", "content": task_prompt})

        # L1 Fix: if a real LLM is available, ask it to generate/refine the SQL.
        # Schema context is injected so Groq knows valid column names.
        actual_query = query
        if self.llm is not None and getattr(self.llm, 'is_available', False):
            schema_hint = (
                "Table: orders(id, user_id, total_amount, order_date). "
                f"Natural language request: {task_prompt}. Generate SQL."
            )
            llm_resp = self.llm.ask(
                context=[{"role": "user", "content": task_prompt}],
                task_hint=schema_hint
            )
            if llm_resp.get("source") == "groq" and llm_resp.get("value"):
                actual_query = llm_resp["value"]

        def _run_sql():
            cursor = self.conn.cursor()
            cursor.execute(actual_query)
            rows = cursor.fetchall()
            return {"rows": rows, "count": len(rows)}

        res, telemetry = self.proxy.intercept_tool_call(
            agent_id="db_agent",
            tool_name="sql_query",
            tool_args={"query": actual_query},
            execute_fn=_run_sql
        )

        if telemetry.raw_error or res is None:
            self.context.append({"role": "assistant", "content": f"SQL execution error: {telemetry.raw_error}"})
            return {"success": False, "result": None, "telemetry": telemetry}

        self.context.append({"role": "assistant", "content": f"Query result: {res}"})
        return {"success": True, "result": res, "telemetry": telemetry}
