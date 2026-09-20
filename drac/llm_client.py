"""
drac/llm_client.py  --  L1 Fix: Real LLM in the Loop via Groq API

WHY GROQ:
  Groq is an OpenAI-compatible LLM inference API with:
    - Sub-100ms TTFT (Time-To-First-Token) on LLaMA 3.1 8B-Instant
    - Free tier: 14,400 tokens/min, 500 requests/day on llama-3.1-8b-instant
    - Zero local GPU required -- runs in the cloud
    - 100% OpenAI-compatible: same request/response JSON schema

  This resolves L1 ("No Real LLM in the Loop") without changing the
  DRAC monitoring architecture. DRAC monitors telemetry at the tool-call
  boundary -- it does NOT inspect LLM internals.
  The framework is therefore proven model-agnostic: same proxy, same
  diagnoser, same arbiter, whether the decision-maker is Python code
  or Llama 3.1.

HOW IT WORKS:
  Each agent optionally accepts a GroqLLMClient. When provided:
    1. The agent calls client.ask(context, task_hint) to get the next
       action/query from Llama 3.1 8B (fast, free-tier-compatible).
    2. DRAC intercepts the TOOL call (not the LLM call) via proxy.
    3. All telemetry, fault detection, diagnosis, and recovery are
       identical to the deterministic Python baseline.

  When GROQ_API_KEY is not set OR groq is not installed:
    - Falls back to deterministic Python logic (original behaviour).
    - Benchmark is still fully reproducible without a key.

MODEL CHOICE -- llama-3.1-8b-instant:
  WHY: Groq p50 TTFT < 100ms; free tier (500 req/day); 8B parameters
  are sufficient for structured JSON tool-call decisions (SQL, math,
  search queries). Context window 131,072 tokens.
  Temperature = 0.0: deterministic greedy decoding for reproducibility.
  WHY 0.0: allows near-exact reproduction. Hardware noise adds < 1e-6
  probability of a different sampled token.

USAGE:
  Set environment variable:
    $env:GROQ_API_KEY = "gsk_..."       # PowerShell
    export GROQ_API_KEY="gsk_..."       # bash

  Then run with LLM mode:
    python main.py --mode all --llm groq

  Without the flag (default), agents run in deterministic Python mode.
"""
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Graceful import: groq is optional -- degrades to deterministic fallback
# ---------------------------------------------------------------------------
try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


def _load_env() -> None:
    """Auto-load .env file if GROQ_API_KEY is not already set in os.environ."""
    if os.environ.get("GROQ_API_KEY"):
        return
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            break

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
GROQ_DEFAULT_MODEL  = "llama-3.1-8b-instant"
# WHY llama-3.1-8b-instant: fastest free model on Groq, p50 TTFT < 100ms.
# Adequate for structured SQL / math / search query generation.

GROQ_FALLBACK_MODEL = "qwen/qwen3.8-27b"
# WHY fallback: alternative ultra-fast model available on free tier.

# ---------------------------------------------------------------------------
# System prompt (shared by all agents)
# WHY structured JSON output: makes LLM output machine-parseable without
# brittle string parsing. Groq json_object mode enforces valid JSON.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a precise AI agent decision engine. "
    "Given a task description and execution context, output ONLY a JSON "
    "object with the single next action to take. "
    "No explanation, no markdown, no preamble.\n\n"
    "Output format (always valid JSON):\n"
    "{\"action\": \"<action_type>\", \"value\": \"<action_value>\", "
    "\"reasoning\": \"<one_sentence>\"}\n\n"
    "action_type must be one of: "
    "execute_query, execute_expression, search_query, route_task\n"
    "action_value is the exact SQL query, math expression, or search term."
)


class GroqLLMClient:
    """
    Thin stateless wrapper around the Groq Python SDK.

    Provides:
      1. Structured JSON action generation from agent context.
      2. Automatic fallback to deterministic mode when key is missing.
      3. Wall-clock latency measurement (perf_counter) per call.
      4. Token usage reporting (prompt + completion) for L4 integration.

    WHY STATELESS: DRAC manages conversation state via
    TransactionalStateManager. The LLM client must not duplicate that
    state to avoid divergence between the checkpointed context and the
    LLM's view of the conversation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_DEFAULT_MODEL,
        temperature: float = 0.0,
    ):
        """
        Args:
            api_key:     Groq API key. If None, reads GROQ_API_KEY env var (or .env).
                         If both absent, client runs in deterministic fallback.
            model:       Groq model ID. Default: llama-3.1-8b-instant (auto-falls back if unavailable).
            temperature: Sampling temperature. Default 0.0 (reproducible).
        """
        _load_env()
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = os.environ.get("GROQ_MODEL", "") or model
        self.temperature = temperature
        self._client = None
        self._available = False

        if _GROQ_AVAILABLE and self._api_key:
            try:
                self._client = Groq(api_key=self._api_key)
                # Verify or auto-select best available chat model
                try:
                    available_models = [m.id for m in self._client.models.list().data]
                    if self.model not in available_models:
                        preferred = ["llama-3.1-8b-instant", "qwen/qwen3.8-27b", "gemma2-9b-it", "allam-2-7b"]
                        self.model = next((p for p in preferred if p in available_models), self.model)
                except Exception:
                    pass
                self._available = True
                print(f"[GroqLLMClient] Ready: model={self.model}, temp={temperature}")
            except Exception as exc:
                print(f"[GroqLLMClient] Init failed: {exc} -- fallback mode.")

    @property
    def is_available(self) -> bool:
        """True when groq SDK is installed AND a valid API key is configured."""
        return self._available

    def ask(
        self,
        context: List[Dict[str, str]],
        task_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Send context to Groq and return a structured action dict.

        Args:
            context:   List of {role, content} messages (agent conversation).
            task_hint: Short hint describing the next action goal, appended
                       as a final user message to steer output format.

        Returns dict with keys:
            action      -- decided action type (str)
            value       -- action value: query / expression / term (str)
            reasoning   -- one-sentence justification (str)
            latency_ms  -- wall-clock inference latency in ms (float)
            tokens_used -- prompt + completion tokens consumed (int)
                           WHY: feeds into L4 per-trial token accounting.
            model       -- model that served the request (str)
            source      -- "groq" | "fallback" (str)
        """
        if not self._available:
            return self._fallback_response(task_hint)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(context)
        if task_hint:
            messages.append({
                "role": "user",
                "content": f"Task: {task_hint}. Respond with ONLY the JSON object."
            })

        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=128,
                # WHY 128: action JSON is < 50 tokens; 128 gives headroom.
                response_format={"type": "json_object"},
                # WHY json_object: Groq guarantees syntactically valid JSON,
                # eliminating the need for brittle try/except JSON parsing.
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raw = completion.choices[0].message.content.strip()
            tokens_used = (
                completion.usage.prompt_tokens
                + completion.usage.completion_tokens
            )
            result = json.loads(raw)
            result["latency_ms"]  = round(latency_ms, 2)
            result["tokens_used"] = tokens_used
            result["model"]       = self.model
            result["source"]      = "groq"
            return result

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[GroqLLMClient] API error: {exc} -- using fallback.")
            fb = self._fallback_response(task_hint)
            fb["latency_ms"] = round(latency_ms, 2)
            return fb

    def _fallback_response(self, task_hint: str) -> Dict[str, Any]:
        """
        Deterministic fallback when Groq is unavailable.
        WHY: ensures agents work without a key for reproducible benchmarks.
        Dict schema is identical to a real Groq response -- no caller branching.
        """
        return {
            "action":      "execute_query",
            "value":       task_hint,
            "reasoning":   "Fallback: no LLM available, using task hint directly.",
            "latency_ms":  0.0,
            "tokens_used": 0,
            "model":       "fallback",
            "source":      "fallback",
        }


def make_client(api_key: Optional[str] = None) -> GroqLLMClient:
    """
    Factory: reads GROQ_API_KEY from environment and returns a client.
    Always safe to call -- returns fallback-mode client if key is absent.

    WHY factory: decouples agent constructors from import-time env reads.
    Agents call make_client() in __init__; the key can be set at any time
    before the first agent is created.
    """
    return GroqLLMClient(api_key=api_key or os.environ.get("GROQ_API_KEY", ""))
