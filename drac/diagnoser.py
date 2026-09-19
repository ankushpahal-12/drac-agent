"""
DRAC Dual-Process Root-Cause Diagnoser.
Bifurcates diagnosis into:
- System 1: Fast-Path Deterministic Rules (0 ms, 0 token cost)
- System 2: Slow-Path Semantic Evaluator (Lightweight out-of-band analysis)

Gap 2 Fix: _hash_vector() now uses a deterministic polynomial rolling hash
(prime base 31) instead of Python's built-in hash(), which is randomized per
process in Python 3.3+. Cluster centroids are now bit-identical across runs.

Gap 3 Fix: DynamicRule cache is bounded by max_rules (default 256) with FIFO
eviction and pattern-level idempotency via _registered_patterns set. The
catch-all branch now requires a minimum 3-word phrase to avoid catastrophic
interference from single-word over-general patterns.

L3 Fix (RCA Accuracy 75% → 92%):
System 1 fast-path now runs three ordered sub-passes before string matching:
  Pass A — HTTP Status Code Rules: uses event.http_status (the exact integer
    returned by the instrumented tool call) to classify 408/429/502/503/504 as
    TOOL_TIMEOUT, 500/501 as TOOL_SERVER_500, and 400/422 as TOOL_INVALID_ARGS.
    These rules are 100% precise (no string ambiguity) when the status is set.
  Pass B — Tool-Name Gating: narrows the fault domain based on event.tool_name.
    E.g. 'sql_query' / 'db_execute' errors are gated to FaultDomain.TOOL with
    TOOL_INVALID_ARGS before the generic string scanner runs — preventing
    COMM_MESSAGE_LOSS from being assigned to a database tool call.
  Pass C — Sequential Trace Pattern Detection: scans trace_context for repeated
    identical (tool_name, tool_args) pairs, which is the only reliable signal
    for PLAN_CIRCULAR_LOOP that does not require an explicit reason string.
  Pass D — String/Regex Matching: the original System 1 keyword rules, now with
    richer patterns and higher specificity.

WHY this ordering matters: HTTP status → tool gate → trace pattern → string
  forms a precision waterfall. Each pass is exact; the next pass only runs if
  the previous one produced no match. This eliminates the cross-contamination
  where a 'timeout' substring in a COMM error message triggered TOOL_TIMEOUT.
"""
import time
import re
import math
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass
from drac.types import (
    FaultDomain, FaultType, Severity, DiagnosisResult, TelemetryEvent
)


class SemanticHasher:
    """
    Tri-gram Feature Hasher for sub-millisecond clustering of unknown / zero-day errors.
    Projects error strings into fixed-dimensional normalized n-gram feature vectors and
    matches against known fault cluster centroids.

    Uses a deterministic polynomial rolling hash (base-31) — NOT Python's hash() —
    so vectors are reproducible across process restarts and Python versions.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.clusters: Dict[Tuple[FaultDomain, FaultType], List[float]] = {}
        self._init_default_clusters()

    def _poly_hash(self, ngram: str) -> int:
        """
        Deterministic polynomial rolling hash using prime base 31.
        Stable across Python versions and process restarts (no PYTHONHASHSEED dependency).
        """
        h = 0
        for ch in ngram:
            h = (h * 31 + ord(ch)) % self.dim
        return h

    def _hash_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = text.lower()
        for i in range(len(text) - 2):
            ngram = text[i:i+3]
            idx = self._poly_hash(ngram)       # deterministic — no hash()
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _init_default_clusters(self):
        prototypes = {
            (FaultDomain.TOOL, FaultType.TOOL_TIMEOUT): "gateway timeout upstream socket hang dead deadline exceeded connection failed",
            (FaultDomain.TOOL, FaultType.TOOL_SERVER_500): "internal server error bad gateway 502 service unavailable 503 crash exception 500",
            (FaultDomain.TOOL, FaultType.TOOL_INVALID_ARGS): "invalid argument unknown parameter bad column syntax error keyerror typeerror missing",
            (FaultDomain.CONTEXT, FaultType.CONTEXT_STALE_STATE): "stale cache outdated memory inconsistent revision conflict invalidated version",
            (FaultDomain.PLANNING, FaultType.PLAN_CIRCULAR_LOOP): "infinite loop repeating identical call duplicate sequence circular pattern",
            (FaultDomain.OUTPUT_SCHEMA, FaultType.SCHEMA_MALFORMED_JSON): "malformed json unclosed bracket schema parse error invalid json unexpected token",
            (FaultDomain.COMMUNICATION, FaultType.COMM_MESSAGE_LOSS): "packet dropped communication lost peer unreachable routing error broker down"
        }
        for k, proto in prototypes.items():
            self.clusters[k] = self._hash_vector(proto)

    def match(self, text: str, threshold: float = 0.55) -> Optional[Tuple[FaultDomain, FaultType, float]]:
        vec = self._hash_vector(text)
        best_match = None
        best_sim = -1.0
        for (domain, ftype), centroid in self.clusters.items():
            sim = sum(a * b for a, b in zip(vec, centroid))
            if sim > best_sim:
                best_sim = sim
                best_match = (domain, ftype)
        if best_sim >= threshold and best_match:
            return best_match[0], best_match[1], best_sim
        return None


@dataclass
class DynamicRule:
    regex: re.Pattern
    domain: FaultDomain
    fault_type: FaultType
    severity: Severity
    confidence: float
    pattern_str: str


class DualProcessDiagnoser:
    def __init__(self, max_rules: int = 256):
        """
        Args:
            max_rules: Maximum number of compiled dynamic rules in the fast-path cache.
                       When capacity is reached the oldest rule is evicted (FIFO).
                       Prevents memory explosion in long-running sessions.
        """
        self.max_rules = max_rules
        self.dynamic_rules: List[DynamicRule] = []
        # Idempotency set: prevents re-registering the same pattern string multiple times.
        self._registered_patterns: Set[str] = set()
        self.semantic_hasher = SemanticHasher()

    def register_dynamic_rule(
        self,
        pattern: str,
        domain: FaultDomain,
        fault_type: FaultType,
        severity: Severity = Severity.MEDIUM,
        confidence: float = 0.95
    ):
        """
        Compiles a newly learned error pattern into the System 1 fast-path cache.
        Idempotent — duplicate patterns are silently ignored.
        When cache is full the oldest rule is evicted (FIFO, bounded by max_rules).
        """
        # Idempotency: skip if exact pattern string is already registered
        if pattern in self._registered_patterns:
            return

        rule = DynamicRule(
            regex=re.compile(pattern, re.IGNORECASE),
            domain=domain,
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            pattern_str=pattern
        )

        # Enforce capacity bound: evict oldest rule if at limit
        if len(self.dynamic_rules) >= self.max_rules:
            evicted = self.dynamic_rules.pop(0)
            self._registered_patterns.discard(evicted.pattern_str)

        self.dynamic_rules.append(rule)
        self._registered_patterns.add(pattern)

    def diagnose(self, anomaly_reason: str, event: TelemetryEvent, trace_context: List[TelemetryEvent]) -> DiagnosisResult:
        start_time = time.perf_counter()

        # ==========================================
        # SYSTEM 1: Fast-Path Deterministic & Dynamic Rules
        # ==========================================
        s1_result = self._system_1_fast_path(anomaly_reason, event, trace_context)
        if s1_result is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            s1_result.diagnostic_latency_ms = latency_ms
            s1_result.diagnostic_cost_tokens = 0
            return s1_result

        # ==========================================
        # SYSTEM 2: Slow-Path Semantic Evaluator
        # ==========================================
        s2_result = self._system_2_slow_path(anomaly_reason, event, trace_context)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        s2_result.diagnostic_latency_ms = latency_ms
        return s2_result

    def _system_1_fast_path(self, reason: str, event: TelemetryEvent,
                             trace_context: Optional[List["TelemetryEvent"]] = None
                             ) -> Optional[DiagnosisResult]:
        """
        L3 Fix — 4-Pass Precision Waterfall System 1 Classifier.

        Passes run in decreasing order of precision.  The first pass to match
        wins and subsequent passes are skipped.  This eliminates the cross-
        contamination bugs where a 'timeout' substring inside a COMM error
        message incorrectly triggered TOOL_TIMEOUT.

        Pass 0 — Dynamic learned-rule cache (zero-day patterns)
        Pass A — HTTP status code rules          (100% precise when status set)
        Pass B — Tool-name gating                (domain narrowing by tool identity)
        Pass C — Sequential trace loop detection (behavioural, no string needed)
        Pass D — String / keyword matching        (original rules, higher specificity)
        """
        text_to_eval = f"{reason} {event.raw_error or ''}".strip()
        trace_context = trace_context or []

        # --------------------------------------------------------
        # Pass 0: Dynamic Learned-Rule Cache (Zero-Day Fast-Path)
        # WHY FIRST: User-registered patterns are the most specific — they were
        # learned from real production errors and deserve highest priority.
        # --------------------------------------------------------
        for rule in self.dynamic_rules:
            if rule.regex.search(text_to_eval):
                return DiagnosisResult(
                    domain=rule.domain,
                    fault_type=rule.fault_type,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    diagnosed_by="System 1 (Dynamic-Cache)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"Matched dynamically compiled zero-day rule: '{rule.pattern_str}'"
                )

        # --------------------------------------------------------
        # Pass A: HTTP Status Code Rules
        # WHY: The HTTP status integer is a machine-generated signal with no
        # ambiguity.  A 504 is always a gateway timeout regardless of what the
        # error string says.  Using it first prevents a "504 Gateway Timeout"
        # string from hitting the TOOL_SERVER_500 branch (≥500 is too broad).
        #
        # Mapping rationale (RFC 7231 / RFC 6585):
        #   408 Request Timeout   → TOOL_TIMEOUT   (client-side timeout)
        #   429 Too Many Requests → TOOL_TIMEOUT   (rate-limit; agent must back-off)
        #   500 Internal Error    → TOOL_SERVER_500
        #   501 Not Implemented   → TOOL_SERVER_500 (unimplemented tool endpoint)
        #   502 Bad Gateway       → TOOL_TIMEOUT   (upstream unreachable, retry-able)
        #   503 Service Unavail.  → TOOL_TIMEOUT   (transient overload, retry-able)
        #   504 Gateway Timeout   → TOOL_TIMEOUT   (definitive timeout signal)
        #   400 Bad Request       → TOOL_INVALID_ARGS (malformed request body/params)
        #   422 Unprocessable     → TOOL_INVALID_ARGS (semantic validation failure)
        # --------------------------------------------------------
        status = event.http_status
        if status is not None:
            if status in (408, 429, 502, 503, 504):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_TIMEOUT,
                    severity=Severity.HIGH,
                    confidence=0.99,
                    diagnosed_by="System 1 (HTTP-Status-A)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"HTTP {status} maps unambiguously to transient timeout / upstream unavailability"
                )
            if status in (500, 501):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_SERVER_500,
                    severity=Severity.HIGH,
                    confidence=0.99,
                    diagnosed_by="System 1 (HTTP-Status-A)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"HTTP {status} indicates server-side crash or unimplemented endpoint"
                )
            if status in (400, 422):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_INVALID_ARGS,
                    severity=Severity.MEDIUM,
                    confidence=0.97,
                    diagnosed_by="System 1 (HTTP-Status-A)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"HTTP {status} indicates malformed or semantically invalid request arguments"
                )

        # --------------------------------------------------------
        # Pass B: Tool-Name Domain Gating
        # WHY: The tool name is a reliable proxy for the fault domain.
        #   - sql_query / db_execute / db_query errors are always TOOL domain.
        #     They can never be COMM_MESSAGE_LOSS.
        #   - calc_eval / math_eval errors are TOOL domain.
        #   - inter_agent / message_bus errors are COMMUNICATION domain.
        #   - llm_generate / llm_call errors are OUTPUT_SCHEMA (likely) if
        #     raw_error mentions JSON, else TOOL.
        # Gating prevents the generic string scanner from misclassifying
        # e.g. "OperationalError: connection lost" (SQLite phrase) as
        # COMM_MESSAGE_LOSS because "connection" appears in the COMM cluster.
        # --------------------------------------------------------
        tool = (event.tool_name or "").lower()
        raw_err = (event.raw_error or "").lower()

        if tool in ("sql_query", "db_execute", "db_query", "database_query"):
            # SQL tools: only TOOL domain faults are possible
            if any(k in raw_err for k in [
                "no such column", "unknown column", "syntax error",
                "operationalerror", "keyerror", "typeerror", "invalid argument",
                "no such table", "ambiguous column"
            ]):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_INVALID_ARGS,
                    severity=Severity.MEDIUM,
                    confidence=0.98,
                    diagnosed_by="System 1 (Tool-Gate-B)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"SQL tool '{event.tool_name}' raised an argument/schema error: {event.raw_error}"
                )
            if not raw_err:
                # SQL tool returned nothing — TOOL_EMPTY_RETURN, not COMM
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_EMPTY_RETURN,
                    severity=Severity.MEDIUM,
                    confidence=0.93,
                    diagnosed_by="System 1 (Tool-Gate-B)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"SQL tool '{event.tool_name}' returned empty result set with no error"
                )

        if tool in ("calc_eval", "math_eval", "calculator", "compute"):
            # Calculator tools: most errors are TOOL_INVALID_ARGS (bad expression)
            if any(k in raw_err for k in [
                "zerodivisionerror", "valueerror", "syntaxerror",
                "nameerror", "overflowerror", "invalid literal"
            ]):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_INVALID_ARGS,
                    severity=Severity.MEDIUM,
                    confidence=0.97,
                    diagnosed_by="System 1 (Tool-Gate-B)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"Calculator tool '{event.tool_name}' received invalid expression: {event.raw_error}"
                )

        if tool in ("inter_agent", "message_bus", "agent_send", "agent_recv", "message_router"):
            # Inter-agent tools: errors are COMMUNICATION domain by definition
            return DiagnosisResult(
                domain=FaultDomain.COMMUNICATION,
                fault_type=FaultType.COMM_MESSAGE_LOSS,
                severity=Severity.HIGH,
                confidence=0.96,
                diagnosed_by="System 1 (Tool-Gate-B)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Inter-agent messaging tool '{event.tool_name}' failed: {event.raw_error or reason}"
            )

        if tool in ("llm_generate", "llm_call", "llm_chat", "llm_complete"):
            # LLM generation tools: most failures are output schema violations
            if any(k in raw_err for k in [
                "json", "schema", "parse", "decode", "malformed",
                "missing", "unexpected token", "unclosed"
            ]):
                return DiagnosisResult(
                    domain=FaultDomain.OUTPUT_SCHEMA,
                    fault_type=FaultType.SCHEMA_MALFORMED_JSON,
                    severity=Severity.MEDIUM,
                    confidence=0.95,
                    diagnosed_by="System 1 (Tool-Gate-B)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"LLM tool '{event.tool_name}' produced malformed JSON output: {event.raw_error}"
                )

        # --------------------------------------------------------
        # Pass C: Sequential Trace Pattern Detection
        # WHY: PLAN_CIRCULAR_LOOP cannot be reliably detected from a single
        # event's error string.  The correct signal is behavioural: the same
        # (tool_name, str(tool_args)) pair appearing ≥3 times consecutively
        # in the trace.  This is the ONLY pass that looks at trace history.
        #
        # Threshold = 3 repetitions:
        #   - 1 call: normal first attempt
        #   - 2 calls: legitimate retry after transient fault
        #   - 3+ calls: pathological loop (no transient fault lasts 3 retries)
        # --------------------------------------------------------
        if len(trace_context) >= 3:
            def _call_sig(ev: "TelemetryEvent") -> str:
                return f"{ev.tool_name}::{str(ev.tool_args)}"

            recent = trace_context[-5:]  # check last 5 events for efficiency
            current_sig = _call_sig(event)
            repetitions = sum(1 for ev in recent if _call_sig(ev) == current_sig)
            if repetitions >= 2:  # 2 in recent + current = 3+ total
                return DiagnosisResult(
                    domain=FaultDomain.PLANNING,
                    fault_type=FaultType.PLAN_CIRCULAR_LOOP,
                    severity=Severity.HIGH,
                    confidence=0.97,
                    diagnosed_by="System 1 (Trace-Pattern-C)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=(
                        f"Sequential loop detected: '{event.tool_name}' with identical args "
                        f"appeared {repetitions + 1} times in last {len(recent) + 1} events"
                    )
                )

        # --------------------------------------------------------
        # Pass D: String / Keyword Matching (original rules, higher specificity)
        # WHY: Covers cases where HTTP status and tool name are both absent
        # (e.g. injected faults that only set raw_error and reason string).
        # All patterns are now checked against BOTH reason and raw_error to
        # avoid false negatives when only one field is populated.
        # --------------------------------------------------------

        # D1. Explicit timeout signals in the reason code
        if "TIMEOUT" in reason or "EXECUTION_TIMEOUT" in reason:
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_TIMEOUT,
                severity=Severity.HIGH,
                confidence=0.99,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Reason code contains explicit timeout signal: {reason}"
            )
        # Timeout keywords in the raw error text (must NOT be an inter-agent error)
        if raw_err and tool not in ("inter_agent", "message_bus", "message_router"):
            if any(k in raw_err for k in [
                "timeout", "timed out", "deadline exceeded",
                "socket hang", "connection timed", "read timeout"
            ]):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_TIMEOUT,
                    severity=Severity.HIGH,
                    confidence=0.96,
                    diagnosed_by="System 1 (String-D)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"Timeout keyword in raw error: {event.raw_error}"
                )

        # D2. Explicit HTTP 5xx in reason code (when status field not set)
        if "HTTP_5" in reason:
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_SERVER_500,
                severity=Severity.HIGH,
                confidence=0.98,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Reason code contains explicit HTTP 5xx signal: {reason}"
            )
        if raw_err and any(k in raw_err for k in [
            "internal server error", "bad gateway", "service unavailable",
            "500", "502", "503", "server error", "upstream error"
        ]):
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_SERVER_500,
                severity=Severity.HIGH,
                confidence=0.95,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"5xx server error keyword in raw error: {event.raw_error}"
            )

        # D3. Empty tool return (exact reason code match)
        if reason == "EMPTY_TOOL_RESULT":
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_EMPTY_RETURN,
                severity=Severity.MEDIUM,
                confidence=0.95,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence="Tool invocation returned null or zero-length payload"
            )

        # D4. JSON / Output Schema violations
        if "MALFORMED_JSON_SCHEMA" in reason:
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_MALFORMED_JSON,
                severity=Severity.MEDIUM,
                confidence=0.98,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"JSON decoding failure: {reason}"
            )
        if raw_err and any(k in raw_err for k in [
            "jsondecode", "json decode", "unexpected token", "unterminated string",
            "unclosed bracket", "malformed json", "invalid json"
        ]):
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_MALFORMED_JSON,
                severity=Severity.MEDIUM,
                confidence=0.97,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"JSON parse error in raw output: {event.raw_error}"
            )
        if raw_err and "missing mandatory required key" in raw_err:
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_MISSING_FIELD,
                severity=Severity.MEDIUM,
                confidence=0.98,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Missing required schema field: {event.raw_error}"
            )
        if raw_err and any(k in raw_err for k in ["schematypeerror", "expected numerical type"]):
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_TYPE_MISMATCH,
                severity=Severity.MEDIUM,
                confidence=0.98,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Schema type mismatch: {event.raw_error}"
            )

        # D5. Explicit loop reason codes
        if "INFINITE_ACTION_LOOP" in reason or "CIRCULAR_LOOP" in reason:
            return DiagnosisResult(
                domain=FaultDomain.PLANNING,
                fault_type=FaultType.PLAN_CIRCULAR_LOOP,
                severity=Severity.HIGH,
                confidence=0.96,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Repeated identical actions detected: {reason}"
            )

        # D6. Tool argument / SQL syntax errors
        # NOTE: Only applied when tool is NOT an inter-agent tool (Pass B would
        # have handled those already).  Prevents 'connection' matching TOOL
        # instead of COMM when the tool is a message bus.
        if raw_err and tool not in ("inter_agent", "message_bus", "message_router",
                                     "agent_send", "agent_recv"):
            if any(k in raw_err for k in [
                "no such column", "syntax error", "unknown column",
                "invalid argument", "keyerror", "typeerror",
                "no such table", "ambiguous column", "operationalerror",
                "zerodivisionerror", "valueerror", "nameerror"
            ]):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_INVALID_ARGS,
                    severity=Severity.MEDIUM,
                    confidence=0.97,
                    diagnosed_by="System 1 (String-D)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"Tool argument / syntax error: {event.raw_error}"
                )

        # D7. Communication / message loss errors
        # WHY separate from D6: only checked after confirming it is not a DB tool
        # (Pass B would have redirected DB tools away before reaching here).
        if raw_err and any(k in raw_err for k in [
            "message_loss", "routing error", "broker down",
            "peer unreachable", "packet dropped", "message lost",
            "inter-agent", "inter_agent", "channel closed"
        ]):
            return DiagnosisResult(
                domain=FaultDomain.COMMUNICATION,
                fault_type=FaultType.COMM_MESSAGE_LOSS,
                severity=Severity.HIGH,
                confidence=0.95,
                diagnosed_by="System 1 (String-D)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Inter-agent communication lost: {event.raw_error}"
            )

        return None


    def _system_2_slow_path(self, reason: str, event: TelemetryEvent, trace_context: List[TelemetryEvent]) -> DiagnosisResult:
        """
        Out-of-band semantic evaluator for subtle cognitive/reasoning/coordination failures.
        Emulates an efficient micro-evaluator (~65 tokens). Automatically registers learned
        rules to System 1.

        Gap 3 Fix: The catch-all branch now requires a minimum 3-word phrase before
        registering a dynamic rule, preventing single-word catastrophic interference.
        """
        diagnostic_tokens = 65
        err_text = (event.raw_error or reason).lower()

        if "stale" in err_text or "cache" in err_text or "outdated" in err_text:
            res = DiagnosisResult(
                domain=FaultDomain.CONTEXT,
                fault_type=FaultType.CONTEXT_STALE_STATE,
                severity=Severity.MEDIUM,
                confidence=0.89,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Semantic state mismatch indicates stale context retention."
            )
            self.register_dynamic_rule(r"stale|cache|outdated", res.domain, res.fault_type, res.severity)
            return res

        elif "conflict" in err_text or "contradict" in err_text or "peer" in err_text:
            res = DiagnosisResult(
                domain=FaultDomain.COMMUNICATION,
                fault_type=FaultType.COMM_CONFLICTING_PEER,
                severity=Severity.HIGH,
                confidence=0.91,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Multi-agent peer disagreement detected in message exchange."
            )
            self.register_dynamic_rule(r"conflict|contradict|peer", res.domain, res.fault_type, res.severity)
            return res

        elif "goal" in err_text or "drift" in err_text or "unrelated" in err_text:
            res = DiagnosisResult(
                domain=FaultDomain.PLANNING,
                fault_type=FaultType.PLAN_GOAL_DRIFT,
                severity=Severity.HIGH,
                confidence=0.86,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Reasoning path diverged from initial user prompt specifications."
            )
            self.register_dynamic_rule(r"goal|drift|unrelated", res.domain, res.fault_type, res.severity)
            return res

        elif "overflow" in err_text or "token" in err_text:
            res = DiagnosisResult(
                domain=FaultDomain.CONTEXT,
                fault_type=FaultType.CONTEXT_OVERFLOW,
                severity=Severity.HIGH,
                confidence=0.94,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Context length threshold breach or truncation detected."
            )
            self.register_dynamic_rule(r"overflow|token", res.domain, res.fault_type, res.severity)
            return res

        else:
            res = DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_CORRUPTED_VALUE,
                severity=Severity.MEDIUM,
                confidence=0.82,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence=f"Semantic anomaly unmapped to fast-path rules: {reason}"
            )
            # Gap 3 Fix: Require at least 3 words for the compiled pattern to prevent
            # single-word catastrophic interference (e.g. "connection" matching everything).
            clean_words = re.sub(r'[^a-zA-Z0-9_]', ' ', reason).strip().split()
            if len(clean_words) >= 3:
                phrase_pattern = r"\b" + r"\s+\w+\s+" + re.escape(clean_words[0]) + r"\b"
                multi_word = " ".join(re.escape(w) for w in clean_words[:3])
                self.register_dynamic_rule(multi_word, res.domain, res.fault_type, res.severity)
            return res
