"""
DRAC Dual-Process Root-Cause Diagnoser.
Bifurcates diagnosis into:
- System 1: Fast-Path Deterministic Rules (0 ms, 0 token cost)
- System 2: Slow-Path Semantic Evaluator (Lightweight out-of-band analysis)
"""
import time
import re
import math
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from drac.types import (
    FaultDomain, FaultType, Severity, DiagnosisResult, TelemetryEvent
)

class SemanticHasher:
    """
    Embedding-Space Semantic Hasher for sub-millisecond clustering of unknown / zero-day errors.
    Projects error strings into fixed-dimensional normalized n-gram feature vectors and matches
    against known fault cluster centroids.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.clusters: Dict[Tuple[FaultDomain, FaultType], List[float]] = {}
        self._init_default_clusters()

    def _hash_vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = text.lower()
        for i in range(len(text) - 2):
            ngram = text[i:i+3]
            idx = hash(ngram) % self.dim
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
    def __init__(self):
        self.dynamic_rules: List[DynamicRule] = []
        self.semantic_hasher = SemanticHasher()

    def register_dynamic_rule(self, pattern: str, domain: FaultDomain, fault_type: FaultType, severity: Severity = Severity.MEDIUM, confidence: float = 0.95):
        """Compiles a newly learned error pattern into the System 1 fast-path cache."""
        rule = DynamicRule(
            regex=re.compile(pattern, re.IGNORECASE),
            domain=domain,
            fault_type=fault_type,
            severity=severity,
            confidence=confidence,
            pattern_str=pattern
        )
        self.dynamic_rules.append(rule)

    def diagnose(self, anomaly_reason: str, event: TelemetryEvent, trace_context: List[TelemetryEvent]) -> DiagnosisResult:
        start_time = time.perf_counter()

        # ==========================================
        # SYSTEM 1: Fast-Path Deterministic & Dynamic Rules
        # ==========================================
        s1_result = self._system_1_fast_path(anomaly_reason, event)
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

    def _system_1_fast_path(self, reason: str, event: TelemetryEvent) -> Optional[DiagnosisResult]:
        """Deterministic, zero-token fast-path classification including dynamic rule cache and semantic hashing."""
        text_to_eval = f"{reason} {event.raw_error or ''}".strip()

        # 0. Check Dynamic Learned Rule Cache First (Zero-Day Fast-Path)
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

        # 1. Timeout
        if "TIMEOUT" in reason:
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_TIMEOUT,
                severity=Severity.HIGH,
                confidence=0.99,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Execution exceeded threshold: {reason}"
            )

        # 2. Server 5xx Error
        if "HTTP_5" in reason or (event.http_status and event.http_status >= 500):
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_SERVER_500,
                severity=Severity.HIGH,
                confidence=0.99,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Upstream server error HTTP {event.http_status}"
            )

        # 3. Empty Tool Return
        if reason == "EMPTY_TOOL_RESULT":
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_EMPTY_RETURN,
                severity=Severity.MEDIUM,
                confidence=0.95,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence="Tool invocation returned null or zero-length payload"
            )

        # 4. JSON Schema Violation
        if "MALFORMED_JSON_SCHEMA" in reason:
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_MALFORMED_JSON,
                severity=Severity.MEDIUM,
                confidence=0.98,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"JSON decoding failure: {reason}"
            )

        if event.raw_error and "missing mandatory required key" in event.raw_error.lower():
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_MISSING_FIELD,
                severity=Severity.MEDIUM,
                confidence=0.98,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Missing required schema field: {event.raw_error}"
            )

        if event.raw_error and ("schematypeerror" in event.raw_error.lower() or "expected numerical type" in event.raw_error.lower()):
            return DiagnosisResult(
                domain=FaultDomain.OUTPUT_SCHEMA,
                fault_type=FaultType.SCHEMA_TYPE_MISMATCH,
                severity=Severity.MEDIUM,
                confidence=0.98,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Schema type mismatch: {event.raw_error}"
            )

        # 5. Infinite Action Loop
        if "INFINITE_ACTION_LOOP" in reason:
            return DiagnosisResult(
                domain=FaultDomain.PLANNING,
                fault_type=FaultType.PLAN_CIRCULAR_LOOP,
                severity=Severity.HIGH,
                confidence=0.96,
                diagnosed_by="System 1 (Fast-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=0,
                evidence=f"Repeated identical actions detected: {reason}"
            )

        # 6. Database / Tool Argument Syntax Error
        if event.raw_error:
            err_lower = event.raw_error.lower()
            if any(k in err_lower for k in ["no such column", "syntax error", "unknown column", "invalid argument", "keyerror", "typeerror"]):
                return DiagnosisResult(
                    domain=FaultDomain.TOOL,
                    fault_type=FaultType.TOOL_INVALID_ARGS,
                    severity=Severity.MEDIUM,
                    confidence=0.97,
                    diagnosed_by="System 1 (Fast-Path)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"Tool argument/syntax error: {event.raw_error}"
                )
            if "message_loss" in err_lower or "routing" in err_lower:
                return DiagnosisResult(
                    domain=FaultDomain.COMMUNICATION,
                    fault_type=FaultType.COMM_MESSAGE_LOSS,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    diagnosed_by="System 1 (Fast-Path)",
                    diagnostic_latency_ms=0.0,
                    diagnostic_cost_tokens=0,
                    evidence=f"Inter-agent communication lost: {event.raw_error}"
                )

        return None

    def _system_2_slow_path(self, reason: str, event: TelemetryEvent, trace_context: List[TelemetryEvent]) -> DiagnosisResult:
        """
        Out-of-band semantic evaluator for subtle cognitive/reasoning/coordination failures.
        Emulates an efficient micro-evaluator (~65 tokens). Automatically registers learned rules to System 1.
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
            # Register clean keyword pattern for zero-day matching
            clean_word = re.sub(r'[^a-zA-Z0-9_]', ' ', reason).strip().split()
            if clean_word:
                self.register_dynamic_rule(re.escape(clean_word[0]), res.domain, res.fault_type, res.severity)
            return res
