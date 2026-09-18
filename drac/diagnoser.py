"""
DRAC Dual-Process Root-Cause Diagnoser.
Bifurcates diagnosis into:
- System 1: Fast-Path Deterministic Rules (0 ms, 0 token cost)
- System 2: Slow-Path Semantic Evaluator (Lightweight out-of-band analysis)
"""
import time
import re
from typing import Optional, List, Dict, Any
from drac.types import (
    FaultDomain, FaultType, Severity, DiagnosisResult, TelemetryEvent
)

class DualProcessDiagnoser:
    def __init__(self):
        pass

    def diagnose(self, anomaly_reason: str, event: TelemetryEvent, trace_context: List[TelemetryEvent]) -> DiagnosisResult:
        start_time = time.perf_counter()

        # ==========================================
        # SYSTEM 1: Fast-Path Deterministic Diagnosis
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
        """Deterministic, zero-token fast-path classification."""
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
        Emulates an efficient micro-evaluator (~65 tokens).
        """
        diagnostic_tokens = 65
        err_text = (event.raw_error or reason).lower()

        if "stale" in err_text or "cache" in err_text or "outdated" in err_text:
            return DiagnosisResult(
                domain=FaultDomain.CONTEXT,
                fault_type=FaultType.CONTEXT_STALE_STATE,
                severity=Severity.MEDIUM,
                confidence=0.89,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Semantic state mismatch indicates stale context retention."
            )
        elif "conflict" in err_text or "contradict" in err_text or "peer" in err_text:
            return DiagnosisResult(
                domain=FaultDomain.COMMUNICATION,
                fault_type=FaultType.COMM_CONFLICTING_PEER,
                severity=Severity.HIGH,
                confidence=0.91,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Multi-agent peer disagreement detected in message exchange."
            )
        elif "goal" in err_text or "drift" in err_text or "unrelated" in err_text:
            return DiagnosisResult(
                domain=FaultDomain.PLANNING,
                fault_type=FaultType.PLAN_GOAL_DRIFT,
                severity=Severity.HIGH,
                confidence=0.86,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Reasoning path diverged from initial user prompt specifications."
            )
        elif "overflow" in err_text or "token" in err_text:
            return DiagnosisResult(
                domain=FaultDomain.CONTEXT,
                fault_type=FaultType.CONTEXT_OVERFLOW,
                severity=Severity.HIGH,
                confidence=0.94,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence="Context length threshold breach or truncation detected."
            )
        else:
            return DiagnosisResult(
                domain=FaultDomain.TOOL,
                fault_type=FaultType.TOOL_CORRUPTED_VALUE,
                severity=Severity.MEDIUM,
                confidence=0.82,
                diagnosed_by="System 2 (Slow-Path)",
                diagnostic_latency_ms=0.0,
                diagnostic_cost_tokens=diagnostic_tokens,
                evidence=f"Semantic anomaly unmapped to fast-path rules: {reason}"
            )
