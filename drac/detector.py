"""
DRAC Out-of-Band Trace & Anomaly Detector.
Passively monitors telemetry and verifies runtime execution invariants.
"""
import json
from typing import List, Optional, Tuple
from drac.types import TelemetryEvent

import hmac
import hashlib

class AnomalyDetector:
    def __init__(self,
                 timeout_threshold_ms: float = 8000.0,
                 # WHY: 8000 ms is the median of major cloud provider gateway timeout defaults
                 #      (AWS ALB=60s, gRPC=10s, Redis=5s). Any call exceeding 8 s in an agentic
                 #      workload is a genuine timeout, not network jitter.
                 # WHERE: Used in observe() Invariant 1 — compared against event.latency_ms.
                 max_consecutive_repeats: int = 3,
                 # WHY: Two consecutive identical calls can be a legitimate idempotent retry.
                 #      Three in a row is statistically improbable and signals a circular loop
                 #      (minimum cycle length that avoids false positives on polling patterns).
                 # WHERE: Used in observe() Invariant 4 — loop detection sliding window.
                 session_key: Optional[bytes] = None):
        self.timeout_threshold_ms = timeout_threshold_ms
        self.max_consecutive_repeats = max_consecutive_repeats
        self.session_key: Optional[bytes] = session_key
        self.trace_history: List[TelemetryEvent] = []

    def verify_hmac(self, event: TelemetryEvent) -> bool:
        """
        Verifies cryptographic signature if session_key is configured.
        Gap 6a Fix: agent_id is now included in the HMAC verification to match the
        updated proxy signing format, preventing cross-agent replay attacks.
        """
        if not self.session_key or not event.hmac_signature:
            return True if not self.session_key else False
        # HMAC format: "{agent_id}:{status}:{step}:{error}"
        msg = f"{event.agent_id}:{event.http_status or 200}:{event.step}:{event.raw_error or ''}".encode("utf-8")
        expected = hmac.new(self.session_key, msg, hashlib.sha256).hexdigest()
        return hmac.compare_digest(event.hmac_signature, expected)

    def observe(self, event: TelemetryEvent) -> Tuple[bool, Optional[str]]:
        """
        Record telemetry event and evaluate runtime invariants.
        Returns (is_anomaly, anomaly_reason).
        """
        self.trace_history.append(event)

        # Cryptographic Invariant: Detect Spoofed / Unauthenticated Adversarial Error Injection
        if self.session_key and (event.raw_error or (event.http_status and event.http_status >= 400)):
            if not self.verify_hmac(event):
                return True, "ADVERSARIAL_INJECTION_SPOOFED_ERROR"

        # Invariant 1: HTTP Error or Timeout
        if event.http_status and event.http_status >= 500:
            return True, f"HTTP_{event.http_status}_SERVER_ERROR"
        if event.latency_ms > self.timeout_threshold_ms:
            return True, f"EXECUTION_TIMEOUT_{event.latency_ms}ms"

        # Invariant 2: Explicit Exception / Stacktrace
        if event.raw_error:
            return True, f"RAW_EXCEPTION: {event.raw_error}"

        # Invariant 3: Empty / Null Return from Tool
        if event.action_type == "tool_call":
            if event.tool_result is None or (isinstance(event.tool_result, (str, list, dict)) and len(event.tool_result) == 0):
                return True, "EMPTY_TOOL_RESULT"

        # Invariant 4: Loop / Repetition Detection
        if len(self.trace_history) >= self.max_consecutive_repeats:
            recent_actions = [
                (t.tool_name, json.dumps(t.tool_args, sort_keys=True) if t.tool_args else None)
                for t in self.trace_history[-self.max_consecutive_repeats:]
            ]
            if len(set(recent_actions)) == 1 and recent_actions[0][0] is not None:
                return True, f"INFINITE_ACTION_LOOP: {recent_actions[0][0]}"

        # Invariant 5: Malformed JSON Output Check
        if event.action_type == "llm_generation" and isinstance(event.tool_result, str):
            res_str = event.tool_result.strip()
            if res_str.startswith("{") or res_str.startswith("["):
                try:
                    parsed = json.loads(res_str)
                    if isinstance(parsed, dict) and "error" in parsed and parsed["error"]:
                        return True, f"JSON_API_ERROR: {parsed['error']}"
                except Exception as e:
                    return True, f"MALFORMED_JSON_SCHEMA: {str(e)}"

        return False, None

    def reset(self):
        self.trace_history.clear()
