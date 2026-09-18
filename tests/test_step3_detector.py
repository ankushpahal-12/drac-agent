"""
Verification Step 3: Test DRAC Anomaly & Invariant Detector.
Verifies that runtime anomalies are accurately caught without false positives.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drac.types import TelemetryEvent
from drac.detector import AnomalyDetector

def test_anomaly_detector():
    print("=======================================================")
    print(" STEP 3: VERIFYING ANOMALY & INVARIANT DETECTOR        ")
    print("=======================================================")
    detector = AnomalyDetector()

    # Case 1: Healthy Event (Should NOT flag anomaly)
    healthy_event = TelemetryEvent(
        timestamp=1.0, step=1, agent_id="calc", action_type="tool_call",
        tool_name="add", tool_result=42, latency_ms=12.0
    )
    is_anom, reason = detector.observe(healthy_event)
    print(f"[Case 1] Healthy Event: Is Anomaly={is_anom} (Expected False), Reason={reason}")

    # Case 2: 504 Timeout Anomaly
    timeout_event = TelemetryEvent(
        timestamp=2.0, step=2, agent_id="calc", action_type="tool_call",
        tool_name="add", tool_result=None, latency_ms=8500.0, http_status=504
    )
    is_anom, reason = detector.observe(timeout_event)
    print(f"[Case 2] Timeout Event: Is Anomaly={is_anom} (Expected True), Reason={reason}")

    # Case 3: Empty Tool Result
    empty_event = TelemetryEvent(
        timestamp=3.0, step=3, agent_id="search", action_type="tool_call",
        tool_name="web_search", tool_result="", latency_ms=25.0
    )
    is_anom, reason = detector.observe(empty_event)
    print(f"[Case 3] Empty Tool Result: Is Anomaly={is_anom} (Expected True), Reason={reason}")

    # Case 4: Malformed JSON Output
    bad_json_event = TelemetryEvent(
        timestamp=4.0, step=4, agent_id="llm", action_type="llm_generation",
        tool_result='{"key": "value", ', latency_ms=40.0
    )
    is_anom, reason = detector.observe(bad_json_event)
    print(f"[Case 4] Malformed JSON: Is Anomaly={is_anom} (Expected True), Reason={reason}")

    # Case 5: Infinite Repetition Loop (3 identical actions)
    detector.reset()
    for i in range(3):
        loop_event = TelemetryEvent(
            timestamp=5.0 + i, step=5 + i, agent_id="db", action_type="tool_call",
            tool_name="sql_query", tool_args={"query": "SELECT * FROM orders"}, tool_result="fail"
        )
        is_anom, reason = detector.observe(loop_event)
    print(f"[Case 5] 3x Repetition Loop: Is Anomaly={is_anom} (Expected True), Reason={reason}")
    print("=======================================================\n")

if __name__ == "__main__":
    test_anomaly_detector()
