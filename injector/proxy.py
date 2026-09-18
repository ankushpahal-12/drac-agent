"""
Runtime Interception Proxy for Chaos Fault Injection.
Non-invasively intercepts tool calls and executes fault injections.
"""
import time
from typing import Dict, Any, Optional, Callable, Tuple
from drac.types import FaultType, TelemetryEvent

class RuntimeFaultProxy:
    def __init__(self):
        self.active_fault: Optional[FaultType] = None
        self.fault_trigger_step: int = 1
        self.current_step: int = 0
        self.injected_count: int = 0

    def arm_fault(self, fault_type: Optional[FaultType], trigger_step: int = 1):
        """Arm the proxy to fire a specific fault at a specific execution step."""
        self.active_fault = fault_type
        self.fault_trigger_step = trigger_step
        self.current_step = 0

    def intercept_tool_call(self, agent_id: str, tool_name: str, tool_args: Dict[str, Any], execute_fn: Callable[[], Any]) -> Tuple[Any, TelemetryEvent]:
        """
        Intercepts tool execution, applies fault injection if armed, and records telemetry.
        """
        self.current_step += 1
        start_time = time.perf_counter()
        
        should_inject = (self.active_fault is not None and self.current_step == self.fault_trigger_step)
        
        raw_error = None
        http_status = 200
        result = None

        if should_inject:
            self.injected_count += 1
            fault = self.active_fault
            self.active_fault = None  # Fire once per armed trial

            if fault == FaultType.TOOL_TIMEOUT:
                time.sleep(0.05)  # Fast simulation
                latency_ms = 8500.0  # Simulated timeout
                return None, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=None,
                    raw_error="Gateway Timeout: Tool execution exceeded 8000ms",
                    http_status=504,
                    latency_ms=latency_ms
                )

            elif fault == FaultType.TOOL_SERVER_500:
                return None, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=None,
                    raw_error="Internal Server Error: Database cluster unreachable",
                    http_status=500,
                    latency_ms=35.0
                )

            elif fault == FaultType.TOOL_EMPTY_RETURN:
                return "", TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result="",
                    raw_error=None,
                    http_status=200,
                    latency_ms=20.0
                )

            elif fault == FaultType.TOOL_INVALID_ARGS:
                err_msg = "OperationalError: (1054, \"Unknown column 'order_total' in 'field list'\")"
                return None, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=None,
                    raw_error=err_msg,
                    http_status=400,
                    latency_ms=15.0
                )

            elif fault == FaultType.SCHEMA_MALFORMED_JSON:
                malformed = '{"result": 42.5, "status": "ok", '  # Unclosed JSON
                return malformed, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="llm_generation",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=malformed,
                    raw_error="JSONDecodeError: Unterminated string starting at line 1 column 32",
                    http_status=200,
                    latency_ms=40.0
                )

            elif fault == FaultType.COMM_MESSAGE_LOSS:
                err_msg = "CommunicationError: Message_loss packet dropped in MAS router"
                return None, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=None,
                    raw_error=err_msg,
                    http_status=408,
                    latency_ms=60.0
                )

            elif fault == FaultType.CONTEXT_STALE_STATE:
                stale_data = {"data": "cached_state_v1", "timestamp": "stale_outdated"}
                return stale_data, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=stale_data,
                    raw_error="ContextWarning: stale cached state returned from memory",
                    http_status=200,
                    latency_ms=10.0
                )

            elif fault == FaultType.PLAN_CIRCULAR_LOOP:
                loop_res = "Looping query repeating action..."
                return loop_res, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=loop_res,
                    raw_error="PlanInvariant: infinite circular repetition detected",
                    http_status=200,
                    latency_ms=25.0
                )

            elif fault == FaultType.TOOL_CORRUPTED_VALUE:
                corrupted = {"result": -999999.0, "unit": "corrupted_kelvin", "status": "anomaly"}
                return corrupted, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=corrupted,
                    raw_error="ValueCorruptionError: tool emitted unphysical numerical value and mismatched unit",
                    http_status=200,
                    latency_ms=18.0
                )

            elif fault == FaultType.CONTEXT_OVERFLOW:
                return None, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="llm_generation",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=None,
                    raw_error="ContextWindowExceeded: prompt token count (32850) exceeds model context window (32768)",
                    http_status=400,
                    latency_ms=10.0
                )

            elif fault == FaultType.CONTEXT_TRUNCATION:
                truncated = "[TRUNCATED... state lost]"
                return truncated, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=truncated,
                    raw_error="ContextTruncationWarning: historical trajectory was abruptly sliced",
                    http_status=200,
                    latency_ms=12.0
                )

            elif fault == FaultType.PLAN_INVALID_SEQUENCE:
                return None, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="planning",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=None,
                    raw_error="PlanPreconditionFailed: action executed prior to required dependency step completion",
                    http_status=400,
                    latency_ms=20.0
                )

            elif fault == FaultType.PLAN_GOAL_DRIFT:
                drift_res = {"irrelevant_topic": "distractor topic executed"}
                return drift_res, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="planning",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=drift_res,
                    raw_error="GoalDriftDetected: agent abandoned primary objective to explore irrelevant distractor",
                    http_status=200,
                    latency_ms=30.0
                )

            elif fault == FaultType.SCHEMA_MISSING_FIELD:
                missing_field_json = '{"status": "success"}'  # Missing required 'result' key
                return missing_field_json, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="llm_generation",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=missing_field_json,
                    raw_error="SchemaValidationError: missing mandatory required key 'result' in response",
                    http_status=200,
                    latency_ms=22.0
                )

            elif fault == FaultType.SCHEMA_TYPE_MISMATCH:
                mismatch_json = '{"result": "twenty-five"}'  # Expected float/int, got string
                return mismatch_json, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="llm_generation",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=mismatch_json,
                    raw_error="SchemaTypeError: expected numerical type for 'result', received string",
                    http_status=200,
                    latency_ms=21.0
                )

            elif fault == FaultType.COMM_CONFLICTING_PEER:
                conflict_msg = {"peer": "Analyst", "claim": "Metric is negative", "conflict": True}
                return conflict_msg, TelemetryEvent(
                    timestamp=time.time(),
                    step=self.current_step,
                    agent_id=agent_id,
                    action_type="tool_call",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=conflict_msg,
                    raw_error="PeerConflictException: peer agent emitted contradictory state assertion",
                    http_status=409,
                    latency_ms=45.0
                )

        # Standard healthy execution
        try:
            result = execute_fn()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return result, TelemetryEvent(
                timestamp=time.time(),
                step=self.current_step,
                agent_id=agent_id,
                action_type="tool_call",
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=result,
                raw_error=None,
                http_status=200,
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return None, TelemetryEvent(
                timestamp=time.time(),
                step=self.current_step,
                agent_id=agent_id,
                action_type="tool_call",
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=None,
                raw_error=str(e),
                http_status=500,
                latency_ms=latency_ms
            )
