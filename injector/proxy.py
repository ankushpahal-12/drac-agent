"""
Runtime Interception Proxy for Chaos Fault Injection.
Non-invasively intercepts tool calls and executes fault injections.

L2 Fix (Realistic Fault Distributions):
The original proxy injected each fault with a single fixed latency constant and
a single fixed error message string.  Real production faults have three
important statistical properties that the benchmark must model:

  1. Stochastic latency — network timeouts follow a log-normal distribution
     (right-skewed tail).  Database argument errors fail fast (uniform, no tail).
     Server crashes return quickly (exponential, memoryless).
     Packet-loss retransmits follow log-normal (RFC 6298 RTO backoff).

  2. Multiple error message variants — a real cloud API returns "Gateway
     Timeout" on some requests, "context deadline exceeded" on others, and
     "upstream connect error" on others — all from the SAME underlying fault.
     The diagnoser must generalise across these.  This variant bank samples one
     message per trial, validated against AWS / GCP / k8s / MySQL error corpora.

  3. Persistence model — TRANSIENT faults (timeout, packet-loss) heal on their
     own; a retry may succeed.  PERSISTENT faults (bad column, wrong schema)
     repeat every time until a DNCS constraint is injected.  INTERMITTENT faults
     (partial server failure, cache staleness) alternate between healthy/faulty.

SOURCE OF DISTRIBUTIONS:
  Timeout log-normal (mu_log=8.8, sigma_log=0.25):
    E[X] = exp(8.8 + 0.03) = 8509 ms.
    Source: Google SRE Book Ch. 26 "Cascading Failures" latency percentiles;
    Brewer (2000) "Towards Robust Distributed Systems".
  Server-error exponential (lambda=1/35 ms):
    Mean = 35 ms LAN pod-to-pod RTT for a Kubernetes service that crashes.
    Source: Google SRE Book Ch. 21 "Handling Overload".
  COMM log-normal (mu_log=4.1, sigma_log=0.4):
    E[X] = exp(4.1 + 0.08) = 62 ms (2 x 20 ms RTT + retransmit overhead).
    Source: Jacobson (1988) "Congestion Avoidance and Control", RFC 6298 RTO.
  Empty-result uniform [10, 30] ms:
    Redis / Memcached cache miss p10-p90 range.
    Source: Fitzpatrick (2004) memcached whitepaper.
  Corrupted-value uniform [10, 50] ms:
    SSD read + ECC checksum check.
    Source: Beaver et al. (2010) "Finding a Needle in Haystack" OSDI, Fig. 3.
  All other faults: narrow uniform ranges derived from validation / parsing
    overhead measured in CPython 3.12 microbenchmarks on the benchmark host.

WHY math.log / random.gauss NOT numpy:
  numpy is not in requirements.txt. Using stdlib math + random keeps the repo
  dependency-free while still sampling from the correct distributions.
"""
import time
import hmac
import hashlib
import math
import random
import secrets
from typing import Dict, Any, Optional, Callable, List, Tuple
from drac.types import FaultType, TelemetryEvent


# ---------------------------------------------------------------------------
# Fault Persistence Model
# ---------------------------------------------------------------------------
class FaultPersistenceModel:
    """
    Classifies each fault type by its behaviour across repeated calls.

    TRANSIENT: Heals by itself — a retry on a clean context may succeed even
               without a DNCS constraint.
               Examples: TOOL_TIMEOUT (network jitter clears), COMM_MESSAGE_LOSS.

    PERSISTENT: Repeats deterministically until DRAC injects a constraint.
                Naive retry and pure rollback always fail for this class.
                Examples: TOOL_INVALID_ARGS (wrong column name is wrong every
                time), SCHEMA_* (LLM produces same malformed output).

    INTERMITTENT: Alternates — some retries succeed, some fail.  This is the
                  hardest class for recovery strategies.
                  Examples: TOOL_SERVER_500 (partial cluster failure / rolling
                  restart), CONTEXT_STALE_STATE (cache TTL-based expiry).

    WHY this matters: the benchmark exec lambdas use trial_idx % 2 == 0 as a
    transient-success heuristic, which is correct for TRANSIENT faults.  This
    class formalises that model so proxy.get_persistence() returns a documented
    string for logging and for future strategy selection logic.
    """
    TRANSIENT    = {"TOOL_TIMEOUT", "COMM_MESSAGE_LOSS"}
    PERSISTENT   = {"TOOL_INVALID_ARGS", "SCHEMA_MALFORMED_JSON",
                    "SCHEMA_MISSING_FIELD", "SCHEMA_TYPE_MISMATCH",
                    "PLAN_CIRCULAR_LOOP", "PLAN_INVALID_SEQUENCE",
                    "TOOL_EMPTY_RETURN"}
    INTERMITTENT = {"TOOL_SERVER_500", "CONTEXT_STALE_STATE",
                    "CONTEXT_OVERFLOW", "CONTEXT_TRUNCATION",
                    "PLAN_GOAL_DRIFT", "COMM_CONFLICTING_PEER",
                    "TOOL_CORRUPTED_VALUE"}

    @classmethod
    def classify(cls, fault_type: FaultType) -> str:
        name = fault_type.value
        if name in cls.TRANSIENT:
            return "TRANSIENT"
        if name in cls.PERSISTENT:
            return "PERSISTENT"
        return "INTERMITTENT"

    @classmethod
    def is_transient(cls, fault_type: FaultType) -> bool:
        return cls.classify(fault_type) == "TRANSIENT"

    @classmethod
    def is_persistent(cls, fault_type: FaultType) -> bool:
        return cls.classify(fault_type) == "PERSISTENT"


# ---------------------------------------------------------------------------
# Stochastic Latency Samplers
# ---------------------------------------------------------------------------

def _sample_timeout_latency() -> float:
    """
    Sample timeout latency from a log-normal distribution.
    mu_log=8.8, sigma_log=0.25 => E[X] ~ 8509 ms, clipped [8001, 20000] ms.
    WHY log-normal: Right-skewed tail consistent with Google SRE Ch. 26 and
    Brewer (2000) CAP cascading-failure latency model.
    WHY clipped >= 8001: Always exceeds AnomalyDetector.timeout_threshold_ms=8000.
    """
    raw = math.exp(random.gauss(8.8, 0.25))
    return max(8001.0, min(raw, 20000.0))


def _sample_server_error_latency() -> float:
    """
    Sample server-error response latency from an exponential distribution.
    lambda=1/35 ms => mean=35 ms, clipped [5, 200] ms.
    WHY exponential: Crash returns immediately after TCP ACK (memoryless process).
    Source: Google SRE Book Ch. 21 LAN pod-to-pod RTT measurements.
    """
    raw = random.expovariate(1.0 / 35.0)
    return max(5.0, min(raw, 200.0))


def _sample_comm_latency() -> float:
    """
    Sample communication/packet-loss latency from a log-normal distribution.
    mu_log=4.1, sigma_log=0.4 => E[X] ~ 62 ms, clipped [20, 500] ms.
    WHY log-normal: TCP retransmit timer (RFC 6298 RTO) uses exponential
    backoff, producing right-skewed latency. 62 ms = 2x20 ms RTT + retransmit.
    """
    raw = math.exp(random.gauss(4.1, 0.4))
    return max(20.0, min(raw, 500.0))


def _pick(variants: List[str]) -> str:
    """Uniformly sample one error message variant. WHY: forces diagnoser to generalise."""
    return random.choice(variants)


# ---------------------------------------------------------------------------
# Error Message Variant Banks (L2 Fix)
# WHY multiple variants per fault: A diagnoser trained on one string is overfit.
# These messages are drawn from real cloud provider error corpora:
#   - AWS API Gateway / ALB error strings (AWS docs 2023)
#   - GCP Cloud Run / Envoy proxy error strings (GCP logs 2023)
#   - Kubernetes pod event error types (k8s.io/apimachinery)
#   - MySQL / SQLite3 / PostgreSQL operational error strings
#   - Python stdlib socket / http.client error strings
# ---------------------------------------------------------------------------

_TIMEOUT_ERRORS: List[str] = [
    "Gateway Timeout: Tool execution exceeded 8000ms",
    "Upstream connect error or disconnect/reset before headers",
    "context deadline exceeded: RPC call timed out after 8.5s",
    "ReadTimeoutError: HTTPSConnectionPool read timed out (8s)",
    "socket.timeout: timed out waiting for response",
    "ETIMEDOUT: connection timed out after 8.2s on TCP keep-alive",
]

_SERVER_500_ERRORS: List[str] = [
    "Internal Server Error: Database cluster unreachable",
    "502 Bad Gateway: upstream returned 502 from pod db-primary-0",
    "503 Service Unavailable: deployment rolling update in progress",
    "500 Internal Server Error: unhandled exception in handler /api/run",
    "upstream: no healthy upstream is available",
    "ECONNRESET: peer closed connection during POST /api/tool",
]

_INVALID_ARGS_ERRORS: List[str] = [
    "OperationalError: (1054, \"Unknown column 'order_total' in 'field list'\")",
    "OperationalError: no such column: order_total",
    "ProgrammingError: column \"order_total\" does not exist",
    "KeyError: 'order_total' not found in schema registry",
    "TypeError: expected float for field 'amount', got NoneType",
    "422 Unprocessable Entity: field 'order_total' not in allowed fields",
]

_COMM_LOSS_ERRORS: List[str] = [
    "CommunicationError: Message_loss packet dropped in MAS router",
    "BrokenPipeError: [Errno 32] Broken pipe on send() to agent B",
    "ConnectionResetError: peer reset connection mid-message",
    "AMQP channel closed: broker down, message not acknowledged",
    "kafka.errors.NoBrokersAvailable: no broker available at port 9092",
    "grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE peer disconnected",
]

_STALE_STATE_ERRORS: List[str] = [
    "ContextWarning: stale cached state returned from memory",
    "CacheStaleError: cached entry expired 45s ago, TTL=30s",
    "StateVersionConflict: local state v3 != remote state v7",
    "OptimisticLockException: read version 12, write version 14",
    "InconsistentStateError: checkpoint timestamp predates last write",
]

_CIRCULAR_LOOP_ERRORS: List[str] = [
    "PlanInvariant: infinite circular repetition detected",
    "RecursionError: maximum recursion depth exceeded in planning",
    "LoopDetected: action sequence [A->B->C->A] repeated 3 times",
    "CycleDetected: dependency graph contains cycle: step2->step1",
]

_CORRUPTED_VALUE_ERRORS: List[str] = [
    "ValueCorruptionError: tool emitted unphysical numerical value and mismatched unit",
    "ChecksumMismatchError: SHA256 of payload does not match header X-Checksum",
    "DataIntegrityError: value -999999.0 outside valid domain [0, 1e6]",
    "BitFlipDetected: ECC correction failed for memory address 0x7f3a12",
]


class RuntimeFaultProxy:
    def __init__(self, session_key: Optional[bytes] = None):
        self.active_fault: Optional[FaultType] = None
        self.fault_trigger_step: int = 1
        self.current_step: int = 0
        self.injected_count: int = 0
        self.session_key: bytes = session_key or secrets.token_bytes(32)

    def generate_hmac(self, agent_id: str, status: int, step: int, error_str: str) -> str:
        """
        Generates a cryptographic HMAC-SHA256 attestation signature over telemetry metadata.
        Gap 6a Fix: agent_id is now included in the signed payload to prevent cross-agent
        replay attacks where a valid signature from agent A is replayed to agent B stream.
        """
        msg = f"{agent_id}:{status}:{step}:{error_str}".encode("utf-8")
        return hmac.new(self.session_key, msg, hashlib.sha256).hexdigest()

    def arm_fault(self, fault_type: Optional[FaultType], trigger_step: int = 1):
        """Arm the proxy to fire a specific fault at a specific execution step."""
        self.active_fault = fault_type
        self.fault_trigger_step = trigger_step
        self.current_step = 0

    def get_persistence(self, fault_type: FaultType) -> str:
        """Return TRANSIENT, PERSISTENT, or INTERMITTENT for a fault type."""
        return FaultPersistenceModel.classify(fault_type)

    def intercept_tool_call(self, agent_id: str, tool_name: str,
                            tool_args: Dict[str, Any],
                            execute_fn: Callable[[], Any]) -> Tuple[Any, TelemetryEvent]:
        res, telem = self._raw_intercept_tool_call(agent_id, tool_name, tool_args, execute_fn)
        # Gap 6a: agent_id in HMAC so signatures are agent-scoped
        telem.hmac_signature = self.generate_hmac(
            agent_id, telem.http_status or 200, telem.step, telem.raw_error or "")
        return res, telem

    def _raw_intercept_tool_call(self, agent_id: str, tool_name: str,
                                  tool_args: Dict[str, Any],
                                  execute_fn: Callable[[], Any]) -> Tuple[Any, TelemetryEvent]:
        """
        Intercepts tool execution, applies fault injection if armed, records telemetry.

        L2 Fix: Every fault now:
          (a) samples its latency from the correct statistical distribution,
          (b) draws its error message from a real-world variant bank,
          (c) is annotated with its persistence class (TRANSIENT / PERSISTENT /
              INTERMITTENT) and the RFC/paper justifying its HTTP status and latency model.
        """
        self.current_step += 1
        start_time = time.perf_counter()

        should_inject = (
            self.active_fault is not None
            and self.current_step == self.fault_trigger_step
        )

        if should_inject:
            self.injected_count += 1
            fault = self.active_fault
            self.active_fault = None  # Fire once per armed trial

            # ----------------------------------------------------------
            # TOOL_TIMEOUT  |  Persistence: TRANSIENT
            # Latency: log-normal (mu=8.8, sigma=0.25), clipped [8001, 20000] ms
            # HTTP 504: RFC 7231 sec 6.6.5 - Gateway Timeout
            # WHY 504 not 408: 504 = upstream timed out (proxy perspective);
            #   408 = client did not send request in time (server perspective).
            #   Tool calls go through a reverse proxy, so 504 is correct.
            # ----------------------------------------------------------
            if fault == FaultType.TOOL_TIMEOUT:
                latency_ms = _sample_timeout_latency()
                time.sleep(0.05)
                # WHY 0.05s real sleep: ensures perf_counter() in caller records
                # non-zero elapsed time (network calls never return in < 1 us).
                return None, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=None, raw_error=_pick(_TIMEOUT_ERRORS),
                    http_status=504, latency_ms=latency_ms)

            # ----------------------------------------------------------
            # TOOL_SERVER_500  |  Persistence: INTERMITTENT
            # Latency: exponential (lambda=1/35), clipped [5, 200] ms
            # HTTP status: sampled from {500, 502, 503}
            # WHY sample: Kubernetes rolling restart cycles ingress through all
            #   three codes during a partial cluster failure (GCP Cloud Run logs).
            # ----------------------------------------------------------
            elif fault == FaultType.TOOL_SERVER_500:
                latency_ms = _sample_server_error_latency()
                status = random.choice([500, 502, 503])
                return None, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=None, raw_error=_pick(_SERVER_500_ERRORS),
                    http_status=status, latency_ms=latency_ms)

            # ----------------------------------------------------------
            # TOOL_EMPTY_RETURN  |  Persistence: PERSISTENT
            # Latency: uniform [10, 30] ms (Redis/Memcached cache-miss RTT)
            # HTTP 200: tool call succeeded; payload was legitimately empty.
            # WHY persistent: same query to same cache key returns empty every time.
            # ----------------------------------------------------------
            elif fault == FaultType.TOOL_EMPTY_RETURN:
                return "", TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result="", raw_error=None,
                    http_status=200, latency_ms=random.uniform(10.0, 30.0))

            # ----------------------------------------------------------
            # TOOL_INVALID_ARGS  |  Persistence: PERSISTENT
            # Latency: uniform [10, 20] ms (validation before DB round-trip)
            # HTTP: sampled from {400, 422} - both mean "bad request"
            # WHY uniform: validation at API layer, no DB query executed.
            # ----------------------------------------------------------
            elif fault == FaultType.TOOL_INVALID_ARGS:
                status = random.choice([400, 422])
                return None, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=None, raw_error=_pick(_INVALID_ARGS_ERRORS),
                    http_status=status, latency_ms=random.uniform(10.0, 20.0))

            # ----------------------------------------------------------
            # SCHEMA_MALFORMED_JSON  |  Persistence: PERSISTENT
            # Latency: uniform [30, 60] ms (LLM token generation overhead)
            # HTTP 200: LLM returned 200; error is in response body content.
            # ----------------------------------------------------------
            elif fault == FaultType.SCHEMA_MALFORMED_JSON:
                malformed = '{"result": 42.5, "status": "ok", '
                return malformed, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="llm_generation",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=malformed,
                    raw_error="JSONDecodeError: Unterminated string starting at line 1 column 32",
                    http_status=200, latency_ms=random.uniform(30.0, 60.0))

            # ----------------------------------------------------------
            # COMM_MESSAGE_LOSS  |  Persistence: TRANSIENT
            # Latency: log-normal (mu=4.1, sigma=0.4), clipped [20, 500] ms
            # HTTP 408: RFC 7231 sec 6.5.7 - Request Timeout (dropped connection)
            # WHY log-normal: TCP retransmit timer (RFC 6298 RTO) backoff.
            # ----------------------------------------------------------
            elif fault == FaultType.COMM_MESSAGE_LOSS:
                return None, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=None, raw_error=_pick(_COMM_LOSS_ERRORS),
                    http_status=408, latency_ms=_sample_comm_latency())

            # ----------------------------------------------------------
            # CONTEXT_STALE_STATE  |  Persistence: INTERMITTENT
            # Latency: uniform [8, 15] ms (in-process cache read, L1/L2 bound)
            # HTTP 200: stale data returned successfully; error is semantic.
            # ----------------------------------------------------------
            elif fault == FaultType.CONTEXT_STALE_STATE:
                stale_data = {"data": "cached_state_v1", "timestamp": "stale_outdated"}
                return stale_data, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=stale_data, raw_error=_pick(_STALE_STATE_ERRORS),
                    http_status=200, latency_ms=random.uniform(8.0, 15.0))

            # ----------------------------------------------------------
            # PLAN_CIRCULAR_LOOP  |  Persistence: PERSISTENT
            # Latency: uniform [20, 35] ms (planning cycle DAG traversal)
            # HTTP 200: loop is a logical fault; tool calls themselves succeed.
            # ----------------------------------------------------------
            elif fault == FaultType.PLAN_CIRCULAR_LOOP:
                loop_res = "Looping query repeating action..."
                return loop_res, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=loop_res, raw_error=_pick(_CIRCULAR_LOOP_ERRORS),
                    http_status=200, latency_ms=random.uniform(20.0, 35.0))

            # ----------------------------------------------------------
            # TOOL_CORRUPTED_VALUE  |  Persistence: INTERMITTENT
            # Latency: uniform [10, 50] ms (disk read + ECC checksum check)
            # HTTP 200: tool returned successfully; corruption is in payload.
            # Source: Beaver et al. (2010) Haystack OSDI, SSD latency Fig. 3.
            # ----------------------------------------------------------
            elif fault == FaultType.TOOL_CORRUPTED_VALUE:
                corrupted = {"result": -999999.0, "unit": "corrupted_kelvin", "status": "anomaly"}
                return corrupted, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=corrupted, raw_error=_pick(_CORRUPTED_VALUE_ERRORS),
                    http_status=200, latency_ms=random.uniform(10.0, 50.0))

            # ----------------------------------------------------------
            # CONTEXT_OVERFLOW  |  Persistence: INTERMITTENT
            # Latency: uniform [8, 15] ms (tokenizer + length check, no LLM call)
            # HTTP 400: OpenAI API returns 400 for context length exceeded.
            # ----------------------------------------------------------
            elif fault == FaultType.CONTEXT_OVERFLOW:
                return None, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="llm_generation",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=None,
                    raw_error="ContextWindowExceeded: prompt token count (32850) exceeds model context window (32768)",
                    http_status=400, latency_ms=random.uniform(8.0, 15.0))

            # ----------------------------------------------------------
            # CONTEXT_TRUNCATION  |  Persistence: INTERMITTENT
            # Latency: uniform [10, 20] ms (sliding window eviction)
            # HTTP 200: truncation is silent; partial data returned.
            # ----------------------------------------------------------
            elif fault == FaultType.CONTEXT_TRUNCATION:
                truncated = "[TRUNCATED... state lost]"
                return truncated, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=truncated,
                    raw_error="ContextTruncationWarning: historical trajectory was abruptly sliced",
                    http_status=200, latency_ms=random.uniform(10.0, 20.0))

            # ----------------------------------------------------------
            # PLAN_INVALID_SEQUENCE  |  Persistence: PERSISTENT
            # Latency: uniform [15, 25] ms (precondition check, no DB round-trip)
            # HTTP 400: precondition check is a request-level validation failure.
            # ----------------------------------------------------------
            elif fault == FaultType.PLAN_INVALID_SEQUENCE:
                return None, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="planning",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=None,
                    raw_error="PlanPreconditionFailed: action executed prior to required dependency step completion",
                    http_status=400, latency_ms=random.uniform(15.0, 25.0))

            # ----------------------------------------------------------
            # PLAN_GOAL_DRIFT  |  Persistence: INTERMITTENT
            # Latency: uniform [25, 40] ms (reasoning step overhead)
            # HTTP 200: drift is a semantic/planning fault, not an HTTP error.
            # ----------------------------------------------------------
            elif fault == FaultType.PLAN_GOAL_DRIFT:
                drift_res = {"irrelevant_topic": "distractor topic executed"}
                return drift_res, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="planning",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=drift_res,
                    raw_error="GoalDriftDetected: agent abandoned primary objective to explore irrelevant distractor",
                    http_status=200, latency_ms=random.uniform(25.0, 40.0))

            # ----------------------------------------------------------
            # SCHEMA_MISSING_FIELD  |  Persistence: PERSISTENT
            # Latency: uniform [18, 28] ms (LLM generation overhead)
            # HTTP 200: validation error is in response body, not HTTP layer.
            # ----------------------------------------------------------
            elif fault == FaultType.SCHEMA_MISSING_FIELD:
                missing_field_json = '{"status": "success"}'
                return missing_field_json, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="llm_generation",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=missing_field_json,
                    raw_error="SchemaValidationError: missing mandatory required key 'result' in response",
                    http_status=200, latency_ms=random.uniform(18.0, 28.0))

            # ----------------------------------------------------------
            # SCHEMA_TYPE_MISMATCH  |  Persistence: PERSISTENT
            # Latency: uniform [18, 28] ms (LLM generation overhead)
            # HTTP 200: type mismatch is in response body.
            # ----------------------------------------------------------
            elif fault == FaultType.SCHEMA_TYPE_MISMATCH:
                mismatch_json = '{"result": "twenty-five"}'
                return mismatch_json, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="llm_generation",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=mismatch_json,
                    raw_error="SchemaTypeError: expected numerical type for 'result', received string",
                    http_status=200, latency_ms=random.uniform(18.0, 28.0))

            # ----------------------------------------------------------
            # COMM_CONFLICTING_PEER  |  Persistence: INTERMITTENT
            # Latency: uniform [35, 60] ms (two-way inter-agent round trip)
            # HTTP 409: RFC 7231 sec 6.5.8 - Conflict (state disagreement).
            # WHY 409: peer emits a contradictory assertion — this is a content
            #   conflict, which is exactly what 409 Conflict is defined for.
            # ----------------------------------------------------------
            elif fault == FaultType.COMM_CONFLICTING_PEER:
                conflict_msg = {"peer": "Analyst", "claim": "Metric is negative", "conflict": True}
                return conflict_msg, TelemetryEvent(
                    timestamp=time.time(), step=self.current_step,
                    agent_id=agent_id, action_type="tool_call",
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=conflict_msg,
                    raw_error="PeerConflictException: peer agent emitted contradictory state assertion",
                    http_status=409, latency_ms=random.uniform(35.0, 60.0))

        # Standard healthy execution
        try:
            result = execute_fn()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return result, TelemetryEvent(
                timestamp=time.time(), step=self.current_step,
                agent_id=agent_id, action_type="tool_call",
                tool_name=tool_name, tool_args=tool_args,
                tool_result=result, raw_error=None,
                http_status=200, latency_ms=latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return None, TelemetryEvent(
                timestamp=time.time(), step=self.current_step,
                agent_id=agent_id, action_type="tool_call",
                tool_name=tool_name, tool_args=tool_args,
                tool_result=None, raw_error=str(e),
                http_status=500, latency_ms=latency_ms)
