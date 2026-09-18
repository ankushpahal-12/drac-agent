"""
DRAC Core Types, Data Structures, and Enums.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time

class FaultDomain(Enum):
    TOOL = "TOOL_FAILURE"
    CONTEXT = "CONTEXT_FAILURE"
    PLANNING = "PLANNING_FAILURE"
    OUTPUT_SCHEMA = "OUTPUT_SCHEMA_FAILURE"
    COMMUNICATION = "COMMUNICATION_FAILURE"
    ENVIRONMENT = "ENVIRONMENT_FAILURE"

class FaultType(Enum):
    # Tool Faults
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_EMPTY_RETURN = "TOOL_EMPTY_RETURN"
    TOOL_CORRUPTED_VALUE = "TOOL_CORRUPTED_VALUE"
    TOOL_INVALID_ARGS = "TOOL_INVALID_ARGS"
    TOOL_SERVER_500 = "TOOL_SERVER_500"
    
    # Context Faults
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    CONTEXT_TRUNCATION = "CONTEXT_TRUNCATION"
    CONTEXT_STALE_STATE = "CONTEXT_STALE_STATE"
    
    # Planning Faults
    PLAN_CIRCULAR_LOOP = "PLAN_CIRCULAR_LOOP"
    PLAN_INVALID_SEQUENCE = "PLAN_INVALID_SEQUENCE"
    PLAN_GOAL_DRIFT = "PLAN_GOAL_DRIFT"
    
    # Output Schema Faults
    SCHEMA_MALFORMED_JSON = "SCHEMA_MALFORMED_JSON"
    SCHEMA_MISSING_FIELD = "SCHEMA_MISSING_FIELD"
    SCHEMA_TYPE_MISMATCH = "SCHEMA_TYPE_MISMATCH"
    
    # Communication Faults (MAS)
    COMM_MESSAGE_LOSS = "COMM_MESSAGE_LOSS"
    COMM_CONFLICTING_PEER = "COMM_CONFLICTING_PEER"

class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class RecoveryAction(Enum):
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    ROLLBACK_WITH_DNCS = "ROLLBACK_WITH_DNCS"
    PURE_ROLLBACK = "PURE_ROLLBACK"
    FALLBACK_TOOL = "FALLBACK_TOOL"
    ALTERNATE_MODEL = "ALTERNATE_MODEL"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"

class OutboxStatus(Enum):
    STAGED = "STAGED"
    COMMITTED = "COMMITTED"
    COMPENSATED = "COMPENSATED"
    ABORTED = "ABORTED"

@dataclass
class TelemetryEvent:
    timestamp: float
    step: int
    agent_id: str
    action_type: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    raw_error: Optional[str] = None
    http_status: Optional[int] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_idempotent: bool = True
    compensating_action: Optional[str] = None
    hmac_signature: Optional[str] = None
    epoch_token: int = 1

@dataclass
class DiagnosisResult:
    domain: FaultDomain
    fault_type: FaultType
    severity: Severity
    confidence: float
    diagnosed_by: str  # "System 1 (Fast-Path)" or "System 2 (Slow-Path)"
    diagnostic_latency_ms: float
    diagnostic_cost_tokens: int
    evidence: str

@dataclass
class Checkpoint:
    checkpoint_id: str
    step: int
    context_history: List[Dict[str, Any]]
    environment_state: Dict[str, Any]
    tool_registry_state: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class ExecutionBudget:
    max_tokens: int = 10000
    max_time_seconds: float = 60.0
    tokens_consumed: int = 0
    time_consumed: float = 0.0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_consumed)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.max_time_seconds - self.time_consumed)
