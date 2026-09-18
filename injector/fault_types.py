"""
Chaos Fault Injection Taxonomy & Definitions.
Supports 15 fine-grained fault types across 6 taxonomy domains.
"""
from drac.types import FaultType, FaultDomain

FAULT_SPECS = {
    FaultType.TOOL_TIMEOUT: {
        "domain": FaultDomain.TOOL,
        "description": "Simulate 504 gateway timeout on tool invocation",
        "inject_handler": "inject_timeout"
    },
    FaultType.TOOL_EMPTY_RETURN: {
        "domain": FaultDomain.TOOL,
        "description": "Tool invocation returns empty string or empty list",
        "inject_handler": "inject_empty"
    },
    FaultType.TOOL_CORRUPTED_VALUE: {
        "domain": FaultDomain.TOOL,
        "description": "Tool returns corrupted semantic data or unit mismatch",
        "inject_handler": "inject_corrupted_value"
    },
    FaultType.TOOL_INVALID_ARGS: {
        "domain": FaultDomain.TOOL,
        "description": "Tool throws argument/syntax error (e.g. unknown SQL column)",
        "inject_handler": "inject_invalid_args"
    },
    FaultType.TOOL_SERVER_500: {
        "domain": FaultDomain.TOOL,
        "description": "Tool returns upstream HTTP 500 internal server error",
        "inject_handler": "inject_http_500"
    },
    FaultType.CONTEXT_OVERFLOW: {
        "domain": FaultDomain.CONTEXT,
        "description": "Token count exceeds context window threshold",
        "inject_handler": "inject_context_overflow"
    },
    FaultType.CONTEXT_TRUNCATION: {
        "domain": FaultDomain.CONTEXT,
        "description": "Truncate historical context abruptly",
        "inject_handler": "inject_context_truncation"
    },
    FaultType.CONTEXT_STALE_STATE: {
        "domain": FaultDomain.CONTEXT,
        "description": "Return stale cached state from a previous step",
        "inject_handler": "inject_stale_state"
    },
    FaultType.PLAN_CIRCULAR_LOOP: {
        "domain": FaultDomain.PLANNING,
        "description": "Force agent into an infinite repetitive loop",
        "inject_handler": "inject_circular_loop"
    },
    FaultType.PLAN_INVALID_SEQUENCE: {
        "domain": FaultDomain.PLANNING,
        "description": "Invert required sequential order of subtasks",
        "inject_handler": "inject_invalid_sequence"
    },
    FaultType.PLAN_GOAL_DRIFT: {
        "domain": FaultDomain.PLANNING,
        "description": "Inject distractors that induce goal drift",
        "inject_handler": "inject_goal_drift"
    },
    FaultType.SCHEMA_MALFORMED_JSON: {
        "domain": FaultDomain.OUTPUT_SCHEMA,
        "description": "Emit unparseable malformed JSON output",
        "inject_handler": "inject_malformed_json"
    },
    FaultType.SCHEMA_MISSING_FIELD: {
        "domain": FaultDomain.OUTPUT_SCHEMA,
        "description": "Omit mandatory required field in output JSON",
        "inject_handler": "inject_missing_field"
    },
    FaultType.COMM_MESSAGE_LOSS: {
        "domain": FaultDomain.COMMUNICATION,
        "description": "Drop message packet between collaborative agents in MAS",
        "inject_handler": "inject_message_loss"
    },
    FaultType.COMM_CONFLICTING_PEER: {
        "domain": FaultDomain.COMMUNICATION,
        "description": "Peer agent emits contradictory claim or state assertion",
        "inject_handler": "inject_conflicting_peer"
    },
}
