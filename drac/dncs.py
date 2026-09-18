"""
DRAC Distilled Negative Constraint Synthesizer (DNCS).
Synthesizes compact (~20 token) negative constraints upon rollback
to prevent amnesia without polluting the context window.
"""
from typing import Dict, Any
from drac.types import DiagnosisResult, FaultType

class DNCSynthesizer:
    @staticmethod
    def synthesize(diagnosis: DiagnosisResult, failed_action: Dict[str, Any]) -> str:
        """
        Synthesize a distilled negative constraint from diagnosis and failed action.
        """
        tool_name = failed_action.get("tool_name", "action")
        raw_error = failed_action.get("raw_error", "")
        args = failed_action.get("tool_args", {})

        if diagnosis.fault_type == FaultType.TOOL_INVALID_ARGS:
            # Extract actionable hint
            if "no such column" in raw_error.lower() or "unknown column" in raw_error.lower():
                return f"[CONSTRAINT: Tool '{tool_name}' failed with '{raw_error}'. Query valid schema columns only; do not re-use invalid column names.]"
            return f"[CONSTRAINT: Tool '{tool_name}' rejected arguments {args}. Modify arguments to conform to valid format.]"

        elif diagnosis.fault_type == FaultType.TOOL_TIMEOUT:
            return f"[CONSTRAINT: Tool '{tool_name}' timed out. Decompose query into a smaller batch or request less data.]"

        elif diagnosis.fault_type == FaultType.TOOL_SERVER_500:
            return f"[CONSTRAINT: Service '{tool_name}' encountered internal error 500. Use fallback tool or simplify payload.]"

        elif diagnosis.fault_type == FaultType.TOOL_EMPTY_RETURN:
            return f"[CONSTRAINT: Tool '{tool_name}' returned empty results for {args}. Reformulate search query or change filter parameters.]"

        elif diagnosis.fault_type == FaultType.SCHEMA_MALFORMED_JSON:
            return "[CONSTRAINT: Output was invalid JSON. Emit strict, valid JSON matching the requested schema with all required keys.]"

        elif diagnosis.fault_type == FaultType.PLAN_CIRCULAR_LOOP:
            return f"[CONSTRAINT: Circular repetition detected on tool '{tool_name}'. Terminate repeating loop and advance to next logical step.]"

        elif diagnosis.fault_type == FaultType.COMM_MESSAGE_LOSS:
            return f"[CONSTRAINT: Message transmission to peer failed. Resend message via alternate channel or verify recipient availability.]"

        elif diagnosis.fault_type == FaultType.CONTEXT_STALE_STATE:
            return "[CONSTRAINT: Previous observation cache is stale. Refresh state before computing final answer.]"

        elif diagnosis.fault_type == FaultType.TOOL_CORRUPTED_VALUE:
            return f"[CONSTRAINT: Tool '{tool_name}' emitted corrupted or unphysical return values. Reject output and re-compute with verified bounds.]"

        elif diagnosis.fault_type == FaultType.CONTEXT_OVERFLOW:
            return "[CONSTRAINT: Context threshold exceeded. Summarize previous turns into working memory and prune verbose history.]"

        elif diagnosis.fault_type == FaultType.CONTEXT_TRUNCATION:
            return "[CONSTRAINT: Context history was truncated. Query explicit state store to reconstruct necessary antecedents.]"

        elif diagnosis.fault_type == FaultType.PLAN_INVALID_SEQUENCE:
            return f"[CONSTRAINT: Action sequence violation. Complete prerequisite dependency before attempting '{tool_name}'.]"

        elif diagnosis.fault_type == FaultType.PLAN_GOAL_DRIFT:
            return "[CONSTRAINT: Goal drift detected. Re-align reasoning strictly with user prompt constraints and terminate extraneous sub-goals.]"

        elif diagnosis.fault_type == FaultType.SCHEMA_MISSING_FIELD:
            return "[CONSTRAINT: Output omitted mandatory fields. Verify required schema keys and include all requested properties in the output.]"

        elif diagnosis.fault_type == FaultType.SCHEMA_TYPE_MISMATCH:
            return "[CONSTRAINT: Output type mismatch. Verify JSON data types (numbers, booleans, arrays) match schema specifications.]"

        elif diagnosis.fault_type == FaultType.COMM_CONFLICTING_PEER:
            return "[CONSTRAINT: Peer contradiction detected. Initiate consensus cross-examination protocol with peer agent.]"

        else:
            return f"[CONSTRAINT: Execution failed at '{tool_name}': {diagnosis.evidence}. Adjust execution strategy.]"
