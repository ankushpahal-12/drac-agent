"""
DRAC State Invariant & Post-Condition Verifier.
Validates state integrity before resuming agent autonomy.
"""
from typing import Dict, Any, Tuple, Optional
import json

class StateVerifier:
    def __init__(self):
        pass

    def verify_remediation(self, action_result: Any, expected_schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Verify that remediation restored a valid, uncorrupted execution state.
        """
        if action_result is None:
            return False, "Remediation produced null result"

        if isinstance(action_result, str):
            if len(action_result.strip()) == 0:
                return False, "Remediation produced empty output"
            # If JSON expected
            if expected_schema or (action_result.strip().startswith("{") and action_result.strip().endswith("}")):
                try:
                    data = json.loads(action_result)
                    if expected_schema:
                        for req_key in expected_schema.get("required", []):
                            if req_key not in data:
                                return False, f"Missing required JSON key '{req_key}'"
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON remediation: {str(e)}"

        if isinstance(action_result, dict):
            if "error" in action_result and action_result["error"]:
                return False, f"Remediation contained unhandled error: {action_result['error']}"

        return True, "Post-condition verification passed successfully."
