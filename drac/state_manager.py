"""
DRAC Transactional State Manager.
Provides checkpointing, context pruning, and transactional state rollback.
"""
import copy
import time
from typing import Dict, List, Any, Optional
from drac.types import Checkpoint

class TransactionalStateManager:
    def __init__(self, max_checkpoints: int = 50):
        self.max_checkpoints = max_checkpoints
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.checkpoint_history: List[str] = []

    def create_checkpoint(self, step: int, context: List[Dict[str, Any]], env_state: Dict[str, Any], tool_state: Dict[str, Any]) -> str:
        """Create and store an isolated deep copy snapshot of system state."""
        # Evict oldest snapshot if capacity reached to bound memory
        if len(self.checkpoint_history) >= self.max_checkpoints:
            oldest = self.checkpoint_history.pop(0)
            self.checkpoints.pop(oldest, None)

        cp_id = f"cp_step_{step}_{int(time.time() * 1000)}"
        cp = Checkpoint(
            checkpoint_id=cp_id,
            step=step,
            context_history=copy.deepcopy(context),
            environment_state=copy.deepcopy(env_state),
            tool_registry_state=copy.deepcopy(tool_state),
            timestamp=time.time()
        )
        self.checkpoints[cp_id] = cp
        self.checkpoint_history.append(cp_id)
        return cp_id

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        if not self.checkpoint_history:
            return None
        return self.checkpoints[self.checkpoint_history[-1]]

    def rollback(self, checkpoint_id: Optional[str] = None, distilled_constraint: Optional[str] = None) -> Checkpoint:
        """
        Roll back state to specified or latest checkpoint.
        Prunes all subsequent corrupted trajectory tokens.
        If distilled_constraint is provided, injects it into context.
        """
        target_id = checkpoint_id or (self.checkpoint_history[-1] if self.checkpoint_history else None)
        if not target_id or target_id not in self.checkpoints:
            raise ValueError(f"Checkpoint '{target_id}' not found.")

        base_cp = self.checkpoints[target_id]
        restored = copy.deepcopy(base_cp)

        # Inject Distilled Negative Constraint (DNCS) if requested
        if distilled_constraint:
            restored.context_history.append({
                "role": "system",
                "content": distilled_constraint
            })

        return restored

    def clear(self):
        self.checkpoints.clear()
        self.checkpoint_history.clear()
