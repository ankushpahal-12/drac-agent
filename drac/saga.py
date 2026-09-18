"""
DRAC Distributed Saga & Two-Phase Outbox Staging Engine.
Solves the Irreversible External Side-Effects limitation by providing:
1. Two-Phase Outbox Staging Gate: buffers external side-effects until pre-flight verification.
2. Saga Coordinator: registers and executes backward compensating transactions on rollback.
"""
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import uuid
import time
from drac.types import OutboxStatus

@dataclass
class CompensatingActionRecord:
    record_id: str
    step: int
    tool_name: str
    compensator_name: str
    inverse_fn: Callable[..., Any]
    args: Dict[str, Any]
    idempotency_key: str
    executed: bool = False
    timestamp: float = field(default_factory=time.time)

@dataclass
class OutboxItem:
    item_id: str
    step: int
    tool_name: str
    tool_args: Dict[str, Any]
    execute_fn: Callable[[], Any]
    status: OutboxStatus = OutboxStatus.STAGED
    result: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)

class OutboxStagingGate:
    """
    Buffers non-idempotent or irreversible network calls.
    Actions remain STAGED until the pre-flight invariant verifier approves the step.
    If DRAC aborts or rolls back, uncommitted staged items are aborted without external side-effects.
    """
    def __init__(self):
        self.staged_items: List[OutboxItem] = []
        self.committed_items: List[OutboxItem] = []
        self.aborted_items: List[OutboxItem] = []

    def stage_action(self, step: int, tool_name: str, tool_args: Dict[str, Any], execute_fn: Callable[[], Any]) -> OutboxItem:
        item = OutboxItem(
            item_id=str(uuid.uuid4())[:8],
            step=step,
            tool_name=tool_name,
            tool_args=tool_args,
            execute_fn=execute_fn,
            status=OutboxStatus.STAGED
        )
        self.staged_items.append(item)
        return item

    def commit_step(self, step: int) -> List[Any]:
        """Dispatches all staged actions for the step to the physical external environment."""
        dispatched_results = []
        remaining = []
        for item in self.staged_items:
            if item.step <= step:
                # Dispatch external side-effect
                res = item.execute_fn()
                item.result = res
                item.status = OutboxStatus.COMMITTED
                self.committed_items.append(item)
                dispatched_results.append(res)
            else:
                remaining.append(item)
        self.staged_items = remaining
        return dispatched_results

    def abort_step(self, step: int) -> int:
        """Purges uncommitted staged actions for the step, preventing external side-effects."""
        aborted_count = 0
        remaining = []
        for item in self.staged_items:
            if item.step >= step:
                item.status = OutboxStatus.ABORTED
                self.aborted_items.append(item)
                aborted_count += 1
            else:
                remaining.append(item)
        self.staged_items = remaining
        return aborted_count

    def clear(self):
        self.staged_items.clear()
        self.committed_items.clear()
        self.aborted_items.clear()

class SagaCoordinator:
    """
    Maintains forward execution journal and executes registered compensating actions in reverse order on rollback.
    """
    def __init__(self):
        self.registered_compensators: Dict[str, Tuple[str, Callable]] = {}
        self.execution_journal: List[CompensatingActionRecord] = []
        self.requires_human_escalation: bool = False

    def register_compensator(self, tool_name: str, compensator_name: str, inverse_fn: Callable):
        """Maps a forward tool to its inverse compensating action."""
        self.registered_compensators[tool_name] = (compensator_name, inverse_fn)

    def record_forward_action(self, step: int, tool_name: str, args: Dict[str, Any], custom_inverse_fn: Optional[Callable] = None):
        """Records executed forward action for potential compensation."""
        if custom_inverse_fn:
            comp_name = f"undo_{tool_name}"
            inv_fn = custom_inverse_fn
        elif tool_name in self.registered_compensators:
            comp_name, inv_fn = self.registered_compensators[tool_name]
        else:
            return  # No compensator registered, action is read-only or idempotent

        rec = CompensatingActionRecord(
            record_id=str(uuid.uuid4())[:8],
            step=step,
            tool_name=tool_name,
            compensator_name=comp_name,
            inverse_fn=inv_fn,
            args=args,
            idempotency_key=str(uuid.uuid4())
        )
        self.execution_journal.append(rec)

    def compensate_rollback(self, to_step: int) -> Tuple[bool, List[str]]:
        """
        Executes compensating transactions for all actions executed after to_step in reverse chronological order.
        Returns (success, list_of_compensated_action_descriptions).
        """
        executed_compensations = []
        # Filter actions that happened after the rollback target checkpoint
        to_compensate = [rec for rec in self.execution_journal if rec.step > to_step and not rec.executed]
        # Reverse order: C_n, C_{n-1}, ..., C_1
        for rec in reversed(to_compensate):
            try:
                rec.inverse_fn(**rec.args)
                rec.executed = True
                executed_compensations.append(f"Compensated {rec.tool_name} via {rec.compensator_name}(args={rec.args})")
            except Exception as e:
                # If compensation fails, mark critical system state for human escalation
                self.requires_human_escalation = True
                executed_compensations.append(f"CRITICAL FAILURE in {rec.compensator_name}: {str(e)}")
                return False, executed_compensations

        # Clean up compensated records from active journal
        self.execution_journal = [rec for rec in self.execution_journal if not rec.executed]
        return True, executed_compensations

    def clear(self):
        self.execution_journal.clear()
        self.requires_human_escalation = False
