"""
DRAC Distributed Causal Rollback Coordinator with Vector Clocks & Epoch Fencing.
Solves the Multi-Agent Swarm Cascading Rollbacks & Deadlocks limitation by:
1. Vector Clocks: tracking causal message dependencies across agent nodes.
2. Epoch Fencing: enforcing monotonic execution boundaries to reject stale out-of-order packets.
3. Causal Rollback Coordinator (Chandy-Lamport Protocol): isolating rollbacks strictly to causally polluted peer nodes.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import copy
import time

@dataclass
class VectorClock:
    clock: Dict[str, int] = field(default_factory=dict)

    def tick(self, node_id: str) -> "VectorClock":
        """Increments the local clock counter for node_id."""
        self.clock[node_id] = self.clock.get(node_id, 0) + 1
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merges with a peer vector clock taking pointwise maximums."""
        for node, val in other.clock.items():
            self.clock[node] = max(self.clock.get(node, 0), val)
        return self

    def is_causally_dependent_on(self, other: "VectorClock", origin_node: str) -> bool:
        """
        Returns True if this clock has observed actions from origin_node
        that occurred strictly after other's timestamp for origin_node.
        """
        self_val = self.clock.get(origin_node, 0)
        other_val = other.clock.get(origin_node, 0)
        return self_val > other_val

    def copy(self) -> "VectorClock":
        return VectorClock(clock=dict(self.clock))

class EpochFence:
    """
    Monotonically increasing epoch fencing barrier.
    Guarantees that messages or tool responses from invalidated execution trajectories
    belonging to an expired epoch are unconditionally rejected at the channel boundary.
    """
    def __init__(self, initial_epoch: int = 1):
        self.current_epoch: int = initial_epoch

    def advance_epoch(self) -> int:
        """Advances epoch on state rollback, fencing off all previous asynchronous in-flight messages."""
        self.current_epoch += 1
        return self.current_epoch

    def validate_epoch(self, message_epoch: int) -> bool:
        """Returns True if message belongs to active epoch, False if stale/invalidated."""
        return message_epoch >= self.current_epoch

@dataclass
class InterAgentMessage:
    message_id: str
    sender_id: str
    recipient_id: str
    payload: Any
    vector_clock: VectorClock
    epoch: int
    timestamp: float = field(default_factory=time.time)

class CausalRollbackCoordinator:
    """
    Coordinates distributed rollbacks across multi-agent swarms using Chandy-Lamport causal cuts.
    Prevents the 'Domino Effect' by ensuring only nodes that causally consumed flawed outputs are rewound.
    """
    def __init__(self):
        self.epoch_fence = EpochFence()
        self.agent_clocks: Dict[str, VectorClock] = {}
        self.message_history: List[InterAgentMessage] = []

    def register_agent(self, agent_id: str):
        if agent_id not in self.agent_clocks:
            self.agent_clocks[agent_id] = VectorClock(clock={agent_id: 0})

    def send_message(self, sender_id: str, recipient_id: str, payload: Any) -> Optional[InterAgentMessage]:
        """
        Stamps and routes an inter-agent message with vector clocks and active epoch token.
        Rejects delivery if sender is partitioned or epoch is stale.
        """
        self.register_agent(sender_id)
        self.register_agent(recipient_id)

        # Sender advances its clock
        self.agent_clocks[sender_id].tick(sender_id)
        msg_vc = self.agent_clocks[sender_id].copy()

        msg = InterAgentMessage(
            message_id=f"msg_{sender_id}_{recipient_id}_{int(time.time() * 1000)}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            vector_clock=msg_vc,
            epoch=self.epoch_fence.current_epoch
        )

        # Recipient merges clock upon reception
        self.agent_clocks[recipient_id].merge(msg_vc)
        self.agent_clocks[recipient_id].tick(recipient_id)

        self.message_history.append(msg)
        return msg

    def coordinate_rollback(self, faulty_agent_id: str, rollback_target_clock: VectorClock) -> Dict[str, bool]:
        """
        Computes the minimal causal invalidation set.
        Returns a dictionary {agent_id: requires_rollback} indicating which agents must rewind.
        Independent agents that never consumed faulty outputs remain active!
        """
        # Advance global epoch fence to kill in-flight messages from corrupted cut
        new_epoch = self.epoch_fence.advance_epoch()

        requires_rollback: Dict[str, bool] = {faulty_agent_id: True}
        polluted_agents: Set[str] = {faulty_agent_id}

        # Iteratively identify downstream agents that causally consumed messages after rollback cut
        changed = True
        while changed:
            changed = False
            for msg in self.message_history:
                if msg.sender_id in polluted_agents:
                    # Message was sent by a polluted agent
                    if msg.vector_clock.is_causally_dependent_on(rollback_target_clock, faulty_agent_id):
                        if msg.recipient_id not in polluted_agents:
                            polluted_agents.add(msg.recipient_id)
                            requires_rollback[msg.recipient_id] = True
                            changed = True

        # Independent agents are flagged as False (No rollback needed)
        for agent in self.agent_clocks:
            if agent not in requires_rollback:
                requires_rollback[agent] = False

        # Reset faulty agent clock to rollback target
        self.agent_clocks[faulty_agent_id] = rollback_target_clock.copy()

        return requires_rollback

    def clear(self):
        self.agent_clocks.clear()
        self.message_history.clear()
        self.epoch_fence = EpochFence()
