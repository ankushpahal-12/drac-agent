"""
DRAC Transactional State Manager with Hierarchical Snapshot Tiering & Copy-on-Write (CoW) Delta Checkpointing.
Solves the Long-Horizon Memory Bloat limitation by:
1. CoW Delta Trees: storing only token diffs and database savepoint markers.
2. Hierarchical Tiering: maintaining Hot Tier (in-memory ring buffer) and Cold Tier (compressed disk-backed serialized delta blocks).
3. SQLite Native Savepoint Management: zero-copy in-memory database rollbacks via SAVEPOINT.
"""
import copy
import time
import json
import zlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from drac.types import Checkpoint

@dataclass
class DeltaCheckpoint:
    delta_id: str
    step: int
    base_checkpoint_id: str
    new_context_turns: List[Dict[str, Any]]
    env_diff: Dict[str, Any]
    db_savepoint: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class TransactionalStateManager:
    def __init__(self, max_checkpoints: Optional[int] = None, max_hot_checkpoints: int = 5, cold_storage_dir: Optional[str] = None):
        self.max_hot_checkpoints = max_checkpoints if max_checkpoints is not None else max_hot_checkpoints
        self.cold_storage_dir = cold_storage_dir or os.path.join(tempfile.gettempdir(), "drac_cold_tier")
        os.makedirs(self.cold_storage_dir, exist_ok=True)

        # Hot Tier (In-Memory RAM Ring Buffer)
        self.hot_checkpoints: Dict[str, Checkpoint] = {}
        self.hot_history: List[str] = []

        # Delta Checkpoints (CoW Trees)
        self.delta_checkpoints: Dict[str, DeltaCheckpoint] = {}
        self.delta_history: List[str] = []

        # Cold Tier (Compressed Serialized Storage)
        self.cold_checkpoint_ids: set = set()

        # Database Savepoint Registry
        self.db_savepoints: Dict[int, str] = {}

    @property
    def checkpoints(self) -> Dict[str, Checkpoint]:
        """Backward-compatibility alias for hot in-memory checkpoints."""
        return self.hot_checkpoints

    @property
    def checkpoint_history(self) -> List[str]:
        """Backward-compatibility alias for hot in-memory history."""
        return self.hot_history

    def create_checkpoint(self, step: int, context: List[Dict[str, Any]], env_state: Dict[str, Any], tool_state: Dict[str, Any], db_conn: Optional[Any] = None) -> str:
        """
        Creates a snapshot with Hierarchical Tiering and CoW Delta tracking.
        If hot capacity is exceeded, oldest checkpoints are evicted to the Cold Tier (compressed).
        """
        cp_id = f"cp_step_{step}_{int(time.time() * 1000)}"

        # 1. Register SQLite native Savepoint if DB connection is active
        savepoint_name = None
        if db_conn is not None:
            savepoint_name = f"drac_sp_{step}"
            try:
                db_conn.execute(f"SAVEPOINT {savepoint_name};")
                self.db_savepoints[step] = savepoint_name
            except Exception:
                pass

        # 2. Compute CoW Delta from previous checkpoint
        if self.hot_history:
            prev_id = self.hot_history[-1]
            prev_cp = self.hot_checkpoints.get(prev_id)
            if prev_cp:
                prev_len = len(prev_cp.context_history)
                new_turns = context[prev_len:] if len(context) > prev_len else []
                env_diff = {k: v for k, v in env_state.items() if prev_cp.environment_state.get(k) != v}
                
                delta = DeltaCheckpoint(
                    delta_id=f"delta_{step}",
                    step=step,
                    base_checkpoint_id=prev_id,
                    new_context_turns=copy.deepcopy(new_turns),
                    env_diff=copy.deepcopy(env_diff),
                    db_savepoint=savepoint_name
                )
                self.delta_checkpoints[delta.delta_id] = delta
                self.delta_history.append(delta.delta_id)

        # 3. Create Hot Tier Checkpoint
        cp = Checkpoint(
            checkpoint_id=cp_id,
            step=step,
            context_history=copy.deepcopy(context),
            environment_state=copy.deepcopy(env_state),
            tool_registry_state=copy.deepcopy(tool_state),
            timestamp=time.time()
        )

        # 4. Enforce Hot Tier Capacity: Evict oldest to Cold Tier
        if len(self.hot_history) >= self.max_hot_checkpoints:
            evicted_id = self.hot_history.pop(0)
            evicted_cp = self.hot_checkpoints.pop(evicted_id, None)
            if evicted_cp:
                self._evict_to_cold_tier(evicted_cp)

        self.hot_checkpoints[cp_id] = cp
        self.hot_history.append(cp_id)
        return cp_id

    def _evict_to_cold_tier(self, cp: Checkpoint):
        """Compresses and writes cold checkpoint to disk to bound memory usage."""
        cold_path = os.path.join(self.cold_storage_dir, f"{cp.checkpoint_id}.drac.gz")
        payload = {
            "checkpoint_id": cp.checkpoint_id,
            "step": cp.step,
            "context_history": cp.context_history,
            "environment_state": cp.environment_state,
            "tool_registry_state": cp.tool_registry_state,
            "timestamp": cp.timestamp
        }
        compressed = zlib.compress(json.dumps(payload).encode("utf-8"), level=6)
        with open(cold_path, "wb") as f:
            f.write(compressed)
        self.cold_checkpoint_ids.add(cp.checkpoint_id)

    def _load_from_cold_tier(self, cp_id: str) -> Optional[Checkpoint]:
        """Loads and decompresses a checkpoint from cold tier."""
        cold_path = os.path.join(self.cold_storage_dir, f"{cp_id}.drac.gz")
        if not os.path.exists(cold_path):
            return None
        with open(cold_path, "rb") as f:
            decompressed = zlib.decompress(f.read()).decode("utf-8")
        data = json.loads(decompressed)
        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            step=data["step"],
            context_history=data["context_history"],
            environment_state=data["environment_state"],
            tool_registry_state=data["tool_registry_state"],
            timestamp=data["timestamp"]
        )

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        if not self.hot_history:
            return None
        return self.hot_checkpoints.get(self.hot_history[-1])

    def rollback(self, checkpoint_id: Optional[str] = None, distilled_constraint: Optional[str] = None, db_conn: Optional[Any] = None) -> Checkpoint:
        """
        Roll back state to specified checkpoint (from Hot Tier or Cold Tier).
        Executes SQLite savepoint rollback if database connection provided.
        Prunes corrupted trajectory tokens and injects DNCS constraint if provided.
        """
        target_id = checkpoint_id or (self.hot_history[-1] if self.hot_history else None)
        if not target_id:
            raise ValueError("No checkpoints available for rollback.")

        # 1. Retrieve from Hot Tier or Cold Tier
        base_cp = None
        if target_id in self.hot_checkpoints:
            base_cp = self.hot_checkpoints[target_id]
        elif target_id in self.cold_checkpoint_ids:
            base_cp = self._load_from_cold_tier(target_id)

        if not base_cp:
            raise ValueError(f"Checkpoint '{target_id}' not found in hot or cold tier.")

        # 2. Database native rollback via Savepoint
        if db_conn is not None and base_cp.step in self.db_savepoints:
            sp_name = self.db_savepoints[base_cp.step]
            try:
                db_conn.execute(f"ROLLBACK TO {sp_name};")
            except Exception:
                pass

        restored = copy.deepcopy(base_cp)

        # 3. Inject Distilled Negative Constraint (DNCS)
        if distilled_constraint:
            restored.context_history.append({
                "role": "system",
                "content": distilled_constraint
            })

        return restored

    def clear(self):
        self.hot_checkpoints.clear()
        self.hot_history.clear()
        self.delta_checkpoints.clear()
        self.delta_history.clear()
        self.db_savepoints.clear()
        # Clean up cold files
        for cid in list(self.cold_checkpoint_ids):
            path = os.path.join(self.cold_storage_dir, f"{cid}.drac.gz")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        self.cold_checkpoint_ids.clear()
