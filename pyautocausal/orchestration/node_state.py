from enum import Enum, auto
from typing import Set

class NodeState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    
    @classmethod
    def terminal_states(cls) -> Set['NodeState']:
        return {cls.COMPLETED, cls.FAILED, cls.SKIPPED}
    
    def is_terminal(self) -> bool:
        return self in self.terminal_states()

