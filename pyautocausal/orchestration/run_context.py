from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class RunContext:
    """
    Base class for holding run-specific configuration and metadata.
    Allows for arbitrary key-value pairs to be stored and accessed.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing metadata values as attributes"""
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting metadata values as attributes"""
        if name == 'metadata':
            super().__setattr__(name, value)
        else:
            self.metadata[name] = value
            
    def get(self, key: str, default: Any = None) -> Any:
        """Safe way to get metadata values with a default"""
        return self.metadata.get(key, default)
