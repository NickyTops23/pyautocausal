from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, Any, TypeVar, Type

T = TypeVar('T', bound='DataInput')

class DataInput(ABC):
    """Abstract base class for pipeline input data"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert input data to dictionary format"""
        pass
    
    @classmethod
    @abstractmethod
    def get_required_fields(cls) -> set[str]:
        """Get set of required field names"""
        pass
    
    @classmethod
    def check_presence_of_required_fields(cls) -> None:
        """Check if all required fields are present in input dictionary"""
        if not is_dataclass(cls):
            raise TypeError(f"Class {cls.__name__} must be decorated with @dataclass")
            
        required_fields = cls.get_required_fields()
        class_fields = {f.name for f in cls.__dataclass_fields__.values()}
        if not required_fields.issubset(class_fields):
            missing = required_fields - class_fields
            raise ValueError(f"Missing required fields: {missing}")
