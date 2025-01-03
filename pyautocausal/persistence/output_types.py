from enum import Enum, auto
from typing import Set

class OutputType(Enum):
    CSV = ".csv"
    PARQUET = ".parquet"
    JSON = ".json"
    PICKLE = ".pkl"
    TEXT = ".txt"
    PNG = ".png"
    BYTES = ".bytes"
    
    @classmethod
    def get_supported_extensions(cls) -> Set[str]:
        return {member.value for member in cls} 