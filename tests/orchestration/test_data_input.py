import pytest
from dataclasses import dataclass
from pyautocausal.orchestration.data_input import DataInput

# Test class that correctly implements DataInput
@dataclass  # This must come first!
class ValidDataInput(DataInput):
    field1: str
    field2: int
    optional_field: str = "default"
    
    def to_dict(self):
        return {
            "field1": self.field1,
            "field2": self.field2,
            "optional_field": self.optional_field
        }
    
    @classmethod
    def get_required_fields(cls) -> set[str]:
        return {"field1", "field2"}

class InvalidDataInput(DataInput):
    def __init__(self, field1: str, field2: int):
        self.field1 = field1
        self.field2 = field2
    
    def to_dict(self):
        return {
            "field1": self.field1,
            "field2": self.field2
        }
    
    @classmethod
    def get_required_fields(cls) -> set[str]:
        return {"field1", "field2"}


def test_valid_data_input():
    """Test that a valid dataclass implementation works"""
    input_data = ValidDataInput(field1="test", field2=42)
    assert input_data.field1 == "test"
    assert input_data.field2 == 42
    assert input_data.optional_field == "default"

def test_non_dataclass_raises_error_when_checking_required_fields():
    """Test that non-dataclass implementation raises TypeError"""
    with pytest.raises(TypeError) as exc_info:
        invalid_data_input = InvalidDataInput(field1="test", field2=42)
        invalid_data_input.check_presence_of_required_fields()
    assert "must be decorated with @dataclass" in str(exc_info.value)

def test_to_dict_conversion():
    """Test that to_dict converts data correctly"""
    input_data = ValidDataInput(field1="test", field2=42, optional_field="custom")
    expected_dict = {
        "field1": "test",
        "field2": 42,
        "optional_field": "custom"
    }
    assert input_data.to_dict() == expected_dict 