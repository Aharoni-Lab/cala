from dataclasses import dataclass

import pytest

from cala.streaming.core.parameters import Parameters


@dataclass
class TestParameters(Parameters):
    """Concrete implementation of Parameters for testing"""

    name: str
    value: int = 0
    optional: str = None

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not isinstance(self.value, int):
            raise ValueError("value must be an integer")
        if self.optional is not None and not isinstance(self.optional, str):
            raise ValueError("optional must be a string if provided")


def test_parameters_initialization():
    """Test basic parameter initialization"""
    params = TestParameters(name="test", value=42)
    assert params.name == "test"
    assert params.value == 42
    assert params.optional is None


def test_parameters_validation():
    """Test parameter validation"""
    with pytest.raises(ValueError):
        TestParameters(name=123, value=42)  # Invalid name type

    with pytest.raises(ValueError):
        TestParameters(name="test", value="42")  # Invalid value type

    with pytest.raises(ValueError):
        TestParameters(name="test", value=42, optional=123)  # Invalid optional type


def test_parameters_to_dict():
    """Test conversion to dictionary"""
    params = TestParameters(name="test", value=42)
    params_dict = params.to_dict()
    assert params_dict["name"] == "test"
    assert params_dict["value"] == 42
    assert "optional" not in params_dict  # None values should be excluded


def test_parameters_from_dict():
    """Test creation from dictionary"""
    data = {"name": "test", "value": 42, "optional": "extra"}
    params = TestParameters.from_dict(data)
    assert params.name == "test"
    assert params.value == 42
    assert params.optional == "extra"


def test_parameters_copy():
    """Test deep copying of parameters"""
    original = TestParameters(name="test", value=42)
    copied = original.copy()

    assert original is not copied  # Different objects
    assert original.name == copied.name
    assert original.value == copied.value


def test_parameters_update():
    """Test parameter updates"""
    original = TestParameters(name="test", value=42)
    updated = original.update(value=99)

    assert original.value == 42  # Original unchanged
    assert updated.value == 99
    assert original.name == updated.name  # Unchanged fields remain the same


def test_parameters_str():
    """Test string representation"""
    params = TestParameters(name="test", value=42)
    str_repr = str(params)

    assert "Parameters:" in str_repr
    assert "name: test" in str_repr
    assert "value: 42" in str_repr


def test_parameters_save_load(tmp_path):
    """Test saving and loading parameters"""
    original = TestParameters(name="test", value=42, optional="extra")
    filename = tmp_path / "params.json"

    # Save parameters
    original.save(str(filename))

    # Load parameters
    loaded = TestParameters.load(str(filename))

    assert loaded.name == original.name
    assert loaded.value == original.value
    assert loaded.optional == original.optional
