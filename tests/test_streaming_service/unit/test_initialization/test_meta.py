import inspect
from typing import List, Optional, Dict, Any

from cala.streaming.initialization.meta import TransformerMeta


# Test class using TransformerMeta
class TestTransformer(metaclass=TransformerMeta):
    def method_with_types(self, x: int, y: float) -> str:
        return str(x + y)

    def method_with_optional(self, x: Optional[int] = None) -> bool:
        return x is not None

    def method_with_complex_types(
        self, lst: List[int], dct: Dict[str, Any]
    ) -> List[str]:
        return [str(x) for x in lst]

    def method_without_types(self, x, y):
        return x + y

    def _private_method(self, x: int) -> int:
        return x


def test_signature_extraction_basic():
    """Test basic type extraction from methods."""
    signature = TestTransformer.method_with_types.__signature__

    assert signature["x"] == int
    assert signature["y"] == float
    assert signature["return"] == str


def test_signature_extraction_optional():
    """Test extraction of Optional types."""
    signature = TestTransformer.method_with_optional.__signature__

    assert signature["x"] == Optional[int]
    assert signature["return"] == bool


def test_signature_extraction_complex():
    """Test extraction of complex type annotations."""
    signature = TestTransformer.method_with_complex_types.__signature__

    assert signature["lst"] == List[int]
    assert signature["dct"] == Dict[str, Any]
    assert signature["return"] == List[str]


def test_signature_extraction_no_types():
    """Test handling of methods without type annotations."""
    signature = TestTransformer.method_without_types.__signature__

    # inspect.signature will use <class 'inspect._empty'> for unannotated params
    assert "x" in signature
    assert signature["x"] == inspect._empty
    assert "y" in signature
    assert signature["y"] == inspect._empty
    assert "return" in signature
    assert signature["return"] == inspect._empty


def test_private_method_signature():
    """Test that private methods (starting with _) are not processed."""
    assert hasattr(TestTransformer._private_method, "__signature__")
    assert TestTransformer._private_method.__signature__["x"] == int
    assert TestTransformer._private_method.__signature__["return"] == int


def test_inheritance():
    """Test that signature extraction works with inheritance."""

    class ChildTransformer(TestTransformer):
        def new_method(self, z: int) -> float:
            return float(z)

        def method_with_types(
            self, x: int, y: float
        ) -> int:  # Override with different return
            return int(x + y)

    # Test new method signature
    assert ChildTransformer.new_method.__signature__["z"] == int
    assert ChildTransformer.new_method.__signature__["return"] == float

    # Test overridden method signature
    assert ChildTransformer.method_with_types.__signature__["x"] == int
    assert ChildTransformer.method_with_types.__signature__["y"] == float
    assert ChildTransformer.method_with_types.__signature__["return"] == int


def test_multiple_inheritance():
    """Test signature extraction with multiple inheritance."""

    class MixinClass:
        def mixin_method(self, m: str) -> List[str]:
            return [m]

    class MultiInheritTransformer(TestTransformer, MixinClass):
        pass

    assert not hasattr(MultiInheritTransformer.mixin_method, "__signature__")
    assert hasattr(MultiInheritTransformer.method_with_types, "__signature__")


def test_method_call_preservation():
    """Test that methods remain callable after signature extraction."""
    transformer = TestTransformer()

    assert transformer.method_with_types(5, 3.14) == "8.14"
    assert transformer.method_with_optional(42) is True
    assert transformer.method_with_optional() is False
    assert transformer.method_with_complex_types([1, 2, 3], {"key": "value"}) == [
        "1",
        "2",
        "3",
    ]


def test_dynamic_method_addition():
    """Test signature extraction for dynamically added methods."""

    class DynamicTransformer(metaclass=TransformerMeta):
        pass

    def new_method(self, x: int) -> str:
        return str(x)

    DynamicTransformer.dynamic_method = new_method

    # The metaclass only processes methods during class creation
    # so dynamically added methods won't have signatures extracted
    assert not hasattr(DynamicTransformer.dynamic_method, "__signature__")
