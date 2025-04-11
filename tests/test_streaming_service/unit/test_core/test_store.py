import pytest

from cala.streaming.core import (
    Component,
)


class TestComponent:
    """Test suite for the Component enumeration."""

    def test_values(self) -> None:
        """Test Component enum values."""
        assert Component.NEURON.value == "neuron"
        assert Component.BACKGROUND.value == "background"

    def test_membership(self) -> None:
        """Test Component enum membership."""
        assert Component("neuron") == Component.NEURON
        assert Component("background") == Component.BACKGROUND

    def test_invalid_value(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            Component("invalid")
