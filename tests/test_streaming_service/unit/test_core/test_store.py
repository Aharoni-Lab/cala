import numpy as np
import pytest

from cala.streaming.core import (
    Component,
    ObservableStore,
)


class TestObservable:
    """Test suite for the Observable base class."""

    def test_inheritance(self) -> None:
        """Test that Observable properly inherits from DataArray."""
        data = np.random.rand(5, 5)
        observable = ObservableStore(data)
        assert isinstance(observable, ObservableStore)


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
