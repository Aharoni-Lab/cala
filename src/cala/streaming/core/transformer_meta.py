import inspect
from abc import ABCMeta
from typing import Callable


class TransformerMeta(ABCMeta):
    """Metaclass for streaming transformers that extracts method signatures."""

    def __new__(cls, name, bases, dct):
        new_cls = super().__new__(cls, name, bases, dct)

        # Extract signatures from methods
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                setattr(attr_value, "__signature__", cls._extract_signature(attr_value))

        return new_cls

    @staticmethod
    def _extract_signature(func: Callable) -> dict[str, type]:
        """
        Extracts parameter and return type hints from a callable.

        Args:
            func: The callable to analyze

        Returns:
            Dictionary mapping parameter names to their type hints
        """

        # Get function signature
        signature = inspect.signature(func)

        # Extract parameter categories (excluding 'self' if present)
        param_types = {
            name: param.annotation
            for name, param in signature.parameters.items()
            if name != "self"
        }

        # Add return type if specified
        if signature.return_annotation:
            param_types["return"] = signature.return_annotation

        return param_types
