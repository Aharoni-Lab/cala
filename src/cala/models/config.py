import importlib
import re
import sys
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, TypeIs

from pydantic import AfterValidator, Field, TypeAdapter

CONFIG_ID_PATTERN = r"[\w\-\/#]+"
"""
Any alphanumeric string (\\w), as well as
- ``-``
- ``/``
- ``#``
(to allow hierarchical IDs as well as fragment IDs).

Specifically excludes ``.`` to avoid confusion between IDs, paths, and python module names

May be made less restrictive in the future, will not be made more restrictive.
"""


def _is_identifier(val: str) -> str:
    assert val.isidentifier(), "Must be a valid python identifier"
    return val


def _is_absolute_identifier(val: str) -> str:

    assert not val.startswith("."), "Cannot use relative identifiers"
    for part in val.split("."):
        assert part.isidentifier(), f"{part} is not a valid python identifier within {val}"
    return val


PythonIdentifier: type = Annotated[str, AfterValidator(_is_identifier)]
"""
A single valid python identifier.

See: https://docs.python.org/3.13/library/stdtypes.html#str.isidentifier
"""

AbsoluteIdentifier: type = Annotated[str, AfterValidator(_is_absolute_identifier)]
"""
- A valid python identifier, including globally accessible namespace like module.submodule.ClassName
OR 
- a name of a builtin function/type
"""

ConfigID: type = Annotated[str, Field(pattern=CONFIG_ID_PATTERN)]
"""
A string that refers to a config file by the ``id`` field in that config
"""

ConfigSource: type = Path | PathLike[str] | ConfigID
"""
Union of all types of config sources
"""


def valid_config_id(val: Any) -> TypeIs[ConfigID]:
    """
    Checks whether a string is a valid config id.
    """
    return bool(re.fullmatch(CONFIG_ID_PATTERN, val))


AbsoluteIdentifierAdapter = TypeAdapter(AbsoluteIdentifier)


def resolve_python_identifier(ref: AbsoluteIdentifier) -> Any:
    """
    Given some fully-qualified package.subpackage.Class identifier,
    return the referenced object, importing if needed.
    """

    ref = AbsoluteIdentifierAdapter.validate_python(ref)

    module_name, obj = ref.rsplit(".", 1)
    module = sys.modules.get(module_name, importlib.import_module(module_name))
    return getattr(module, obj)
