"""
Mixin classes that are to be used alongside specific models
to use composition for functionality and inheritance for semantics.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T")


class YamlDumper(yaml.SafeDumper):
    """Dumper that can represent extra types like Paths"""

    def represent_path(self, data: Path) -> yaml.ScalarNode:
        """Represent a path as a string"""
        return self.represent_scalar("tag:yaml.org,2002:str", str(data))


YamlDumper.add_representer(type(Path()), YamlDumper.represent_path)


class YAMLMixin:
    """
    Mixin class that provides :meth:`.from_yaml` and :meth:`.to_yaml`
    classmethods
    """

    @classmethod
    def from_yaml(cls: type[T], file_path: str | Path) -> T:
        """Instantiate this class by passing the contents of a yaml file as kwargs"""
        with open(file_path) as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    def to_yaml(self, path: Path | None = None, **kwargs: Any) -> str:
        """
        Dump the contents of this class to a yaml file, returning the
        contents of the dumped string
        """
        data_str = self.to_yamls(**kwargs)
        if path:
            with open(path, "w") as file:
                file.write(data_str)

        return data_str

    def to_yamls(self, **kwargs: Any) -> str:
        """
        Dump the contents of this class to a yaml string

        Args:
            **kwargs: passed to :meth:`.BaseModel.model_dump`
        """
        data = self._dump_data(**kwargs)
        return yaml.dump(data, Dumper=YamlDumper, sort_keys=False)

    def _dump_data(self, **kwargs: Any) -> dict:
        data = self.model_dump(**kwargs) if isinstance(self, BaseModel) else self.__dict__
        return data
