from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING
from unittest.mock import patch

import pytest
import tomli_w
import yaml

from cala import Config
from cala.config import _dirs

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


def _flatten(d, parent_key="", separator="__") -> dict:
    """https://stackoverflow.com/a/6027615/13113166"""
    items = []
    for key, value in d.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


@pytest.fixture(scope="module", autouse=True)
def dodge_existing_global_config(tmp_path_factory):
    """
    Suspend any existing global config file during config tests
    """
    tmp_path = tmp_path_factory.mktemp("config_backup")
    default_global_config_path = Path(_dirs.user_config_dir) / "cala_config.yaml"
    backup_global_config_path = tmp_path / "cala_config.yaml.bak"

    configured_global_config_path = Config().config_file
    backup_configured_global_path = tmp_path / "cala_config_custom.yaml.bak"

    if default_global_config_path.exists():
        default_global_config_path.rename(backup_global_config_path)
    if configured_global_config_path.exists():
        default_global_config_path.rename(backup_configured_global_path)

    yield

    if backup_global_config_path.exists():
        default_global_config_path.unlink(missing_ok=True)
        backup_global_config_path.rename(default_global_config_path)
    if backup_configured_global_path.exists():
        configured_global_config_path.unlink(missing_ok=True)
        backup_configured_global_path.rename(configured_global_config_path)


@pytest.fixture(autouse=True)
def tmp_cwd(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def set_env(monkeypatch) -> Callable[[dict[str, Any]], None]:
    """
    Function fixture to set environment variables using a nested dict
    matching a GlobalConfig.model_dump()
    """

    def _set_env(config: dict[str, Any]) -> None:
        for key, value in _flatten(config).items():
            key = "CALA_" + key.upper()
            monkeypatch.setenv(key, str(value))

    return _set_env


@pytest.fixture()
def set_dotenv(tmp_cwd) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a .env file
    """
    dotenv_path = tmp_cwd / ".env"

    def _set_dotenv(config: dict[str, Any]) -> Path:
        with open(dotenv_path, "w") as dfile:
            for key, value in _flatten(config).items():
                key = "CALA_" + key.upper()
                dfile.write(f"{key}={value}\n")
        return dotenv_path

    return _set_dotenv


@pytest.fixture()
def set_pyproject(tmp_cwd) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a pyproject.toml file
    """
    toml_path = tmp_cwd / "pyproject.toml"

    def _set_pyproject(config: dict[str, Any]) -> Path:
        config = {"tool": {"cala": {"config": config}}}

        with open(toml_path, "wb") as tfile:
            tomli_w.dump(config, tfile)

        return toml_path

    return _set_pyproject


@pytest.fixture()
def set_local_yaml(tmp_cwd) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a local linkml_config.yaml file
    """
    yaml_path = tmp_cwd / "cala_config.yaml"

    def _set_local_yaml(config: dict[str, Any]) -> Path:
        with open(yaml_path, "w") as yfile:
            yaml.safe_dump(config, yfile)
        return yaml_path

    return _set_local_yaml


@pytest.fixture()
def set_global_yaml() -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to reversibly set config variables in a global linkml_config.yaml file
    """
    global_config_path = Path(_dirs.user_config_dir) / "cala_config.yaml"
    backup_path = Path(_dirs.user_config_dir) / "cala_config.yaml.bak"
    restore_backup = global_config_path.exists()

    try:
        if restore_backup:
            global_config_path.rename(backup_path)

        def _set_global_yaml(config: dict[str, Any]) -> Path:
            with open(global_config_path, "w") as gfile:
                yaml.safe_dump(config, gfile)
            return global_config_path

        yield _set_global_yaml

    finally:
        global_config_path.unlink(missing_ok=True)
        if restore_backup:
            backup_path.rename(global_config_path)


@pytest.mark.parametrize(
    "setter",
    ["set_env", "set_dotenv", "set_pyproject", "set_local_yaml", "set_global_yaml"] * 2,
)
def test_config_sources(setter, request):
    """
    Base test that each of the settings sources in isolation can set values.

    Run twice so we confirm old settings don't hang out between tests
    and that we're properly tearing things down
    """
    fixture_fn = request.getfixturevalue(setter)
    expected = "MEATBALL SALAD"

    assert Config().data_name != expected
    fixture_fn({"data_name": expected})
    assert Config().data_name == expected


@pytest.fixture
def mock_path_exists():
    with patch.object(Path, "exists", return_value=True):
        yield


@pytest.mark.parametrize("setter_name", ["set_env", "set_dotenv"])
def test_split_video_files(setter_name, request: "FixtureRequest", mock_path_exists):
    """
    When video files are a comma-separated list string, as in an .env file,
    split them into an actual list
    """
    setter = request.getfixturevalue(setter_name)
    expected = [
        Path("video1.mp4"),
        Path("video2.mp4"),
        Path("video3.mp4"),
        Path("video4.mp4"),
    ]
    assert Config().video_files != expected
    setter({"video_files": "video1.mp4,video2.mp4, video3.mp4 , video4.mp4"})
    assert Config().video_files == expected


def test_config_file_is_absolute(set_local_yaml):
    """
    When a cala_config.yaml file is not found relative to cwd,
    make it absolute relative to the global config directory.
    Otherwise, just resolve it.
    """
    default_config_file = Config().config_file
    assert default_config_file.is_absolute()
    assert default_config_file == Path(_dirs.user_config_dir) / "cala_config.yaml"

    # make one in cwd, and we should find that
    set_local_yaml({"data_name": "something"})
    cwd_config_file = Config().config_file
    assert cwd_config_file.is_absolute()
    assert cwd_config_file == Path("cala_config.yaml").resolve()
    assert cwd_config_file != default_config_file


def test_config_sources_overrides(
    set_env, set_dotenv, set_pyproject, set_local_yaml, set_global_yaml
):
    """
    The various config sources should override one another in order
    """
    set_global_yaml({"data_name": "0"})
    assert Config().data_name == "0"
    set_pyproject({"data_name": "1"})
    assert Config().data_name == "1"
    set_local_yaml({"data_name": "2"})
    assert Config().data_name == "2"
    set_dotenv({"data_name": "3"})
    assert Config().data_name == "3"
    set_env({"data_name": "4"})
    assert Config().data_name == "4"
    assert Config(**{"data_name": "5"}).data_name == "5"

    # order shouldn't matter - highest priority should always be highest
    set_dotenv({"data_name": "6"})
    assert Config().data_name == "4"
    set_local_yaml({"data_name": "7"})
    assert Config().data_name == "4"
    set_pyproject({"data_name": "8"})
    assert Config().data_name == "4"
    set_local_yaml({"data_name": "9"})
    assert Config().data_name == "4"
    set_global_yaml({"data_name": "10"})
    assert Config().data_name == "4"
