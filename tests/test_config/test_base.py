from pathlib import Path

import pytest
import yaml

from cala.config.base import Config
from cala.config.pipe import Initializer, Iterator, Preprocessor


class TestBaseConfig:
    @pytest.fixture
    def sample_config_yaml(self, tmp_path):
        """Create a sample config file for testing"""
        config_path = tmp_path / "cala_config.yaml"
        config_data = {
            "video_directory": str(tmp_path / "videos"),
            "pipeline": {
                "preprocess": {
                    "downsample": {
                        "transformer": "downsample",
                        "params": {
                            "method": "mean",
                            "dimensions": ["width", "height"],
                            "strides": [2, 2],
                        },
                    },
                    "denoise": {
                        "transformer": "denoise",
                        "params": {
                            "method": "gaussian",
                            "kwargs": {"ksize": [3, 3], "sigmaX": 1.5},
                        },
                        "requires": ["downsample"],
                    },
                },
                "initialization": {
                    "footprints": {
                        "transformer": "footprints",
                        "params": {"threshold_factor": 0.2, "kernel_size": 3},
                        "n_frames": 3,
                    }
                },
                "iteration": {"traces": {"transformer": "traces", "params": {"window_size": 10}}},
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return str(config_path)

    def test_config_loads_pipeline(self, sample_config_yaml):
        """Test that pipeline config loads and converts transformers to enums"""
        config = Config(config_file=sample_config_yaml)

        # Check pipeline config was loaded
        assert config.pipeline is not None

        # Test preprocessing transformers
        preprocess = config.pipeline.preprocess
        assert preprocess["downsample"].transformer == Preprocessor.DOWNSAMPLE
        assert preprocess["denoise"].transformer == Preprocessor.DENOISE

        # Test initialization transformers
        init = config.pipeline.initialization
        assert init["footprints"].transformer == Initializer.FOOTPRINTS
        assert init["footprints"].n_frames == 3

        # Test iteration transformers
        iter_config = config.pipeline["iteration"]
        assert iter_config["traces"].transformer == Iterator.TRACES

    def test_config_validates_transformer_names(self, tmp_path):
        """Test that invalid transformer names are caught"""
        config_path = tmp_path / "invalid_config.yaml"
        invalid_config = {
            "pipeline": {
                "preprocess": {"invalid": {"transformer": "not_a_real_transformer", "params": {}}}
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError):
            Config(config_file=config_path)

    def test_config_validates_dependencies(self, sample_config_yaml):
        """Test that dependencies are properly loaded"""
        config = Config(config_file=sample_config_yaml)

        # Check that denoise requires downsample
        preprocess = config.pipeline.preprocess
        assert "requires" in preprocess["denoise"]
        assert preprocess["denoise"]["requires"] == ["downsample"]

    def test_config_validates_params(self, tmp_path):
        """Test that params are properly validated"""
        config_path = tmp_path / "missing_params.yaml"
        invalid_config = {
            "pipeline": {
                "preprocess": {
                    "downsample": {
                        "transformer": "downsample"
                        # Missing required params
                    }
                }
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError):
            Config(config_file=config_path)

    def test_config_file_paths(self, sample_config_yaml):
        """Test that file paths are properly resolved"""
        config = Config(config_file=sample_config_yaml)

        # Check that video directory is a Path
        assert isinstance(config.video_directory, Path)
        assert config.video_directory.is_absolute()
