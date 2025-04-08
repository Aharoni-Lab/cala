# Cala

[![PyPI - Version](https://img.shields.io/pypi/v/cala)](https://pypi.org/project/cala/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cala)
![PyPI - Status](https://img.shields.io/pypi/status/cala)
[![codecov](https://codecov.io/gh/Aharoni-Lab/cala/graph/badge.svg?token=Apn4YtSvbU)](https://codecov.io/gh/Aharoni-Lab/cala)

## Features

Cala is a neural endoscope image processing tool designed for neuroscience research, with a focus on long-term massive recordings. It features a no-code approach through configuration files, making it accessible to researchers of all programming backgrounds.

## Requirements

- Python 3.11, 3.12, or 3.13
- Dependencies are handled through [pdm](https://pdm-project.org/en/latest/)

## Installation

```shell
pip install cala==0.1.0
```

## Quick Start

1. Prepare your video files
2. Create a configuration file (YAML format)
3. Run the pipeline:

```bash
python main.py --visual --config cala_config.yaml
```

## Architecture

Cala uses a graph-&-state based architecture with three key components:

1. **Configuration System**
   - Supports YAML and env-based configuration
   - No-code pipeline setup
   - Flexible node configuration

2. **Processing Nodes**
   - Modular transformation units
   - Managed automatically by the runner
   - Connected to storage through parameter types

3. **Storage System**
   - Automatically created and updated by the distributor
   - Leverages [Zarr](https://zarr.dev/) for large-scale data storage

Schematics of the architecture can be found [here](https://lucid.app/documents/embedded/808097f9-bf66-4ea8-9df0-e957e6bd0931).

## Documentation

Detailed documentation is available in three main sections:

1. **User Guide**: Step-by-step guide for using Cala
   - Configuration file setup
   - Pipeline structure
   - Processing nodes
   - Advanced features

2. **Developer Guide**: Information for extending Cala
   - Adding new nodes
   - Working with stores
   - Best practices

3. **API Reference**: Available on [Read the Docs](https://cala.readthedocs.io/en/latest/)

## Roadmap

*EOM 04/2025:* UI first iteration complete

## Contributing

We welcome contributions! Please fork this repository and submit a pull request if you would like to contribute to the
project. You can also open issues for bug reports, feature requests, or discussions.

## Test Coverage Status

https://app.codecov.io/gh/Aharoni-Lab/cala

## License

## Contact

For questions or support, please reach out to Raymond Chang
at [raymond@physics.ucla.edu](mailto:raymond@physics.ucla.edu).
