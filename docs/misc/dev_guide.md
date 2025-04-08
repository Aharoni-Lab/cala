# Developer Guide

## Overview

Cala closely follows the Single Responsibility Principle, making it easy to extend with new functionality. This guide will walk you through the process of adding new processing nodes to the system.

## Adding New Nodes

### 1. Create Your Node Class

1. Choose the appropriate stage folder under `cala/streaming/nodes/`:
   - `preprocess/`
   - `initialization/`
   - `iteration/`

2. Create a new file for your node class:

```python
from river import BaseTransformer
from cala.streaming.parameters import Parameters
from dataclasses import dataclass

@dataclass
class NewNodeParameters(Parameters):
    # Define your node's parameters here
    param1: float
    param2: str
    # ...

class NewNode(BaseTransformer):
    def __init__(self, parameters: NewNodeParameters):
        self.parameters = parameters

    # Implement required abstract methods
    def learn_one(self, X: NewType1, y: NewType2):
        # Implementation
        pass

    def transform_one(self, X: NewType1) -> NewType3:
        # Implementation
        pass
```

### 2. Register Your Node

Add your node to `enum.py` to make it available in user configurations:

```python
from enum import Enum

class TransformerEnum(Enum):
    # ... existing nodes ...
    NEW_NODE = "NewNode"
```

### 3. Working with Stores

⚠️ **IMPORTANT**: State Management in Cala

All state that needs to be tracked by the `runner` must be managed through stores. Each store must:
1. Inherit from `ObservableStore`
2. Define an update method
3. Have a corresponding Annotated type

Example of a custom store:

```python
from typing import Annotated, Dict
from cala.streaming.core import ObservableStore

class DictionaryStore(ObservableStore):
    """Store that uses dictionaries instead of xarray."""

    def __init__(self):
        self.warehouse: Dict = {}

    def update(self, data: Dict) -> None:
        """Update the dictionary warehouse.

        Args:
            data: New dictionary data to incorporate
        """
        if not data:
            return None

        # Merge new data into warehouse
        self.warehouse.update(data)
        return None

# Create the Annotated type for dictionary data
NewType = Annotated[Dict, DictionaryStore]
```

As long as the nodes are populated with the correct annotated types, the `runner` and `distributor` will automatically build, assign and update the stores.

### 4. Best Practices

1. **Parameter Management**
   - Create a dedicated parameter class inheriting from `Parameters`
   - Use type hints for all parameters
   - Document parameter purposes and valid ranges

2. **Implementation Guidelines**
   - Follow the `river` streaming interface
   - Typehints are mandatory. Nodes without proper typehints will not plug into the runner properly.
   - Implement all required abstract methods

3. **Documentation**
   - Document expected inputs and outputs
   - Provide usage examples

## Testing Your Node

1. Create unit tests in the `tests` directory
2. Verify integration with the pipeline
3. Test with sample configurations
4. Validate outputs

## Common Issues

1. **Parameter Inheritance**
   - Ensure your parameter class inherits from `Parameters`
   - All parameters must be properly typed

2. **Node Registration**
   - Verify node is correctly listed in `enum.py`
   - Check for naming conflicts

3. **Store Integration**
   - Ensure your store is correctly defined in `cala/streaming/stores` and inherits from `Store`

## Need Help?

- Check existing node implementations for examples
- Review the `river` documentation for transformer guidelines

That's it! Your node is now ready to be used in Cala configurations.
