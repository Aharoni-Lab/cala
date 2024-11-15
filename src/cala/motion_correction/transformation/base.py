from abc import ABC, abstractmethod
import numpy as np
from typing import Dict


class Transformation(ABC):
    @abstractmethod
    def compute_shift(
        self, base_frame: np.ndarray, current_frame: np.ndarray
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    def apply_transformation(
        self, frame: np.ndarray, params: Dict[str, float]
    ) -> np.ndarray:
        pass
