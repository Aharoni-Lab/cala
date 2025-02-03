from abc import ABC, abstractmethod


class BaseDeconvolver(ABC):
    """Base class for deconvolution algorithms"""

    @abstractmethod
    def deconvolve(self, traces):
        pass


class OASIS(BaseDeconvolver):
    """Handles spike deconvolution"""

    def deconvolve(self, traces):
        pass
