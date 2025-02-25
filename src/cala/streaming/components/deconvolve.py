from abc import abstractmethod

from river.base import Transformer


class BaseDeconvolver(Transformer):
    """Base class for deconvolution algorithms"""

    @abstractmethod
    def deconvolve(self, traces):
        pass


class OASIS(BaseDeconvolver):
    """Handles spike deconvolution"""

    def deconvolve(self, traces):
        pass
