from cala.streaming.types import Observable


class OnlineDictLearning(Observable):
    __slots__ = ()


# pixels x components
class PixelStats(OnlineDictLearning):
    __slots__ = ()


# components x components
class ComponentStats(OnlineDictLearning):
    __slots__ = ()


# this doesn't technically need a store. no association with components
class Residual(OnlineDictLearning):
    __slots__ = ()
