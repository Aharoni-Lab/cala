from cala.streaming.core import Observable


# pixels x components
class PixelStats(Observable):
    __slots__ = ()


# components x components
class ComponentStats(Observable):
    __slots__ = ()


# this doesn't technically need a store. no association with components
class Residual(Observable):
    __slots__ = ()
