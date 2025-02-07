class CNMFInitializer:
    """Base class for initialization methods"""

    def initialize(self, data):
        pass


class BareInitializer(CNMFInitializer):
    """No initialization"""

    pass


class SeededInitializer(CNMFInitializer):
    """Initialization from seeds"""

    pass
