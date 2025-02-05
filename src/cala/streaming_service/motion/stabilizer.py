from river.base import Transformer


class MotionStabilizer(Transformer):
    """Handles motion correction"""

    def correct_frame(self, frame, template):
        pass


class RigidMotionStabilizer(MotionStabilizer):
    """Implements rigid motion correction"""

    pass


class PiecewiseRigidMotionStabilizer(MotionStabilizer):
    """Implements piecewise rigid motion correction"""

    pass
