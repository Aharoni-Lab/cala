class Axis:
    """Mixin providing common axis-related attributes."""

    frame_dim: str = "frame"
    height_dim: str = "height"
    width_dim: str = "width"
    component_dim: str = "component"

    id_coord: str = "id_"
    timestamp_coord: str = "timestamp"
    detect_coord: str = "detected_on"
    frame_coord: str = "frame_idx"
    width_coord: str = "width"
    height_coord: str = "height"

    @property
    def spatial_dims(self) -> tuple[str, str]:
        """Names of the dimensions representing 2-d spatial coordinates Default: (height, width)."""
        return self.height_dim, self.width_dim

    @property
    def component_rename(self) -> dict[str, str]:
        return {
            axis: self.duplicate(axis)
            for axis in (self.component_dim, self.id_coord, self.detect_coord)
        }

    @staticmethod
    def duplicate(axis: str) -> str:
        return f"{axis}'"


AXIS = Axis()
