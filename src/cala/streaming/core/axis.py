class Axis:
    """Mixin providing common axis-related attributes."""

    frames_dim: str = "frame"
    """Name of the dimension representing time points."""

    height_dim: str = "height"
    """Name of the dimension representing height points."""

    width_dim: str = "width"
    """Name of the dimension representing width points."""

    component_dim: str = "component"
    """Name of the dimension representing individual components."""

    id_coord: str = "id_"
    """Name of the coordinate used to identify individual components with unique IDs."""

    type_coord: str = "type_"
    """Name of the coordinate used to specify component types (e.g., neuron, background)."""

    time_coord: str = "timestamp"

    confidence_coord: str = "confidence"

    frame_coord: str = "frame"

    width_coord: str = "width"

    height_coord: str = "height"

    @property
    def spatial_dims(self) -> tuple[str, str]:
        """Names of the dimensions representing 2-d spatial coordinates Default: (height, width)."""
        return self.height_dim, self.width_dim

    @property
    def spatial_coords(self) -> tuple[str, str]:
        """Names of the dimensions representing 2-d spatial coordinates Default: (height, width)."""
        return self.height_coord, self.width_coord
