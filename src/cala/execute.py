from noob import SynchronousRunner
from pydantic import BaseModel

from cala.assets import Frame
from cala.logging import init_logger
from cala.models.spec import Pipe

logger = init_logger(__name__)


class Executor(BaseModel):
    """Manages the execution of streaming image analysis pipeline.

    This class orchestrates the steps of the imaging analysis pipeline
    according to a provided configuration.
    """

    pipe: Pipe
    """Configuration defining the pipeline structure and parameters."""

    def preprocess(self) -> Frame:
        """
        Execute preprocessing steps on a single frame.

        Returns:
            Preprocessed frame.
        """
        runner = SynchronousRunner(self.pipe.prep)
        frame = runner.process()

        return frame

    def iterate(self) -> None:
        """Execute iterate steps on a single frame."""
        ...
