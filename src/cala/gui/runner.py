import threading
from dataclasses import dataclass
from enum import Enum, auto
from logging import Logger
from queue import Queue, Empty

from noob import SynchronousRunner
from noob.types import ReturnNodeType

from cala.logging import init_logger


class RunnerAlreadyRunningError(Exception):
    """Raised when attempting to start or configure the runner when it's already running."""

    def __str__(self):
        return "Runner is already running"


class RunnerNotRunningError(Exception):
    """Raised when attempting to shutdown the runner when it's not running."""

    def __str__(self):
        return "Runner is not running"


class RunState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


@dataclass(kw_only=True)
class BackgroundRunner(SynchronousRunner):
    _q: Queue = Queue()
    _state: RunState = RunState.STOPPED
    _thread: threading.Thread | None = None
    logger: Logger = init_logger(__name__)

    @property
    def state(self) -> RunState:
        return self._state

    def join(self):
        if self._thread is not None:
            self._thread.join()
        else:
            raise RunnerNotRunningError

    def _read_queue(self) -> None:
        try:
            item = self._q.get(False)
        except Empty:
            item = None

        if item:
            self._state = RunState.STOPPED

    def run(self, n: int | None = None) -> None | list[ReturnNodeType]:
        thread = threading.Thread(target=self._main_loop, args=(n,))
        thread.start()
        self._thread = thread

    def _main_loop(self, n: int | None = None) -> None | list[ReturnNodeType]:
        self.init()

        self._state = RunState.RUNNING

        outputs = []
        current_iter = 0
        try:
            while self._state == RunState.RUNNING and (n is None or current_iter < n):
                self._read_queue()
                out = self.process()
                if out is not None:
                    outputs.append(out)
                current_iter += 1
        except KeyboardInterrupt:
            # fine, just return
            pass

        finally:
            self.deinit()

        return outputs if outputs else None

    def shutdown(self, timeout: float | None = None) -> None:
        """
        Shuts down the runner.

        :raises RunnerNotRunningError: if the runner has not been started yet
        """
        if self._state == RunState.STOPPED or self._thread is None:
            raise RunnerNotRunningError

        self._q.put(RunState.STOPPED)
        self._thread.join(timeout)
        self._thread = None

        self._state = RunState.STOPPED

        self._logger.info("Runner has been shut down")

    def pause(self):
        """
        Pause job processing in the runner.

        This will prevent the runner from waking up to do job processing until :meth:`resume`
        is called.

        """

    def resume(self):
        """Resume job processing in the runner."""
