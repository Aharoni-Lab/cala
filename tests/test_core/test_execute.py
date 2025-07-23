from cala.core.execute import Executor
from cala.models import Pipe


def test_prep():
    pipe = Pipe.from_specification("cala-prep")
    exe = Executor(pipe)

    frame = exe.preprocess()
