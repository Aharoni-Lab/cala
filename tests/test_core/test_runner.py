from cala.runner import Runner
from cala.models.spec import Pipe


def test_prep():
    tube = Pipe.from_specification("cala-prep")
    runner = Runner(pipe=tube)
    while True:
        try:
            runner.preprocess()
        except RuntimeError as e:
            assert isinstance(e, RuntimeError)
            assert e.args == ("Generator node stopped its iteration",)
            break
