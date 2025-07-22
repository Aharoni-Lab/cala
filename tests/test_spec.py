from cala.models.spec import Pipe


class TestPipeline:

    def test_from_id(self):
        assert Pipe.from_specification("cala-basic")
