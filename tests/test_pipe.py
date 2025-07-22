from cala.models.spec import NodeSpec, PipeSpec


class TestPipeline:

    def test_from_init(self):
        step1 = NodeSpec(id="cala.testing.nodes.NodeA", params={"a": 1}, requires=[])
        assert PipeSpec(
            cala_id="testing-from-init",
            buff={"buffer": 100},
            prep={"prep-node-a": step1},
            init={"init-node-a": step1},
            iter={"iter-node-a": step1},
        )

    def test_from_id(self):
        assert PipeSpec.from_id("cala-basic")
