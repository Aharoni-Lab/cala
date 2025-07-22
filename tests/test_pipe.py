from cala.config.pipe import Node, Pipeline


class TestPipeline:

    def test_from_init(self):
        step1 = Node(id="cala.testing.nodes.NodeA", params={"a": 1}, requires=[])
        assert Pipeline(
            cala_id="testing-from-init",
            buff={"buffer": 100},
            prep={"node_a": step1},
            init={"node_a": step1},
            iter={"node_a": step1},
        )

    def test_from_id(self): ...

    def test_from_yaml(self): ...
