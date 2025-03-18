import numpy as np
import xarray as xr

from cala.streaming.core.stores import (
    Observable,
    Footprints,
    Traces,
    FluorescentObject,
    Neuron,
    Background,
    NeuronFootprints,
    NeuronTraces,
    BackgroundFootprints,
    BackgroundTraces,
    generate_cross_product_classes,
)


class TestBaseTypes:
    # Test base Observable class and its subclasses
    def test_observable_inheritance(self):
        # Create test data
        data = np.random.rand(10, 10)
        coords = {"x": range(10), "y": range(10)}

        # Test Observable base class
        obs = Observable(data, coords=coords)
        assert isinstance(obs, xr.DataArray)
        assert isinstance(obs, Observable)

        # Test Footprints
        footprints = Footprints(data, coords=coords)
        assert isinstance(footprints, Observable)
        assert isinstance(footprints, Footprints)

        # Test Traces
        traces = Traces(data, coords=coords)
        assert isinstance(traces, Observable)
        assert isinstance(traces, Traces)

    # Test FluorescentObject hierarchy
    def test_fluorescent_object_inheritance(self):
        assert issubclass(Neuron, FluorescentObject)
        assert issubclass(Background, FluorescentObject)

        neuron = Neuron()
        background = Background()

        assert isinstance(neuron, FluorescentObject)
        assert isinstance(background, FluorescentObject)


class TestCompositeTypes:
    # Test cross-product class generation
    def test_generate_cross_product_classes(self):
        classes = generate_cross_product_classes()

        # Check all expected classes are generated
        expected_classes = {
            "NeuronFootprints",
            "NeuronTraces",
            "BackgroundFootprints",
            "BackgroundTraces",
        }
        assert set(classes.keys()).issuperset(expected_classes)

        # Check each generated class has correct inheritance
        assert issubclass(classes["NeuronFootprints"], (Neuron, Footprints))
        assert issubclass(classes["NeuronTraces"], (Neuron, Traces))
        assert issubclass(classes["BackgroundFootprints"], (Background, Footprints))
        assert issubclass(classes["BackgroundTraces"], (Background, Traces))

    def test_cross_product_class_instances(self):
        # Test data
        data = np.random.rand(10, 10)
        coords = {"x": range(10), "y": range(10)}

        # Test NeuronFootprints
        nf = NeuronFootprints(data, coords=coords)
        assert isinstance(nf, NeuronFootprints)
        assert isinstance(nf, Neuron)
        assert isinstance(nf, Footprints)
        assert isinstance(nf, Observable)
        assert isinstance(nf, xr.DataArray)

        # Test NeuronTraces
        nt = NeuronTraces(data, coords=coords)
        assert isinstance(nt, NeuronTraces)
        assert isinstance(nt, Neuron)
        assert isinstance(nt, Traces)
        assert isinstance(nt, Observable)
        assert isinstance(nt, xr.DataArray)

        # Test BackgroundFootprints
        bf = BackgroundFootprints(data, coords=coords)
        assert isinstance(bf, BackgroundFootprints)
        assert isinstance(bf, Background)
        assert isinstance(bf, Footprints)
        assert isinstance(bf, Observable)
        assert isinstance(bf, xr.DataArray)

        # Test BackgroundTraces
        bt = BackgroundTraces(data, coords=coords)
        assert isinstance(bt, BackgroundTraces)
        assert isinstance(bt, Background)
        assert isinstance(bt, Traces)
        assert isinstance(bt, Observable)
        assert isinstance(bt, xr.DataArray)

    def test_cross_product_class_data_integrity(self):
        # Create test data with specific values
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        coords = {"x": [0, 1], "y": [0, 1]}

        # Test data integrity for each class
        for cls in [
            NeuronFootprints,
            NeuronTraces,
            BackgroundFootprints,
            BackgroundTraces,
        ]:
            instance = cls(data, coords=coords)

            # Check data values
            np.testing.assert_array_equal(instance.values, data)

            # Check coordinates
            assert instance.coords["x"].values.tolist() == coords["x"]
            assert instance.coords["y"].values.tolist() == coords["y"]
