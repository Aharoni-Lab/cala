# Introduction

Building a streaming CNMF package - starting with the module / class designs.
This pacakge should be able to accomplish the following:

1. Function like an ML pipeline (hopefully inherit from Sklearn classes)
2. Support streaming operations (Sklearn class instances are light, so that's good. Not sure how it marries into the
   streaming operations, since it's primarily designed for batch one-shot fit / transform. Operations like partial_fit
   might have to be custom written.)
3. Maybe instead of frame by frame, a few frames at a time?
4. Output / save visualizations and matrix data real time at all stages of processing
5. Store parameters and hyperparameters, and exposed so that the user can view/modify them while processing.
6. Parameters get updated real time as the new data streams in.
7. The learned / updated parameters should be able to retroactively propagate the earlier data to refine the results.
   This may have to be done post-processing.
8. Plug smoothly into the batch processing side of the pacakge.

## Streaming

```python
from river import compose, base
from dataclasses import dataclass
from typing import Dict, Tuple, NamedTuple

@dataclass
class ProcessedFrame:
    """Container for frame processing results"""
    footprints: np.ndarray
    fluorescence_traces: np.ndarray
    spikes: np.ndarray
    timestamp: float

class StreamState:
    """Container for maintaining stream state"""
    def __init__(self):
        self.pixel_stats = None
        self.component_stats = None
        self.residual_buffer = None
        self.num_components = 0
        self.overlapping_groups = None

    def update(self, result: Dict[str, Any]):
        """Update state with latest processing results"""
        self.pixel_stats = result.get('pixel_stats', self.pixel_stats)
        self.component_stats = result.get('component_stats', self.component_stats)
        self.residual_buffer = result.get('residual_buffer', self.residual_buffer)
        self.num_components = result.get('num_components', self.num_components)
        self.overlapping_groups = result.get('overlapping_groups', self.overlapping_groups)

class CompleteStreamProcessor:
    def __init__(self):
        # Initialize component manager
        self.component_manager = ComponentManager()
        
        # Initialize preprocessor and motion stabilizer
        self.preprocessor = compose.Pipeline(
            ('preprocessor', PreprocessorTransformer()),
            ('motion_stabilizer', MotionStabilizerTransformer()),
        )

        # Initialize components
        self.initializers = compose.Pipeline(
            ('footprints', FootprintsInitializer(
                params=FootprintsInitializerParams()
            )),
            ('traces', TracesInitializer(
                params=TracesInitializerParams()
            )),
            ('pixel_stats', PixelStatsTransformer(
                params=PixelStatsParams()
            ))
        )

        # Create processing pipeline using River's compose
        self.pipeline = compose.Pipeline(
            ('trace_updater', TraceUpdaterTransformer()),
            ('deconvolver', DeconvolverTransformer()),
            ('component_detector', ComponentDetectorTransformer()),
            ('stats_tracker', StatsTrackerTransformer()),
            ('footprint_tracker', FootprintTrackerTransformer())
        )

        # Initialize state containers
        self.state = StreamState()
    
    def process_video_stream(self, video_stream):
        """Process video stream using River's incremental learning approach"""
        for frame, timestamp in video_stream:
            # Preprocess frame
            frame = self.preprocessor.learn_transform_one({
                'frame': frame,
                'timestamp': timestamp
            })

            if not (self.initializers.footprints.is_initialized and 
                   self.initializers.traces.is_initialized and ...):
                # Still in initialization phase
                self.component_manager = self.initializers.learn_transform_one(
                    self.component_manager, 
                    frame
                )
                
                # Skip update phase until both initializers are ready
                continue

            # Update and transform pipeline with new frame
            result = self.pipeline.learn_transform_one({
                'timestamp': timestamp,
                'state': self.state,
                'frame': frame,
                'component_manager': self.component_manager
            })
            
            # Update state
            self.state.update(result)

            yield ProcessedFrame(
                footprints=result.component_manager.footprints,
                fluorescence_traces=result.component_manager.traces,
                spikes=result.component_manager.spikes,
                timestamp=timestamp
            )
```

## Example usage

```python
# Example usage:
def process_video(initial_frames: xr.DataArray, streaming_frames):
    processor = CompleteStreamProcessor()
    
    # Process streaming data
    for result in processor.process_video_stream(streaming_frames):
        # Handle results (visualize, save, etc.)
        pass
```

## Example of transformer implementation

```python
class StreamComponent(base.Transformer):
    """Base class for all stream processing components"""
    def learn_one(self, frame: Any) -> 'StreamComponent':
        return self

    def transform_one(self, frame: Any) -> Any:
        raise NotImplementedError

class PreprocessorTransformer(StreamComponent):
    def transform_one(self, X: Dict[str, Any]) -> Dict[str, Any]:
        frame = X['frame']
        processed_frame = self.process_frame(frame)
        return {**X, 'frame': processed_frame}

    def process_frame(self, frame):
        # Implementation of frame preprocessing
        pass
      
class MotionStabilizerTransformer(StreamComponent):
    def transform_one(self, X: Dict[str, Any]) -> Dict[str, Any]:
        frame = X['frame']
        stabilized = self.process_frame(frame)
        return {**X, 'frame': stabilized}

class DeconvolverTransformer(StreamComponent):
    def transform_one(self, X: Dict[str, Any]) -> Dict[str, Any]:
        traces = X['fluorescence_traces']
        fluorescence_traces, spiking_traces = self.deconvolve_traces(
            traces, gamma=0.95, lambda_=0.01, min_spike_size=0.1
        )
        return {
            **X,
            'fluorescence_traces': fluorescence_traces,
            'spiking_traces': spiking_traces
        }
```

## Monitoring

```python
from river import metrics

class MonitoredStreamProcessor(CompleteStreamProcessor):
    def __init__(self):
        super().__init__()
        self.processing_time = metrics.Rolling(window_size=100)
        self.memory_usage = metrics.Rolling(window_size=100)

    def process_video_stream(self, video_stream):
        for frame, timestamp in video_stream:
            start_time = time.time()
            
            result = yield from super().process_video_stream([(frame, timestamp)])
            
            # Update metrics
            self.processing_time.update(time.time() - start_time)
            self.memory_usage.update(psutil.Process().memory_info().rss)
            
            yield result
```
