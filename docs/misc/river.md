# River

Functions and classes from the River library that are relevant to the streaming CNMF package. Besides the ones listed
below, River also has a `rolling` module that allows for rolling analyses, `active` module which allows for user
intervention, and a `preprocessing` module that allows for easy creation of feature extractors.

## Drift Detection

Detecting when video characteristics change significantly - For Cell Detection / Merging / Splitting

```python
from river import drift
from river import metrics
import numpy as np

# Initialize drift detector and metric
drift_detector = drift.ADWIN()
metric = metrics.MAE()


class VideoStreamProcessor:
    def process_stream(self, video_stream):
        for frame in video_stream:
            # Convert frame to features (e.g., brightness, color histograms)
            features = self.extract_features(frame)

            # Make prediction and update model
            pred = model.predict_one(features)

            # Check for drift in brightness
            avg_brightness = np.mean(features['brightness'])
            if drift_detector.update(avg_brightness):
                print(f"Change detected at frame {frame_number}")
                # Could trigger model retraining or adaptation here
```

## Running Statistics

Maintaining statistics without storing historical frames - For statistics that are updated as the data streams in.

```python
from river import stats


class VideoStatsTracker:
    def __init__(self):
        # Initialize running statistics
        self.brightness_mean = stats.Mean()
        self.motion_var = stats.Var()
        self.rolling_mean = stats.RollingMean(window_size=30)  # Last 30 frames

    def update_stats(self, frame):
        features = self.extract_features(frame)

        # Update statistics with new frame
        self.brightness_mean.update(features['brightness'])
        self.motion_var.update(features['motion'])
        self.rolling_mean.replace(features['activity'])

        current_stats = {
            'avg_brightness': self.brightness_mean.get(),
            'motion_variance': self.motion_var.get(),
            'recent_activity': self.rolling_mean.get()
        }
        return current_stats
```

## Time-based Features

Processing video with temporal context - For real-time processing, closed-loop processing, etc.

```python
from river import preprocessing, feature_extraction


class TemporalVideoProcessor:
    def __init__(self):
        # Create temporal feature extractors
        self.time_features = feature_extraction.TargetAgg(
            window_size=timedelta(seconds=5)  # 5-second window
        )
        self.activity_window = preprocessing.SlidingWindow(
            window_size=30  # Last 30 frames
        )

    def process_frame(self, frame, timestamp):
        features = self.extract_features(frame)

        # Add temporal features
        temporal_features = {
            'time_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'activity_window': self.activity_window.replace(features['activity']),
            'avg_motion_5sec': self.time_features.update(features['motion'])
        }
        return temporal_features
```
