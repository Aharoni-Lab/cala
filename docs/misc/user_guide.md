# User Guide

## Introduction

Cala is an endoscope image processing tool designed to be accessible to researchers of all programming backgrounds. Its
no-code approach is designed to be as intuitive and painless as possible while still providing the flexibility to
customize the processing pipeline.

## Quick Start

To run Cala, you'll need just two things:

1. Your video files
2. A configuration file

Then simply run:

```bash
python main.py --visual --config cala_config.yaml
```

## Detailed Setup Guide

### 1. Preparing Your Data

1. **Organize Your Videos**
    - Place all your video files in a single directory
    - Make note of the video order if processing multiple files

2. **Choose Output Location**
    - Select a directory where processed data will be saved
    - Ensure you have sufficient disk space

### 2. Configuration File Setup

The configuration file is where you set up the entire process. It consists of three main components:

#### A. Basic Settings

```yaml
user_dir: .                                    # Working directory
config_path: cala_config.yaml                  # Name of this config file
video_dir: videos                              # Directory containing your videos
video_files: [ 01.mp4, 02.mp4, 03.mp4 ]       # List of video files to process
output_dir: output                             # Where to save results
output_name: 12252025                          # Name prefix for output files
```

#### B. Pipeline Structure

The pipeline is organized into three main stages:

1. **Preprocessing**: Initial video cleanup and preparation

2. **Initialization**: Setting up analysis parameters

3. **Iteration**: Refining and updating results

#### C. Processing Nodes

Each processing step is handled by a specialized node. To specify a node, you need to specify the following:

- **Transformer**: The node to use
- **Params**: The parameters to use
- **N_frames**: The number of frames to use for the node
- **Requires**: The nodes that must be run before this one

An example of a node is the **Downsampler** node (reduces video resolution):

```yaml
  downsample:
    transformer: Downsampler
    params:
      method: mean
      dimensions: [ width, height ]
      strides: [ 2, 2 ]
    n_frames: 1
    requires: [ denoise, glow_removal ]
```

### 3. Advanced Features

- **Visual Mode**: Use `--visual` flag for real-time processing visualization

## Example Configuration

A complete example configuration file is provided below. You can use this as a template and modify parameters as needed
for your specific use case.

```yaml
# cala_config.yaml

user_dir: .
config_path: cala_config.yaml
video_dir: videos
video_files: [ 01.mp4, 02.mp4, 03.mp4 ]
output_dir: output
output_name: 12252025

pipeline:
  preprocess:
    downsample:
      transformer: Downsampler
      params:
        method: mean
        dimensions: [ width, height ]
        strides: [ 2, 2 ]

    denoise:
      transformer: Denoiser
      params:
        method: gaussian
        kwargs:
          ksize: [ 3, 3 ]
          sigmaX: 1.5
      requires: [ downsample ]

    glow_removal:
      transformer: GlowRemover
      params:
        learning_rate: 0.1
      requires: [ denoise ]

    motion_stabilization:
      transformer: RigidStabilizer
      params:
        drift_speed: 1
        anchor_frame_index: 0
      requires: [ glow_removal ]

  initialization:
    footprints:
      transformer: FootprintsInitializer
      params:
        threshold_factor: 0.5
        kernel_size: 3
        distance_metric: 2 # cv2.DIST_L2
        distance_mask_size: 5

    traces:
      transformer: TracesInitializer
      params: { }
      n_frames: 3
      requires: [ footprints ]

    pixel_stats:
      transformer: PixelStatsInitializer
      params: { }
      n_frames: 3
      requires: [ traces ]

    component_stats:
      transformer: ComponentStatsInitializer
      params: { }
      n_frames: 3
      requires: [ traces ]

    residual:
      transformer: ResidualInitializer
      params:
        buffer_length: 3
      n_frames: 3
      requires: [ footprints, traces ]

    overlap_groups:
      transformer: OverlapsInitializer
      params: { }
      requires: [ footprints ]

  iteration:
    traces:
      transformer: TracesUpdater
      params:
        tolerance: 0.001

    pixel_stats:
      transformer: PixelStatsUpdater
      params: { }
      requires: [ traces ]

    component_stats:
      transformer: ComponentStatsUpdater
      params: { }
      requires: [ traces ]

    detect:
      transformer: Detector
      params:
        num_nmf_residual_frames: 10
        gaussian_std: 4.0
        spatial_threshold: 0.8
        temporal_threshold: 0.8
      requires: [ pixel_stats, component_stats ]

    footprints:
      transformer: FootprintsUpdater
      params:
        boundary_expansion_pixels: 1
      requires: [ detect ]

    overlaps:
      transformer: OverlapsUpdater
      params: { }
      requires: [ footprints ]
```
