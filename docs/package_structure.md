# Design Doc

## Roadmap

Performing test-driven development, starting with the unit tests. Hence the empty classes in the src/cala/streaming_service/ folder.

## Introduction

Building a streaming CNMF package - starting with the module / class designs.
This pacakge should be able to accomplish the following:

1. Function like an ML pipeline (hopefully inherit from Sklearn classes)
2. Support streaming operations (Sklearn class instances are light, so that's good. Not sure how it marries into the streaming operations, since it's primarily designed for batch one-shot fit / transform. Operations like partial_fit might have to be custom written.)
3. Maybe instead of frame by frame, a few frames at a time? 
4. Output / save visualizations and matrix data real time at all stages of processing 
5. Store parameters and hyperparameters, and exposed so that the user can view/modify them while processing. 
6. Parameters get updated real time as the new data streams in. 
7. The learned / updated parameters should be able to retroactively propagate the earlier data to refine the results. This may have to be done post-processing. 
8. Plug smoothly into the batch processing side of the pacakge.

## Data Flow

1. Initialization Phase:
    * The process starts with initialize_online() method which:
        * Loads initial batch of frames from the movie
        * Performs optional motion correction on the initial batch
        * Normalizes the data if specified
        * Initializes the spatial (A) and temporal (C) components either through:
          * "bare" initialization (no initialization)
          * "seeded" initialization (using provided spatial footprints)

2. Main Processing Loop (fit_online()):
    * For each epoch and each file:
        * Frames are loaded iteratively using caiman.base.movies.load_iter()
        * For each frame:  

a. Pre-processing:

   * Optional background model subtraction using CNN if specified
   * Downsampling if ds_factor > 1
   * Normalization if enabled  

b. Motion Correction (mc_next()):

   * Corrects motion using either rigid or piecewise-rigid registration
   * Returns motion-corrected frame  

c. Core Processing (fit_next()):

   * Updates temporal components (C) using HALS algorithm
   * Deconvolves neural activity using OASIS
   * Updates spatial components (A) if needed
   * Detects new components if enabled:
      * Computes correlation image
      * Finds local maxima
      * Tests candidate components
      * Adds accepted components to the model

d. Visualization (optional):

   * Creates visualization frame showing:
     * Raw data
     * Reconstructed components
     * Residuals
     * Newly detected components highlighted

3. Data Structures and Buffers:
    * Uses ring buffers to maintain recent frames and residuals
    * Maintains sufficient statistics (CY, CC) for online updates
    * Tracks components in estimates object containing:
      * Spatial components (Ab)
      * Temporal components (C_on)
      * Background components (b, f)
      * Noise estimates (sn)
4. Output:
    * Final estimates contain:
      * Spatial footprints of neurons (A)
      * Temporal traces (C)
      * Background components (b, f)
      * Deconvolved neural activity (S)

## Class Structure

* core
* data
* components
* preprocessing
* motion
* initialization
* detection
* visualization
* utils
