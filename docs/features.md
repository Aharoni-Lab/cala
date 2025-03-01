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
