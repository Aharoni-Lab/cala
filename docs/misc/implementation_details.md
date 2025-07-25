# Implementation Details

by Raymond Chang

## Contents

1. Structure
2. Reference
3. Deviation

## 3. Deviation

Due to performance / efficiency issues, we have different implementations from what is written on the publications.

* within iteration loop, `W` and `M` are updated before the new cell detection stage: This prevents the sufficient
  statistics of the new components from getting updated twice with the new frame.

* boundary exploration in footprints update: boundary of the footprints can expand by n-pixels during the update
  iterations. this also prevents "the glow" of an existing cell possibly getting registered as a new cell.

* we do not ask for cell size from users. instead, we calculate the average cell size from the existing ones, and
  uses that value to search for possible new footprint.

* footprint update uses tolerance instead of a set iteration count.

* overlap is updated after footprint update. since footprints can shrink/expand, this step is critical.

* we only build cells when the residuals are higher than a threshold value: this prevents detection stage from trying
  to build useless cells from trace amounts of residual

* the footprint and traces in detect steps are "normalized" to the actual movie pixel values. (max of the footprint = max of the video patch)
this stems from an effort to be able to merge two parts of the same cell that are discovered during the same cycle.

* the cold start / detect algorithm have been modified to account for cell overlaps
  * we detect the most energetic area in the residual (what if this point is 1. entirely negative, or 2. has negative sections within)
  * we slice out the energetic area from the residual
  * perform a rank-1 nmf with this parfait
  * calculate trace correlations of this new component with spatially overlapping components
  * if the correlation is above threshold, we merge (what if there are more than one component to merge?)
  
* what if the initial guess is an overlap point
  * during the next trace update, if only one of them fires, there will be error
  * this will become residual and gets added as a cell!

* TODO: traces and sufficient stats should be updated after footprints are updated.
