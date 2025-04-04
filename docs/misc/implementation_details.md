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

* TODO: traces and sufficient stats should be updated after footprints are updated.
