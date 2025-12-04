"""
Merge components that have already been registered with each other,
if they spatially overlap and temporally correlate significantly.

This step is complementary to the catalog node, for the cases in which
two components should have been merged, but the buffered data's SNR was
temporarily too low to build a reliable merge matrix, causing the two
components to remain separated.

"""
