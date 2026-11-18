r"""Provisional numerical utilities for some moment calculations in Consenrich.

- These functions get called in deep loops and should be
`nogil/inline/noexcept` whenever possible.

- Functions placed here need heavy stress testing!

"""

from libc.math cimport fabs

# chan 1983 doi:10.1080/00031305.1983.10483115.
# CALLERS: cconsenrich._accumulateObservationValue,
#          cconsenrich._collapseObservationStats.
cdef inline void centeredMomentsAdd(
    double value,
    double weight,
    double* totalWeight,
    double* mean,
    double* M2,
) noexcept nogil:
    """Update totalWeight, mean, and the weighted sum of squared deviations (M2) while adding up replicate data

    - `value` is the current replicate's signal or prediction residual.
    - `weight` is the current observation precision val
    - `totalWeight` is the sum of replicate precision weights
    - `mean` is the precision-weighted mean or prediction residual
    - `M2` is the precision-weighted sum of squared deviations from `mean`
    """
    cdef double total, delta, fraction
    if weight == 0.0:
        return

    # first value -- init totalWeight and mean, return
    if totalWeight[0] == 0.0:
        totalWeight[0] = weight
        mean[0] = value
        return

    # Use the deviation from the current mean to update second moment,
    # this avoids cancellation errors in sum(w*x*x) - sum(w)*mean**2.
    total = totalWeight[0] + weight
    delta = value - mean[0]
    fraction = weight / total
    M2[0] += delta * delta * (totalWeight[0] * fraction)
    mean[0] += delta * fraction
    totalWeight[0] = total


# CALLERS: cuncertainty.cobservationTotalInformation,
#          cuncertainty.cmakeFoldMaskAndInformation.
cdef inline void shiftedMomentsAdd(
    double value,
    double* count,
    double* origin,
    double* sumOffset,
    double* sumSquaredOffset,
) noexcept nogil:
    r"""Accumulate shifted moments of square-root precision for the replicate dependence adjustment."""
    cdef double offset
    if count[0] == 0.0:
        origin[0] = value
    offset = value - origin[0]
    sumOffset[0] += offset
    sumSquaredOffset[0] += offset * offset
    count[0] += 1.0


# CALLERS: cconsenrich._muncObservationMomentSeedPassInterval.
cdef inline void compensatedSumAdd(
    double value,
    double* total,
    double* correction,
) noexcept nogil:
    """Add value to total while tracking rounding error in correction

    Consenrich uses the corrected sum for covariance centering, MUNC
    weight updates, and effective sample size estimation

    - `correction` tracks rounding errors, so callers use
      `total + correction` as the corrected sum
    """
    cdef double updated = total[0] + value
    if fabs(total[0]) >= fabs(value):
        correction[0] += (total[0] - updated) + value
    else:
        correction[0] += (value - updated) + total[0]
    total[0] = updated
