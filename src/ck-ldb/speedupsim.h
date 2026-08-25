/**
 * \addtogroup CkLdb
 */
/*@{*/

/** Predicting this job's makespan at widths it is not running at.
 *
 * An elastic scheduler has to compare "how fast is this job now" against "how
 * fast would it be on twelve workers", and the second half is not measurable
 * -- the job is not there. The usual answer is to fit a two-parameter curve to
 * whatever widths the job has happened to run at, which extrapolates smoothly
 * through exactly the features that matter: the granularity ceiling where
 * objects run out, and the imbalance step when the largest object stops
 * sharing a processor.
 *
 * The load balancer already knows those. It holds every object's time for the
 * window just measured, so re-packing that same set of objects onto a
 * different number of bins answers the question by construction, with no
 * extrapolation. What it cannot answer is anything about *ordering* -- the LB
 * statistics are per-window totals, not a trace, so there is no critical path
 * here and no event simulation is possible. What comes out is the makespan of
 * the work, not the makespan of the job.
 *
 * The model therefore comes in two tiers, and the second is opportunistic:
 *
 *  - **Packing** (always). Objects are re-packed onto the target width, giving
 *    granularity and imbalance exactly. Needs only `objData`, which a central
 *    balancer already gathers on every step, so this costs nothing beyond a
 *    sort on the balancer's PE.
 *
 *  - **Communication** (only when the job already runs with `+LBCommOn`).
 *    Given the hypothetical placement, every edge that would newly cross
 *    processors is charged at the balancer's own alpha/beta model. This is the
 *    term that makes wider look worse, and it is the reason the packing tier
 *    alone reads as optimistic.
 *
 * Communication instrumentation is off by default in Charm++ and this code
 * deliberately does not turn it on: it uses `commData` when the job is already
 * collecting it and reports which tier ran, rather than imposing the cost of
 * an edge list on every job that wants a speedup estimate.
 */

#ifndef CK_SPEEDUP_SIM_H
#define CK_SPEEDUP_SIM_H

#include "BaseLB.h"

/** One point on the predicted curve. */
struct CkWidthPrediction
{
  int numPes;      /**< the width predicted for */
  double predWall; /**< predicted makespan of this window at that width */
};

/** The packing tier ran: granularity and imbalance are modelled. */
#define CK_WIDTHMODEL_PACKING (1u << 0)
/** The communication tier ran too, because `commData` was populated. */
#define CK_WIDTHMODEL_COMM (1u << 1)

/** Predict this window's makespan at each of `widths`.
 *
 * `out` must hold `nwidths` entries. Returns the CK_WIDTHMODEL_* tiers that
 * actually ran, or 0 if there was nothing to model. Does not modify `stats`
 * apart from building its communication hash, which is idempotent and which
 * the caller's own accounting builds anyway.
 */
unsigned int CkPredictWidths(BaseLB::LDStats* stats, const int* widths, int nwidths,
                             CkWidthPrediction* out);

/** Choose which widths to predict at: log-spaced over [lo, hi], always
 *  including `current` so the caller can check a prediction against the
 *  measurement it already has.
 *
 * Log spacing because the interesting variable is objects per processor, and
 * equal ratios in width are equal steps in that. `numObjs` clamps the top:
 * past one object per processor there is nothing left to divide, so wider
 * widths are all the same point on the curve. Writes at most `maxOut` widths,
 * ascending and deduplicated, and returns how many.
 */
int CkChooseWidths(int lo, int hi, int current, int numObjs, int* out, int maxOut);

#endif

/*@}*/
