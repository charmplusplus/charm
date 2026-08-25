/**
 * \addtogroup CkLdb
 */
/*@{*/

/** Application-declared rescale points.
 *
 * An elastic scheduler can ask a job to change width at any moment, but a job
 * can only change width where its objects are quiescent, which in practice
 * means an iteration boundary. The runtime already has one such boundary --
 * the load balancing step -- so without help the scheduler's latency is the
 * job's LB period, which for a job that balances every ten or fifty
 * iterations is far longer than the rescale itself.
 *
 * These calls let an application offer every iteration boundary instead. It
 * asks, at each one, whether a rescale is pending; when the answer is yes it
 * quiesces and calls AtSync() there rather than at its usual cadence. Nothing
 * new happens at the barrier -- CentralLB::CheckForLB already asks whether a
 * rescale is pending on every round -- so this adds no second path through
 * the most delicate code in the runtime. The cost when no rescale is pending
 * is one comparison.
 *
 * The pending state is an iteration number rather than a flag, and that
 * matters. A flag is carried by a broadcast, so different PEs observe it at
 * different iterations, and elements would enter the barrier spread across a
 * few iterations rather than all at the same one. Two things go wrong:
 *
 *  - The checkpoint then sees a heterogeneous population of sent-but-
 *    unconsumed messages, deeper for elements that are behind. In-flight
 *    messages across a rescale are the most fragile part of this path, and
 *    every rescale exercised so far has come from a uniform `iter % period`
 *    boundary where that population is uniform too.
 *
 *  - There is a narrow liveness race. An element whose neighbours have
 *    already stopped can run only as far as their last sent data allows,
 *    typically one iteration; if the broadcast has not been dequeued by then
 *    it parks waiting for data that will never come, and never reaches the
 *    boundary that would have let it observe the flag.
 *
 * Agreeing on an iteration number up front removes both. The agreement is a
 * three-phase consensus rather than a guess plus a margin:
 *
 *   PE 0   broadcast RescaleTentative(t),  t = its own iterSeen + lead
 *   all    local = max(t, iterSeen + 1); arm at local; report local
 *   PE 0   final = max over PEs (reduction)
 *   all    broadcast RescaleCommit(final); release anyone held
 *
 * The ordering inside the middle phase is the whole point. Arming happens
 * before reporting, in the same entry method, so no element can advance
 * between the read of iterSeen and the ceiling going in. The reduced maximum
 * is therefore a bound no element has passed -- by construction, not because
 * a margin happened to be generous enough.
 *
 * `iterSeen + 1` rather than `iterSeen`: iterSeen is the last *completed*
 * iteration, so the furthest element is already executing iterSeen+1 and will
 * next stand at that boundary. It is the earliest ceiling that can still catch
 * it.
 *
 * Between arming and commit an element can reach the ceiling before the
 * agreed iteration is known. It stops there -- checkRescale() answers true and
 * the application calls AtSync() as usual -- but the runtime keeps the load
 * balancing barrier shut for the duration, so the element parks as an arrived
 * client instead of starting a round that could not be taken back. On commit
 * the barrier reopens; if the agreed iteration turned out to be later than
 * this PE's ceiling, the parked elements are rewound and put back to work to
 * reach it. None of that is visible to the application.
 *
 * Parking cannot deadlock: the reduction is contributed by the LBManager group
 * branch, not by the elements, so it completes whether or not any element is
 * making progress, and the commit always arrives.
 *
 * See CkRescaleCheck().
 */

#ifndef CK_RESCALEPOINT_H
#define CK_RESCALEPOINT_H

#include <functional>

/** Is this iteration the one the job should change width at?
 *
 * Call at every iteration boundary; when it returns true, quiesce whatever
 * must not be in flight across a checkpoint and call AtSync().
 *
 *     if (checkRescale(iteration)) { quiesce(); AtSync(); }
 *
 * That is the entire application-side contract. It returns false at every
 * boundary but one, at the cost of a comparison, and it also records `iter`
 * -- which is how the runtime learns what iteration to agree on without
 * asking the application separately.
 *
 * `iter` must advance in step across every element that participates: the
 * runtime picks a single iteration and all of them stop at it. A counter that
 * means different things to different elements is not usable here.
 *
 * It can return true while the exact iteration is still being agreed. That is
 * deliberate and it is why the answer is safe: the element stops at the
 * earliest iteration anyone could still be caught at, and the runtime holds
 * the load balancing barrier shut until the agreement lands. If the agreed
 * iteration turns out to be later, the runtime puts the element back to work
 * to reach it -- the application sees nothing but an AtSync() that took a
 * little longer to be answered.
 *
 * Declining to call this is safe: the rescale then lands at the job's next
 * ordinary load balancing step, which is the behaviour without it. The cost of
 * forgetting is latency, not a hang.
 */
bool CkRescaleCheck(int iter);

/** As above, for a job with no globally meaningful iteration counter.
 *
 * Weaker: without an iteration to agree on, elements stop at whatever boundary
 * follows the broadcast rather than at one agreed value. Prefer the iteration
 * form wherever the application has a counter to give. The one place this is
 * as strong is inside a callback from a reduction the application already
 * performs every iteration -- every element reads it in the same round, so
 * they agree without needing a number.
 */
bool CkRescaleArmed(void);

/* Runtime-internal. The scheduler-facing arming protocol lives in LBManager. */

/** Install a tentative ceiling and shut the barrier, so elements that reach it
 *  park there instead of starting a load balancing round. */
void CkRescaleArmAt(int iter);
/** Publish the agreed iteration, put back to work any element that stopped
 *  short of it, and reopen the barrier. */
void CkRescaleCommit(int iter);
/** Stop offering boundaries. Called once a barrier round has begun, so that an
 *  arming which does not lead to a rescale cannot leave the job balancing on
 *  every iteration. */
void CkRescaleDisarm(void);
/** The largest iteration any element on this PE has reported, or -1 if none
 *  has. PE 0 picks the rescale iteration from its own value: a PE running
 *  ahead of the target simply enters at its next boundary, which is later
 *  than everyone else rather than earlier, and only entering early is
 *  unsafe. */
int CkRescaleIterSeen(void);
/** Whether any element on this PE has ever declared a boundary. False means
 *  the job has not opted in, and the rescale takes the pre-existing path of
 *  waiting for the next load balancing step. */
bool CkRescaleUsesBoundaries(void);
/** Reset to disarmed. Called from LBManager::init, so a newcomer starts
 *  unarmed and a fresh run starts unarmed. */
void CkRescalePointInit(void);

/** Arm immediately: every boundary from now on. Used when the job opted in
 *  but no element has reported an iteration number, i.e. it is using the
 *  CkRescaleArmed() form. */
#define CK_RESCALE_ARM_NOW (-2147483647 - 1) /* INT_MIN, without <limits.h> */

#endif

/*@}*/
