/**
 * \addtogroup CkLdb
 */
/*@{*/

#include "rescalepoint.h"

#include "charm++.h"
#include "converse.h"
#include "cksyncbarrier.h"

/** Per-PE, not per-object: every element on a PE agrees about whether a
 *  rescale is pending, and the reduction that picks the iteration is over
 *  PEs. Per-PE rather than file-scope so an SMP build's worker threads do not
 *  share one word.
 *
 *  _loadbalancerInit resets these on every start, survivor restarts included,
 *  which is what disarms a job whose rescale has committed. */
CkpvDeclare(int, _rescaleArmedAt);
CkpvDeclare(int, _rescaleIterSeen);
CkpvDeclare(bool, _rescaleUsesBoundaries);
/** Whether _rescaleArmedAt is the agreed iteration or only the earliest one
 *  this PE could still be caught at. While it is only the latter, the load
 *  balancing barrier is shut, so an element that reaches the ceiling and calls
 *  AtSync() parks there instead of starting a round. */
CkpvDeclare(bool, _rescaleFinalKnown);
/** Whether the barrier was open when we shut it, so the commit restores what
 *  it found rather than forcing it open over, say, an object registration. */
CkpvDeclare(bool, _rescaleBarrierWasOn);

/** Disarmed. Chosen so the armed test is a single comparison with no separate
 *  flag to read: no iteration number reaches it. */
#define RESCALE_DISARMED 2147483647 /* INT_MAX */

void CkRescalePointInit(void)
{
  CkpvInitialize(int, _rescaleArmedAt);
  CkpvInitialize(int, _rescaleIterSeen);
  CkpvInitialize(bool, _rescaleUsesBoundaries);
  CkpvInitialize(bool, _rescaleFinalKnown);
  CkpvInitialize(bool, _rescaleBarrierWasOn);
  CkpvAccess(_rescaleArmedAt) = RESCALE_DISARMED;
  CkpvAccess(_rescaleIterSeen) = -1;
  CkpvAccess(_rescaleFinalKnown) = false;
  CkpvAccess(_rescaleBarrierWasOn) = false;
#if CMK_SHRINK_EXPAND
  // Whether this job's elements declare boundaries decides which rescale path
  // a request takes: the boundary consensus, or an immediate barrier-less
  // load balancing round. A survivor must remember the answer across the
  // longjmp -- its elements will not have re-called checkRescale by the time
  // a chained request, deferred through the restore, asks the question -- or
  // a boundary app's second rescale would silently lose its boundary
  // guarantee. Newcomers start false; the deciding reduction takes the max.
  extern bool _reuseRegistrationStateOnRestart;
  if (!_reuseRegistrationStateOnRestart) CkpvAccess(_rescaleUsesBoundaries) = false;
#else
  CkpvAccess(_rescaleUsesBoundaries) = false;
#endif
}

bool CkRescaleCheck(int iter)
{
  CkpvAccess(_rescaleUsesBoundaries) = true;
  if (iter > CkpvAccess(_rescaleIterSeen)) CkpvAccess(_rescaleIterSeen) = iter;
  return iter >= CkpvAccess(_rescaleArmedAt);
}

bool CkRescalePending(int iter) { return CkRescaleCheck(iter); }

bool CkRescaleArmed(void)
{
  CkpvAccess(_rescaleUsesBoundaries) = true;
  return CkpvAccess(_rescaleArmedAt) != RESCALE_DISARMED;
}

void CkRescaleArmAt(int iter)
{
  CkpvAccess(_rescaleArmedAt) = iter;
  CkpvAccess(_rescaleFinalKnown) = false;
  // Shut the barrier for the length of the agreement. Elements that reach the
  // ceiling call AtSync() and park as ordinary arrived clients; without this
  // the round would start at an iteration that is not yet agreed, and could
  // not be taken back.
  CkSyncBarrier* b = CkSyncBarrier::object();
  if (b != NULL)
  {
    CkpvAccess(_rescaleBarrierWasOn) = b->isOn();
    b->turnOff();
  }
}

void CkRescaleCommit(int iter)
{
  const int previous = CkpvAccess(_rescaleArmedAt);
  // Publish first. Any element put back to work below re-checks its boundary
  // immediately, and must see the agreed iteration rather than the ceiling it
  // just stopped at -- otherwise it would stop again on the spot.
  CkpvAccess(_rescaleArmedAt) = iter;
  CkpvAccess(_rescaleFinalKnown) = true;

  CkSyncBarrier* b = CkSyncBarrier::object();
  if (b == NULL) return;

  // Agreed later than this PE guessed: elements parked at the old ceiling are
  // put back to work to reach the agreed iteration. Only those -- an element
  // that arrived at its own regular load balancing boundary stays put, and a
  // round already in flight is never touched (both filtered inside).
  if (iter > previous)
  {
    const int n = b->rewindArrivedClients(previous);
    if (n > 0)
      CmiPrintf("CharmLB> [%d] Rescale point moved from %d to %d; resumed %d element(s) to reach it.\n",
                CkMyPe(), previous, iter, n);
  }

  // Reopening runs checkBarrier, so a regular round that formed while the
  // barrier was shut -- every element arriving at its own LB boundary during
  // the agreement -- fires here, and the rescale rides it through the
  // pending_realloc_state path instead of the consensus target.
  if (CkpvAccess(_rescaleBarrierWasOn)) b->turnOn();
}

void CkRescaleDisarm(void)
{
  CkpvAccess(_rescaleArmedAt) = RESCALE_DISARMED;
  CkpvAccess(_rescaleFinalKnown) = false;
}

int CkRescaleIterSeen(void) { return CkpvAccess(_rescaleIterSeen); }

bool CkRescaleUsesBoundaries(void) { return CkpvAccess(_rescaleUsesBoundaries); }

/*@}*/
