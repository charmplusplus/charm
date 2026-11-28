/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBOBJ_H
#define LBOBJ_H

#include "lbdb.h"

class LBDatabase;

class LBObj
{
friend class LBDatabase;

public:
  LBObj(const LDObjHandle &_h, void *usr_ptr = NULL, bool _migratable=true, bool _asyncArrival = false) {
    data.handle = _h;
    data.migratable = _migratable;
    data.asyncArrival = _asyncArrival;
    Clear();
//    data.cpuTime = 0.;
//    data.wallTime = 0.;
//    data.minWall = 1e6;
//    data.maxWall = 0.;
    localUserData = usr_ptr;
//    migratable = _migratable;
//    registered = true;
    startWTime = -1.0;
    lastWallTime = .0;
#if CMK_LB_CPUTIMER
    startCTime = -1.0;
    lastCpuTime = .0;
#endif
  }

  ~LBObj() { };

  void Clear(void);

  void IncrementTime(LBRealType walltime, LBRealType cputime);
  void IncrementGPUTime(LBRealType gputime);

  // Has this object joined the current load balancing step? While it has, its
  // work is not billed to it: the strategy reads (or has read) the load it
  // accumulated up to its join, and anything after belongs to the next round.
  // Under the unsplit barrier that is never true for long, because a joined
  // element waits and does no work; under +LBAsync it keeps running.
  //
  // Per object, deliberately: a PE-wide switch stops measuring elements that
  // have NOT joined yet, which biases the late ones the other way.
  bool joinedStep = false;
  // A device load declared while the window was closed (LBDatabase::
  // EstObjGPULoad), which the round therefore does not see; cleared when the
  // window opens again.
  bool declaredWhileJoined = false;
  // When this object's measurement window last opened, or -1 while it is
  // closed. Its open time accumulates into data.windowTime.
  double windowOpenedAt = -1.0;
  // The span this interval's load stands for: from the object's first
  // activity after ClearLoads (activeSince, set by StartTimer) to its join
  // (closedAt), or to now while it is open. Under +LBAsync an object keeps
  // computing through the step with its window closed, so the span is the
  // whole interval and the window a fraction of it; under the barrier it joins
  // and then waits idle, so span and window coincide and nothing is scaled.
  // Wall time since ClearLoads would count that idle wait as unmeasured work
  // and inflate exactly the light objects.
  double activeSince = -1.0;
  double closedAt = -1.0;

  inline void setJoinedStep(bool v) {
    if (v && !joinedStep) {            // closing: bank the time it was open
      const double now = CkWallTimer();
      if (windowOpenedAt >= 0.0) data.windowTime += now - windowOpenedAt;
      windowOpenedAt = -1.0;
      closedAt = now;
    } else if (!v && joinedStep) {     // opening
      windowOpenedAt = CkWallTimer();
      closedAt = -1.0;
      declaredWhileJoined = false;
    }
    joinedStep = v;
  }
  inline bool hasJoinedStep(void) const { return joinedStep; }
  // The window time so far this interval, the open tail included.
  inline double windowTime() const {
    return data.windowTime + (windowOpenedAt >= 0.0 ? CkWallTimer() - windowOpenedAt : 0.0);
  }
  // A new interval starts: nothing banked, and a window that is open restarts
  // its clock here, where the interval's own clock restarts (ClearLoads).
  inline void clearWindow(double now) {
    data.windowTime = 0.0;
    if (windowOpenedAt >= 0.0) windowOpenedAt = now;
    activeSince = -1.0;
    closedAt = -1.0;
  }
  // The span this interval's sample stands for; see activeSince.
  inline double activeSpan() const {
    if (activeSince < 0.0) return 0.0;
    const double end = closedAt >= 0.0 ? closedAt : CkWallTimer();
    return end > activeSince ? end - activeSince : 0.0;
  }
  // activeSpan / window: the factor that turns this object's sampled load and
  // traffic into a rate over the time it was actually active. 1 when the
  // window covers the span; capped, since a window seen for a tenth of the
  // span supports no finer an extrapolation than that.
  static constexpr double kMaxSampleScale = 10.0;
  inline double sampleScale() const {
    const double w = windowTime();
    const double span = activeSpan();
    if (w <= 0.0 || span <= w) return 1.0;
    const double f = span / w;
    return f > kMaxSampleScale ? kMaxSampleScale : f;
  }

  inline void StartTimer(void) {
	startWTime = CkWallTimer();
	if (activeSince < 0.0) activeSince = startWTime;  // first activity this interval
#if CMK_LB_CPUTIMER
	startCTime = CkCpuTimer();
#endif
  }
  inline void StopTimer(LBRealType* walltime, LBRealType* cputime) {
	if (startWTime >= 0.0) {	// in case startOn in middle of entry
          const double endWTime = CkWallTimer();
	  *walltime = endWTime - startWTime;
#if CMK_LB_CPUTIMER
          const double endCTime = CkCpuTimer();
	  *cputime = endCTime - startCTime;
#else
	  *cputime = *walltime;
#endif
	}
        else {
          *walltime = *cputime = 0.0;
        }
  }

  inline void getTime(LBRealType *w, LBRealType *c) {
    *w = data.wallTime;
#if CMK_LB_CPUTIMER
    *c = data.cpuTime;
#else
    *c = *w;
#endif
  }

  inline void setTiming(LBRealType cputime)
  {
    data.wallTime = cputime;
#if CMK_LB_CPUTIMER
    data.cpuTime = cputime;
#endif
  }

  inline void getGPUTime(LBRealType *w)
  {
#if CMK_CUDA
    *w = data.gpuTime;
#else
    CmiAbort("LBObj::getGPUTime called but CMK_CUDA is not set");
#endif
  }

  inline void setGPUTiming(LBRealType gputime)
  {
#if CMK_CUDA
    data.gpuTime = gputime;
#else
    CmiAbort("LBObj::setGPUTiming called but CMK_CUDA is not set");
#endif
  }

  inline LDOMHandle &parentOM() { return data.handle.omhandle; }
  inline const LDObjHandle &GetLDObjHandle() const { return data.handle; }
  inline void SetMigratable(bool mig) { data.migratable = mig; }
  inline void setPupSize(size_t obj_pup_size) {
    data.pupSize = pup_encodeSize(obj_pup_size);
  }
  inline void setGPUPupSize(size_t obj_gpu_pup_size) {
#if CMK_CUDA
    data.gpuPupSize = obj_gpu_pup_size;
#else
    CmiAbort("LBObj::setGPUPupSize called but CMK_CUDA is not set");
#endif
  }
  inline void UseAsyncMigrate(bool async) { data.asyncArrival = async; }
  inline LDObjData &ObjData() { return data; };
  inline void lastKnownLoad(LBRealType *w, LBRealType *c) {
    *w = lastWallTime;
#if CMK_LB_CPUTIMER
    *c = lastCpuTime;
#else
    *c = *w;
#endif
  }
  inline void *getLocalUserData() { return  localUserData; }
#if CMK_LB_USER_DATA
  inline void *getDBUserData(int idx) { return  data.getUserData(idx); }
#endif
private:

  void *localUserData;               // local user data, not in database
//  LDOMHandle parentOM;
//  LDObjHandle myhandle;
  LDObjData data;
//  bool registered;
  double startWTime;             // needs double precision
  LBRealType lastWallTime;
#if CMK_LB_CPUTIMER
  double startCTime;
  LBRealType lastCpuTime;
#endif
//  bool migratable;   // temp
};

#endif

/*@}*/
