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
    // Unknown until the application declares them (setPupSize/setGPUPupSize).
    // Left uninitialised these reached the cost model as garbage -- 10^16
    // bytes per object, a migration priced at 10^8 seconds -- and every priced
    // decision refused to move anything.
    data.pupSize = 0;
#if CMK_GPU
    data.gpuPupSize = 0;
#endif
    // Not replicable until the application declares it (setLBReplica).
    data.replicaGroup = -1;
    data.replicaRank = 0;
    data.replicaCount = 1;
    data.replicaMax = 1;
    data.replicaExtraBytes = 0;
    data.replicaCostNode = data.replicaCostRemote = 0.0;
    data.replicaParts = 0;
    data.replicaExchangeBytes = 0;
    data.replicaExchangeMsgs = 0.0;
    data.replicaExtraBytesPerPeer = 0;
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

  // The application declared this object's device load for the current
  // interval (EstObjGPULoad), so the CUPTI attribution must not replace it.
  // An application that declares knows its cost before the work runs -- moe
  // prices an expert at its token count times a measured rate -- and the
  // attribution can be far off it: measured on moe, CUPTI credited the
  // experts with 17% of the interval while the devices were 68-77% busy.
  // Cleared with the loads at each balancing step, so an object that stops
  // declaring goes back to being measured.
  bool gpuDeclared = false;
  inline void setGPUDeclared(bool v) { gpuDeclared = v; }
  inline bool hasGPUDeclared(void) const { return gpuDeclared; }

  inline void StartTimer(void) {
	startWTime = CkWallTimer();
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

  // Object position, for geometric load balancers (DiffusionLB's centroid
  // metric). Empty unless the application calls CkMigratable::setObjPosition,
  // so every reader has to cope with a zero-length vector.
  inline void setPosition(const std::vector<LBRealType>& pos) { data.position = pos; }
  inline const std::vector<LBRealType>& getPosition() const { return data.position; }

  // Replication declaration; see LDObjData::replicaGroup.
  inline void setReplica(int group, int rank, int count, int maxCount, size_t extraBytes,
                         double costNode = 0.0, double costRemote = 0.0, int parts = 0,
                         size_t exchangeBytes = 0, double exchangeMsgs = 0.0, size_t extraPerPeer = 0)
  {
    data.replicaGroup = group;
    data.replicaRank = rank;
    data.replicaCount = count;
    data.replicaMax = maxCount;
    data.replicaExtraBytes = extraBytes;
    data.replicaCostNode = costNode;
    data.replicaCostRemote = costRemote;
    data.replicaParts = parts;
    data.replicaExchangeBytes = exchangeBytes;
    data.replicaExchangeMsgs = exchangeMsgs;
    data.replicaExtraBytesPerPeer = extraPerPeer;
  }
  inline void getReplica(int& group, int& rank, int& count, int& maxCount, size_t& extraBytes) const
  {
    group = data.replicaGroup;
    rank = data.replicaRank;
    count = data.replicaCount;
    maxCount = data.replicaMax;
    extraBytes = data.replicaExtraBytes;
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
