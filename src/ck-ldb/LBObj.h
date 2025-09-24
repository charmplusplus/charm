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
    localUserData = usr_ptr;
  }

  ~LBObj() { };

  void Clear(void);

  void IncrementTime(LBRealType walltime, LBRealType cputime);
  void IncrementGPUTime(LBRealType walltime);

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

  inline LDOMHandle &parentOM() { return data.handle.omhandle; }
  inline const LDObjHandle &GetLDObjHandle() const { return data.handle; }
  inline void SetMigratable(bool mig) { data.migratable = mig; }
  inline void setPupSize(size_t obj_pup_size) {
    data.pupSize = pup_encodeSize(obj_pup_size);
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
  double gpuTime;
#if CMK_LB_CPUTIMER
  double startCTime;
  LBRealType lastCpuTime;
#endif
//  bool migratable;   // temp
};

#endif

/*@}*/
