/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBOBJ_H
#define LBOBJ_H

#include "lbdb.h"

class LBDB;

class LBObj
{
friend class LBDB;

public:
  LBObj(LBDB *_parentDB, const LDObjHandle &_h, void *usr_ptr = NULL, CmiBool _migratable=CmiTrue, CmiBool _asyncArrival = CmiFalse) {
    data.handle = _h;
    data.migratable = _migratable;
    data.asyncArrival = _asyncArrival;
    Clear();
//    data.cpuTime = 0.;
//    data.wallTime = 0.;
//    data.minWall = 1e6;
//    data.maxWall = 0.;
    userData = usr_ptr;
    parentDB = _parentDB;
//    migratable = _migratable;
//    registered = CmiTrue;
    startWTime = -1.0;
    lastWallTime = .0;
#if CMK_LB_CPUTIMER
    startCTime = -1.0;
    lastCpuTime = .0;
#endif
  }

  ~LBObj() { };

#if 0
  LBObj(LBDB *_parentDB, LDOMHandle _omhandle, LDObjid _id,
	void *_userData = 0, CmiBool _migratable=CmiTrue) {
    parentDB = _parentDB;
//    parentOM = _omhandle;
//    myhandle.id = _id;
//    userData = _userData;
    migratable = _migratable;
    registered = CmiFalse;
  };

  void DepositHandle(const LDObjHandle &_h) {
//    CkAssert(_h.id == myhandle.id);
//    myhandle = _h;
    data.handle = _h;
//    data.omHandle = _h.omhandle;
    data.migratable = migratable;
    data.cpuTime = 0.;
    data.wallTime = 0.;
    registered = CmiTrue;
  };
#endif

  void Clear(void);

  void IncrementTime(double walltime, double cputime);
  inline void StartTimer(void) {
	startWTime = CkWallTimer();
#if CMK_LB_CPUTIMER
	startCTime = CkCpuTimer();
#endif
  }
  inline void StopTimer(double* walltime, double* cputime) {
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

  inline void getTime(double *w, double *c) {
    *w = data.wallTime;
#if CMK_LB_CPUTIMER
    *c = data.cpuTime;
#else
    *c = *w;
#endif
  }

  inline void setTiming(double cputime)
  {
    data.wallTime = cputime;
#if CMK_LB_CPUTIMER
    data.cpuTime = cputime;
#endif
  }

  inline LDOMHandle &parentOM() { return data.handle.omhandle; }
  inline const LDObjHandle &GetLDObjHandle() const { return data.handle; }
  inline void SetMigratable(CmiBool mig) { data.migratable = mig; }
  inline void UseAsyncMigrate(CmiBool async) { data.asyncArrival = async; }
  inline LDObjData &ObjData() { return data; };
  inline void lastKnownLoad(double *w, double *c) {
    *w = lastWallTime;
#if CMK_LB_CPUTIMER
    *c = lastCpuTime;
#else
    *c = *w;
#endif
  }
  inline void *getUserData() { return  userData; }
private:

  LBDB* parentDB;
  void *userData;
//  LDOMHandle parentOM;
//  LDObjHandle myhandle;
  LDObjData data;
//  CmiBool registered;
  double startWTime;
  double lastWallTime;
#if CMK_LB_CPUTIMER
  double startCTime;
  double lastCpuTime;
#endif
//  CmiBool migratable;   // temp
};

#endif

/*@}*/
