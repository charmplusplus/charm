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
  LBObj(LBDB *_parentDB, const LDObjHandle &_h, void *usr_ptr = NULL, CmiBool _migratable=CmiTrue) {
    data.handle = _h;
    data.migratable = _migratable;
    data.cpuTime = 0.;
    data.wallTime = 0.;
    userData = usr_ptr;
    parentDB = _parentDB;
    migratable = _migratable;
    registered = CmiTrue;
    lastCpuTime = lastWallTime = .0;
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
  void StartTimer(void);
  void StopTimer(double* walltime, double* cputime);

  inline LDOMHandle &parentOM() { return data.handle.omhandle; }
  inline const LDObjHandle &GetLDObjHandle() const { return data.handle; }
  inline void SetMigratable(int mig) { data.migratable = mig; }
  inline LDObjData &ObjData() { return data; };
  inline void lastKnownLoad(double *c, double *w) {*c=lastCpuTime; *w=lastWallTime; }
  inline void *getUserData() { return  userData; }
private:

  LBDB* parentDB;
//  LDOMHandle parentOM;
//  LDObjHandle myhandle;
  LDObjData data;
  void *userData;
  double startWTime;
  double startCTime;
  double lastCpuTime;
  double lastWallTime;
  CmiBool migratable;   // temp
  CmiBool registered;
};

#endif

/*@}*/
