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
  LBObj(LBDB *_parentDB, const LDObjHandle &_h, CmiBool _migratable=CmiTrue) {
    data.handle = _h;
    data.migratable = _migratable;
    data.cpuTime = 0.;
    data.wallTime = 0.;
    parentDB = _parentDB;
    migratable = _migratable;
    registered = CmiTrue;
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
private:
  inline LDObjData ObjData() { return data; };

  LBDB* parentDB;
//  LDOMHandle parentOM;
//  LDObjHandle myhandle;
  LDObjData data;
//  void *userData;
  double startWTime;
  double startCTime;
  CmiBool migratable;   // temp
  CmiBool registered;
};

#endif

/*@}*/
