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
  LBObj(LBDB *_parentDB, LDOMHandle _omhandle, LDObjid _id,
	void *_userData = 0, CmiBool _migratable=CmiTrue) {
    parentDB = _parentDB;
    parentOM = _omhandle;
    myhandle.id = _id;
    userData = _userData;
    migratable = _migratable;
    registered = CmiFalse;
  };

  ~LBObj() { };

  void DepositHandle(const LDObjHandle &_h) {
    CkAssert(_h.id == myhandle.id);
    myhandle = _h;
    data.handle = myhandle;
    data.omHandle = _h.omhandle;
    data.migratable = migratable;
    data.cpuTime = 0.;
    data.wallTime = 0.;
    registered = CmiTrue;
  };

  void Clear(void);

  void IncrementTime(double walltime, double cputime);
  void StartTimer(void);
  void StopTimer(double* walltime, double* cputime);

  inline const LDObjHandle &GetLDObjHandle() const { return myhandle; }
private:
  inline LDObjData ObjData() { return data; };

  LBDB* parentDB;
  LDOMHandle parentOM;
  LDObjHandle myhandle;
  LDObjData data;
  void *userData;
  double startWTime;
  double startCTime;
  CmiBool migratable;
  CmiBool registered;
};

#endif

/*@}*/
