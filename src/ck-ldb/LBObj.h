#ifndef LBOBJ_H
#define LBOBJ_H

#include "lbdb.h"

class LBObj
{
friend class LBDB;

public:
  LBObj(LBDB* _parentDB, LDOMHandle _omhandle, LDObjid _id,
	void *_userData = 0, CmiBool _migratable=CmiTrue) {
    parentDB = _parentDB;
    parentOM = _omhandle;
    myid = _id;
    userData = _userData;
    migratable = _migratable;
    registered = CmiFalse;
  };

  ~LBObj() { };

  void DepositHandle(LDObjHandle _h) {
    myhandle = _h;
    data.handle = myhandle;
    data.id = myid;
    data.omHandle = _h.omhandle;
    data.omID = _h.omhandle.id;
    data.migratable = migratable;
    data.cpuTime = 0.;
    data.wallTime = 0.;
    registered = CmiTrue;
  };

  void Clear(void);
  void IncrementTime(double walltime, double cputime);
  void StartTimer(void);
  void StopTimer(double* walltime, double* cputime);

private:
  LDObjData ObjData() { return data; };

  LBDB* parentDB;
  LDOMHandle parentOM;
  LDObjHandle myhandle;
  LDObjid myid;
  void *userData;
  CmiBool migratable;
  LDObjData data;
  double startWTime;
  double startCTime;
  CmiBool registered;
};

#endif
