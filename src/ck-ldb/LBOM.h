#ifndef LBOM_H
#define LBOM_H

#include "lbdb.h"
#include "LBObj.h"

class LBDB;

class LBOM
{
friend class LBDB;

public:
  LDOMid id() { return myid; };

private:
  LBOM() { };

  LBOM(LBDB *_parent, LDOMid _id,
       void *_userData, LDCallbacks _callbacks)  {
    parent = _parent;
    myid = _id;
    userData = _userData;
    callbacks = _callbacks;
    registering_objs = False;
  };
  ~LBOM() { }

  void DepositHandle(LDOMHandle _h) { myhandle = _h; };
  void Migrate(LDObjHandle _h, int dest) { callbacks.migrate(_h,dest); };
  Bool RegisteringObjs() { return registering_objs; };
  void SetRegisteringObjs(Bool _set) { registering_objs = _set; };

  LBDB *parent;
  LDOMid myid;
  LDOMHandle myhandle;
  void *userData;
  LDCallbacks callbacks;
  Bool registering_objs;

};

#endif
