/**
 * \addtogroup CkLdb
*/
/*@{*/

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

  void *getUserData() { return userData; }

private:
  LBOM() { };

  LBOM(LBDB *_parent, LDOMid _id,
       void *_userData, LDCallbacks _callbacks)  {
    parent = _parent;
    myid = _id;
    userData = _userData;
    callbacks = _callbacks;
    registering_objs = false;
  };
  ~LBOM() { }

  void DepositHandle(LDOMHandle _h) { myhandle = _h; };
  void Migrate(LDObjHandle _h, int dest) { callbacks.migrate(_h,dest); };
#if CMK_LBDB_ON
  void MetaLBResumeWaitingChares(LDObjHandle _h, int lb_ideal_period) {
    callbacks.metaLBResumeWaitingChares(_h, lb_ideal_period);
  }
  void MetaLBCallLBOnChares(LDObjHandle _h) {
    callbacks.metaLBCallLBOnChares(_h);
  }
#endif
  bool RegisteringObjs() { return registering_objs; };
  void SetRegisteringObjs(bool _set) { registering_objs = _set; };

  LBDB *parent;
  LDOMid myid;
  LDOMHandle myhandle;
  void *userData;
  LDCallbacks callbacks;
  bool registering_objs;

};

#endif

/*@}*/
