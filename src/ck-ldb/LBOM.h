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
    registering_objs = CmiFalse;
  };
  ~LBOM() { }

  void DepositHandle(LDOMHandle _h) { myhandle = _h; };
  void Migrate(LDObjHandle _h, int dest) { callbacks.migrate(_h,dest); };
  int DimInfo(const LDObjHandle &_h, int dim) { callbacks.dimInfo(_h,dim); };
  CmiBool RegisteringObjs() { return registering_objs; };
  void SetRegisteringObjs(CmiBool _set) { registering_objs = _set; };

  LBDB *parent;
  LDOMid myid;
  LDOMHandle myhandle;
  void *userData;
  LDCallbacks callbacks;
  CmiBool registering_objs;

};

#endif

/*@}*/
