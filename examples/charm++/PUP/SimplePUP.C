/////////////////////////////////////
//
//  SimplePUP.C
//
//  Definition of chares in SimplePUP
//
//  Author: Eric Bohm
//  Date: 2012/7/30
//
/////////////////////////////////////

#include "SimplePUP.h"

main::main(CkArgMsg *m)
{
  
  CProxy_SimpleArray simpleProxy= CProxy_SimpleArray::ckNew(30);
  SimpleObject exampleObject(20,false);
  simpleProxy[29].acceptData(exampleObject);
}

#include "SimplePUP.def.h"
