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
  //normal object construction
  HeapObject exampleObject(20, false);

  //normal chare array construction
  CProxy_SimpleArray simpleProxy = CProxy_SimpleArray::ckNew(30);

  //pass object to remote method invocation on the chare array
  simpleProxy[29].acceptData(exampleObject);
}

#include "SimplePUP.def.h"
