////////////////////////////////////
//
//  hello.C
//
//  Definition of chares in hello
//
//  Author: Michael Lang
//  Date: 6/15/99
//
////////////////////////////////////

#include "hello.h"

main::main(CkArgMsg *m)
{
  CkPrintf("Hello World \n");
  delete m;
  CkExit();
}

#include "hello.def.h"
