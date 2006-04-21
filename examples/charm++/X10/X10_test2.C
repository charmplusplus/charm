//-------------------------------------------------------------
// file   : X10_test2.C
// author : Isaac Dooley
// date   : April 2006
//

#include "X10_lib.h"

// Statements which can be executed by the application
// These eventually will be created by a compiler
void foo(){
  printf("foo\n");
}


void mainThread(){
  
  FinishHandle f = beginFinish();
  
  CkAssert(f);
  asyncCall(f,1,1,NULL); // An X11 async method with place, whichFunction, and packed variables
  foo();  // Do local work
  
  endFinish(f);	// 	Wait on async to finish
}

