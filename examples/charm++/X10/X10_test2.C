//-------------------------------------------------------------
// file   : X10_test2.C
// author : Isaac Dooley
// date   : April 2006
//

#include "X10_lib.h"

// Statements which can be executed by the application
// These eventually will be created by a compiler

void asnycHandler(int whichStatement){
  printf("Faking execution of statement %d\n", whichStatement);
}

// Returning of data will probably be done via a serialized message. I would like to know
// how the compiler folks would like to handle this.
void futureHandler(int whichStatement){
  printf("Faking execution of future %d\n", whichStatement);
}




void mainThread(){
  
  FinishHandle f = beginFinish();
  asyncCall(f,1,100,NULL); // An X11 async method with place, whichFunction, and packed variables  
  printf("do some of my work here\n");;  // Do local work
  asyncCall(f,2,101,NULL);
  asyncCall(f,3,102,NULL);
  asyncCall(f,2,103,NULL);
  endFinish(f);	// 	Wait on async to finish

  FutureHandle fut = futureCall(3, 2, NULL);  
  futureForce(fut); // currently the return value is ignored, but once we choose the 
                    // format for returning the data, it will be simple to add.
  
  printf("Done!\n");

}

