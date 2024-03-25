#include "server.h"

/* global readonly variable */
CProxy_Main mainProxy;
CProxy_MyArray mypython;
double granularity;

Main::Main (CkArgMsg *msg) {
  char **argv = msg->argv;
  int argc = msg->argc;
  /* Load all data needed by the program and then create the interactor for python,
     it can also be created as Array or Nodegroup with the consequent differences
     for the python code been inserted */
  elem = 10;
  granularity = 1e-5;
  if(argc > 1){
    elem = atoi(argv[1]);
  }
  if(argc > 2){
    granularity = atof(argv[2]);
  }

  mypython = CProxy_MyArray::ckNew(elem);


  count=0;
  total=0;
  mainProxy = thishandle;
  pythonHandle1=0;
  pythonHandle2=0;

  // register handler for callback
  //CcsRegisterHandler("pyCode", CkCallback(CkIndex_Main::pyRequest(0),thishandle));
  mainProxy.registerPython("pyCode");
  CcsRegisterHandler("kill", CkCallback(CkIndex_Main::exit(),thishandle));

}

void Main::exit () {
  CkExit();
}

void Main::ccs_kill (CkCcsRequestMsg *msg) {
  CkExit();
}

void Main::runhigh(int i) {
  void *total;
  mypython.run(CkCallbackPython(total));
  CkReductionMsg *msg = (CkReductionMsg*)total;
  int *value = (int*)msg->getData();
  pythonReturn(i, Py_BuildValue("i",*value));
}

/*
void Main::arrayResult (int value) {
  total += value;
  if (++count == elem) {
    count = 0;
    int *pythonHandle;
    if (pythonHandle1) pythonHandle=&pythonHandle1;
    else pythonHandle=&pythonHandle2;
    pythonAwake(*pythonHandle);
    pythonReturn(*pythonHandle,Py_BuildValue("i",total));
    *pythonHandle = 0;
    total=0;
  }
}
*/

MyArray::MyArray () {mynumber = thisIndex+1000;}

MyArray::MyArray (CkMigrateMessage *msg) {}

void MyArray::run(const CkCallback &cb) {
//  CkPrintf("[%d] in run %d\n",thisIndex,mynumber);
   double _startTime = CmiWallTimer();

   while(CmiWallTimer() - _startTime < granularity){
   	mynumber++;
   }
//  sleep(0);
  //mainProxy.arrayResult(mynumber++);
  mynumber++;
  contribute(sizeof(int), &mynumber, CkReduction::sum_int, cb);
}

#include "server.def.h"
