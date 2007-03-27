#include "server.h"

/* global readonly variable */
CProxy_Main mainProxy;
CProxy_MyArray mypython;

Main::Main (CkArgMsg *msg) {

  /* Load all data needed by the program and then create the interactor for python,
     it can also be created as Array or Nodegroup with the consequent differences
     for the python code been inserted */
  elem = 10;

  mypython = CProxy_MyArray::ckNew(elem, 2);

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
  if (pythonHandle1) pythonHandle2=i;
  else pythonHandle1=i;
  pythonSleep(i);
  mypython.run();
}

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

MyArray::MyArray (int a) {mynumber = thisIndex+1000;}

MyArray::MyArray (CkMigrateMessage *msg) {}

void MyArray::run() {
  CkPrintf("[%d] in run %d\n",thisIndex,mynumber);
  sleep(0);
  mainProxy.arrayResult(mynumber++);
}

#include "server.def.h"
