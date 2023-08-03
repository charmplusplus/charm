#include "server.decl.h"

/*
  This test will generate a method called "runhigh" available to python
  programs. This will be declared [python] in che .ci file, and available
  through the module "charm" from any python script sent to Main.
*/

class Main : public CBase_Main {
 public:
  Main(CkArgMsg *msg);
  void exit();
  void arrayResult(int mycontrib);
  void ccs_kill (CkCcsRequestMsg *msg);
  void runhigh(int);
  PyObject *returnValue;
 private:
  int count;
  int total;
  int elem;
  int pythonHandle1;
  int pythonHandle2;
};

class MyArray : public CBase_MyArray {
 public:
  MyArray();
  MyArray(CkMigrateMessage *msg);
  void run(const CkCallback &cb);
 private:
  int mynumber;
};
