//#include "PythonCCS.h"
#include "server.decl.h"

/*
class Python_Main : public PythonChare {
 protected:
  static PyMethodDef CkPy_MethodsCustom[];
  PyObject *localArgs;
  PyObject **localResult;
  CthThread localThread;
  PyThreadState *localPyThread;
 public:
  PyMethodDef *getMethods() {return CkPy_MethodsCustom;}
  void _Py_runhigh(PyObject*, PyObject**, CthThread, PyThreadState*);
};
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
  void run();
 private:
  int mynumber;
};
