// README: in order for this library to work, there should be a link
// from /usr/include/python to the current python include version

#ifndef __CKPYTHON_H
#define __CKPYTHON_H

#include "ckcallback-ccs.h"
#include "python/Python.h"
#include "python/compile.h"
#include "python/eval.h"
#include "python/node.h"
#include "PythonCCS.decl.h"
#include "PythonCCS-client.h"
#include "string"
#include "map"

#define PYTHON_ENABLE_HIGH  private: \
  static PyMethodDef CkPy_MethodsCustom[]; \
 public: \
  PyMethodDef *getMethods() {return CkPy_MethodsCustom;}

extern PyMethodDef CkPy_MethodsDefault[];

class PythonObject {
  static PyMethodDef CkPy_MethodsCustom[];
 public:
  void execute(CkCcsRequestMsg *msg);
  void cleanup(PythonExecute *pyMsg, PyThreadState *pts, CmiUInt4 pyVal);
  void getPrint(CkCcsRequestMsg *msg);
  static void _callthr_executeThread(CkThrCallArg *impl_arg);
  void executeThread(CkCcsRequestMsg *msg);
  virtual PyMethodDef *getMethods() {return CkPy_MethodsCustom;}

  // utility functions to manipulate python objects
  // in order to use Dictionaries, Lists, Tuples, refer to
  // "Python/C API Reference Manual" (http://docs.python.org/)
  void pythonSetString(PyObject*, char*, char*);
  void pythonSetString(PyObject*, char*, char*, int);
  void pythonSetInt(PyObject*, char*, long);
  void pythonSetLong(PyObject*, char*, long);
  void pythonSetLong(PyObject*, char*, unsigned long);
  void pythonSetLong(PyObject*, char*, double);
  void pythonSetFloat(PyObject*, char*, double);
  void pythonSetComplex(PyObject*, char*, double, double);

  void pythonGetString(PyObject*, char*, char**);
  void pythonGetInt(PyObject*, char*, long*);
  void pythonGetLong(PyObject*, char*, long*);
  void pythonGetLong(PyObject*, char*, unsigned long*);
  void pythonGetLong(PyObject*, char*, double*);
  void pythonGetFloat(PyObject*, char*, double*);
  void pythonGetComplex(PyObject*, char*, double*, double*);

  // utility functions to deal with threads
  PyObject *pythonGetArg(int);
  void pythonPrepareReturn(int);
  void pythonReturn(int);
  void pythonReturn(int, PyObject*);
  void pythonAwake(int);
  void pythonSleep(int);

  // the following methods should be overwritten by the user if used

  // methods for accessing charm varibles from inside python
  // read: input an object which describes where the data should be read from, return a PyObject with the given data
  // write: input two object describing where the data should be written, and the actual data
  virtual PyObject* read(PyObject* where) {CkAbort("PythonCCS: Method read should be reimplemented"); return NULL; };
  virtual void write(PyObject* where, PyObject* what) {CkAbort("PythonCCS: Method write should be reimplemented");};

  // methods to create iterators for iterative python invocations
  // buildIterator: input a PyObject, which is an empty class to be filled with data, and a void pointer describing over what to iterate (user defined format). Should return 1, if it returns 0 no computation is done
  // nextIteratorUpdate: input a PyObject to be filled with the next iterator, this contains the previous iterator, so if the python code modified the object, here the new information can be found. A Python object with the result returned by the python code, and the description of the iterator (as in buildIterator). Return 1 if there is a new object, 0 if there are no more objects in the iterator.
  virtual int buildIterator(PyObject*, void*) {CkAbort("PythonCCS: Method buildIterator should be reimplemented"); return 0; };
  virtual int nextIteratorUpdate(PyObject*, PyObject*, void*) {CkAbort("PythonCCS: Method nextIteratorUpdate should be reimplemented"); return 0; };

};

typedef struct {
  PythonObject *object; /* The c++ object running the job */
  bool inUse;
  PyObject *arg;
  PyObject **result;
  CthThread thread;       /* The charm thread running the python code */
  PyThreadState *pythread; /* The python interpreter interpreting the code */
  int clientReady; /* whether or not a client has sent a request for print */
  /* meanings of clientReady:
     1  - there is a client wainting for data
     0  - no client waiting for data
     -1 - no client watiing, and the current structure is alive only because of
     "KeepPrint". it has to be deleted when a client synchonizes
  */
  CcsDelayedReply client; /* Where to send the printed data when ready */
  std::string printed; /* Union of all printed string and not yet shipped to the client */
} PythonStruct;
typedef std::map<CmiUInt4,PythonStruct> PythonTable;

CsvExtern(CmiNodeLock, pyLock);
CsvExtern(PythonTable *, pyWorkers);
CsvExtern(CmiUInt4, pyNumber);
CtvExtern(PyObject *, pythonReturnValue);

#endif //__CKPYTHON_H
