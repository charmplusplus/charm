// README: in order for this library to work, there should be a link
// from /usr/include/python to the current python include version

#include "ckcallback-ccs.h"
#include "python/Python.h"
#include "python/compile.h"
#include "python/eval.h"
#include "python/node.h"
#include "PythonCCS.decl.h"
#include "string"
#include "map"

class PythonMain : public Chare {
 public:
  PythonMain(CkArgMsg *msg);
};

class PythonObject {
 public:
  void execute(CkCcsRequestMsg *msg);
  void iterate(CkCcsRequestMsg *msg);

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

  // the following methods should be overwritten by the user if used

  // methods for accessing charm varibles from inside python
  // read: input an object which describes where the data should be read from, return a PyObject with the given data
  // write: input two object describing where the data should be written, and the actual data
  virtual PyObject* read(PyObject* where) {CkAbort("PythonCCS: Method read should be reimplemented");};
  virtual void write(PyObject* where, PyObject* what) {CkAbort("PythonCCS: Method write should be reimplemented");};

  // methods to create iterators for iterative python invocations
  // buildIterator: input a PyObject, which is an empty class to be filled with data, and a void pointer describing over what to iterate (user defined format). Should return 1, if it returns 0 no computation is done
  // nextIteratorUpdate: input a PyObject to be filled with the next iterator, this contains the previous iterator, so if the python code modified the object, here the new information can be found. A Python object with the result returned by the python code, and the description of the iterator (as in buildIterator). Return 1 if there is a new object, 0 if there are no more objects in the iterator.
  virtual int buildIterator(PyObject*, void*) {CkAbort("PythonCCS: Method buildIterator should be reimplemented");};
  virtual int nextIteratorUpdate(PyObject*, PyObject*, void*) {CkAbort("PythonCCS: Method nextIteratorUpdate should be reimplemented");};

  //virtual void registerPython();
};

typedef std::map<int,PythonObject*>  PythonTable;

class PythonChare : public Chare, public PythonObject {
 public:
  PythonChare() {}
  PythonChare(CkMigrateMessage *msg) {}
};

class PythonGroup : public Group, public PythonObject {
 public:
  PythonGroup() {}
  PythonGroup(CkMigrateMessage *msg) {}
};

class PythonNodeGroup : public NodeGroup, public PythonObject {
 public:
  PythonNodeGroup() {}
  PythonNodeGroup(CkMigrateMessage *msg) {}
};

class PythonArray1D : public CBase_PythonArray1D, public PythonObject {
 public:
  PythonArray1D() {}
  PythonArray1D(CkMigrateMessage *msg) {}
};

class PythonArray2D : public CBase_PythonArray2D, public PythonObject {
 public:
  PythonArray2D() {}
  PythonArray2D(CkMigrateMessage *msg) {}
};

class PythonArray3D : public CBase_PythonArray3D, public PythonObject {
 public:
  PythonArray3D() {}
  PythonArray3D(CkMigrateMessage *msg) {}
};
