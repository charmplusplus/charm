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

//#define PY_INT     0
//#define PY_LONG    1
//#define PY_FLOAT   2
//#define PY_DOUBLE  3

/*
enum Py_types {PY_INT, PY_LONG, PY_FLOAT, PY_DOUBLE};
enum Py_objects {PY_CHARE, PY_GROUP, PY_NODEGROUP, PY_ARRAY1D, PY_ARRAY2D};

typedef struct TypedValue_ {
  Py_types type;
  union {
    int i;
    long l;
    float f;
    double d;
  } value;

  TypedValue_(Py_types t, int i0) {
    type = t;
    value.i = i0;
  };

  TypedValue_(Py_types t, PyObject *o) {
    int i0;
    long l0;
    float f0;
    double d0;
    type = t;
    switch (t) {
    case PY_INT:
      i0 = (int)PyInt_AsLong(o);
      value.i = i0;
      break;
    case PY_LONG:
      l0 = PyLong_AsLong(o);
      value.l = l0;
      break;
    case PY_FLOAT:
      f0 = (float)PyFloat_AsDouble(o);
      value.f = f0;
      break;
    case PY_DOUBLE:
      d0 = PyFloat_AsDouble(o);
      value.d = d0;
      break;
    }
  }
} TypedValue;
*/

class PythonObject {
 public:
  void execute(CkCcsRequestMsg *msg);
  void iterate(CkCcsRequestMsg *msg);

  // the following methods should be overwritten by the user if used

  // methods for accessing charm varibles from inside python
  virtual PyObject* read(PyObject*) {CkAbort("PythonCCS: Method read should be reimplemented");};
  virtual void write(PyObject*, PyObject*) {CkAbort("PythonCCS: Method write should be reimplemented");};

  // methods to create iterators for iterative python invocations
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
