#include "PythonCCS.h"

//CProxy_PythonGroup python;

PythonMain::PythonMain (CkArgMsg *msg) {
  //CProxy_PythonGroup python = CProxy_PythonGroup::ckNew();

  //CcsRegisterHandler("pyCode", CkCallback(CkIndex_PythonGroup::execute(0),python));
}

CsvStaticDeclare(CmiNodeLock, pyLock);
CsvStaticDeclare(PythonTable *, pyWorkers);
CsvStaticDeclare(int, pyNumber);
/*
CkpvStaticDeclare(PythonChare *, curWorkerChare);
CkpvStaticDeclare(PythonGroup *, curWorkerGroup);
CkpvStaticDeclare(PythonNodeGroup *, curWorkerNodeGroup);
CkpvStaticDeclare(PythonArray1D *, curWorkerArray1D);
CkpvStaticDeclare(PythonArray2D *, curWorkerArray2D);
*/

// One-time per-processor setup routine
// main interface for python to access common charm methods
static PyObject *CkPy_printstr(PyObject *self, PyObject *args) {
  char *stringToPrint;
  if (!PyArg_ParseTuple(args, "s:printstr", &stringToPrint))
    return NULL;
  CkPrintf("%s\n",stringToPrint);
  Py_INCREF(Py_None);return Py_None; //Return-nothing idiom
}

static PyObject *CkPy_mype(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, ":mype")) return NULL;
  return Py_BuildValue("i", CkMyPe());
}

static PyObject *CkPy_numpes(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, ":numpes")) return NULL;
  return Py_BuildValue("i", CkNumPes());
}

static PyObject *CkPy_myindex(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, ":myindex")) return NULL;
  PythonArray1D *pyArray;
  if ((pyArray = (dynamic_cast<PythonArray1D*>((*CsvAccess(pyWorkers))[0])))) return Py_BuildValue("i", pyArray->thisIndex);
  else { Py_INCREF(Py_None);return Py_None;}
  //return Py_BuildValue("i", (*CsvAccess(pyWorkers))[0]->thisIndex);
}

// method to read a variable and convert it to a python object
static PyObject *CkPy_read(PyObject *self, PyObject *args) {
  char * str;
  if (!PyArg_ParseTuple(args, "s:read", str)) return NULL;
  std::string cstr = str;
  PythonObject *pyWorker = (*CsvAccess(pyWorkers))[0];
  TypedValue result = pyWorker->read(cstr);
  switch (result.type) {
  case PY_INT:
    return Py_BuildValue("i", result.value.i);

  case PY_LONG:
    return Py_BuildValue("l", result.value.l);
  case PY_FLOAT:
    return Py_BuildValue("f", result.value.f);
  case PY_DOUBLE:
    return Py_BuildValue("d", result.value.d);
  }
}

// method to convert a python object into a variable and write it
static PyObject *CkPy_write(PyObject *self, PyObject *args) {
  char * str;
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "sO:write", &str, &obj)) return NULL;
  std::string cstr = str;
  Py_types varType = (*CsvAccess(pyWorkers))[0]->getType(cstr);
  (*CsvAccess(pyWorkers))[0]->write(cstr, TypedValue(varType, obj));
  Py_INCREF(Py_None);return Py_None;
}

static PyMethodDef CkPy_MethodsDefault[] = {
  {"printstr", CkPy_printstr , METH_VARARGS},
  {"mype", CkPy_mype, METH_VARARGS},
  {"numpes", CkPy_numpes, METH_VARARGS},
  {"myindex", CkPy_myindex, METH_VARARGS},
  {"read", CkPy_read, METH_VARARGS},
  {"write", CkPy_write, METH_VARARGS},
  {NULL,      NULL}        /* Sentinel */
};

void PythonObject::execute (CkCcsRequestMsg *msg) {
  CkPrintf("executing script\n");
  CmiLock(CsvAccess(pyLock));
  //CsvAccess(pyWorkers)[++CsvAccess(pyNumber)] = this;
  (*CsvAccess(pyWorkers))[CsvAccess(pyNumber)] = this;
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  Py_EndInterpreter(pts);
  CmiUnlock(CsvAccess(pyLock));
}
/*
void PythonChare::execute (CkCcsRequestMsg *msg) {
  CkPrintf("executing chare script\n");
  CsvAccess(curWorkerChare)=this;
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  Py_EndInterpreter(pts);
}

void PythonGroup::execute (CkCcsRequestMsg *msg) {
  CkPrintf("executing group script\n");
  CsvAccess(curWorkerGroup)=this;
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  Py_EndInterpreter(pts);
}

void PythonNodeGroup::execute (CkCcsRequestMsg *msg) {
  CkPrintf("executing node group script\n");
  CsvAccess(curWorkerNodeGroup)=this;
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  Py_EndInterpreter(pts);
}

void PythonArray1D::execute (CkCcsRequestMsg *msg) {
  CkPrintf("executing array 1D script\n");
  CsvAccess(curWorkerArray1D)=this;
  //PyThreadState *pts = Py_NewInterpreter();
  //Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  //Py_EndInterpreter(pts);
}

void PythonArray2D::execute (CkCcsRequestMsg *msg) {
  CkPrintf("executing array2D script\n");
  CsvAccess(curWorkerArray2D)=this;
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  Py_EndInterpreter(pts);
}
*/

static void initializePythonDefault(void) {
  CsvInitialize(int, pyNumber);
  CsvAccess(pyNumber) = 0;
  CsvInitialize(PythonTable *,pyWorkers);
  CsvAccess(pyWorkers) = new PythonTable();
  CsvInitialize(CmiNodeLock, pyLock);
  CsvAccess(pyLock) = CmiCreateLock();

  /*
  CkpvInitialize(PythonChare *,curWorkerChare);
  CkpvInitialize(PythonGroup *,curWorkerGroup);
  CkpvInitialize(PythonNodeGroup *,curWorkerNodeGroup);
  CkpvInitialize(PythonArray1D *,curWorkerArray1D);
  CkpvInitialize(PythonArray2D *,curWorkerArray2D);
  CkpvAccess(curWorkerChare)=NULL;
  CkpvAccess(curWorkerGroup)=NULL;
  CkpvAccess(curWorkerNodeGroup)=NULL;
  CkpvAccess(curWorkerArray1D)=NULL;
  CkpvAccess(curWorkerArray2D)=NULL;
  */

  Py_Initialize();
  PyObject *ck = Py_InitModule("ck", CkPy_MethodsDefault);
}

#include "PythonCCS.def.h"
