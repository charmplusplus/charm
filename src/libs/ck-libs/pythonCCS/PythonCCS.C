#include "PythonCCS.h"

//CProxy_PythonGroup python;

PythonMain::PythonMain (CkArgMsg *msg) {
  //CProxy_PythonGroup python = CProxy_PythonGroup::ckNew();

  //CcsRegisterHandler("pyCode", CkCallback(CkIndex_PythonGroup::execute(0),python));
}

CsvStaticDeclare(CmiNodeLock, pyLock);
CsvStaticDeclare(PythonTable *, pyWorkers);
CsvStaticDeclare(int, pyNumber);

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
  CmiLock(CsvAccess(pyLock));
  PythonArray1D *pyArray = dynamic_cast<PythonArray1D*>((*CsvAccess(pyWorkers))[0]);
  CmiUnlock(CsvAccess(pyLock));
  if ((pyArray = (dynamic_cast<PythonArray1D*>((*CsvAccess(pyWorkers))[0])))) return Py_BuildValue("i", pyArray->thisIndex);
  else { Py_INCREF(Py_None);return Py_None;}
  //return Py_BuildValue("i", (*CsvAccess(pyWorkers))[0]->thisIndex);
}

// method to read a variable and convert it to a python object
static PyObject *CkPy_read(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "O:read")) return NULL;
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = (*CsvAccess(pyWorkers))[0];
  CmiUnlock(CsvAccess(pyLock));
  return pyWorker->read(args);
}

// method to convert a python object into a variable and write it
static PyObject *CkPy_write(PyObject *self, PyObject *args) {
  PyObject *where, *what;
  if (!PyArg_ParseTuple(args, "OO:write",&where,&what)) return NULL;
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = (*CsvAccess(pyWorkers))[0];
  CmiUnlock(CsvAccess(pyLock));
  pyWorker->write(where, what);
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
  CmiUnlock(CsvAccess(pyLock));
  PyEval_AcquireLock();
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);
  PyRun_SimpleString((char *)msg->data);
  Py_EndInterpreter(pts);
  PyEval_ReleaseLock();
}

static void initializePythonDefault(void) {
  CsvInitialize(int, pyNumber);
  CsvAccess(pyNumber) = 0;
  CsvInitialize(PythonTable *,pyWorkers);
  CsvAccess(pyWorkers) = new PythonTable();
  CsvInitialize(CmiNodeLock, pyLock);
  CsvAccess(pyLock) = CmiCreateLock();

  Py_Initialize();
  PyEval_InitThreads();
  PyEval_ReleaseLock();
  PyObject *ck = Py_InitModule("ck", CkPy_MethodsDefault);
}

#include "PythonCCS.def.h"
