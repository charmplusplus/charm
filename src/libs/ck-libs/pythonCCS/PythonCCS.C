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
  int pyNumber = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonArray1D *pyArray = dynamic_cast<PythonArray1D*>((*CsvAccess(pyWorkers))[pyNumber]);
  CmiUnlock(CsvAccess(pyLock));
  if (pyArray) return Py_BuildValue("i", pyArray->thisIndex);
  else { Py_INCREF(Py_None);return Py_None;}
  //return Py_BuildValue("i", (*CsvAccess(pyWorkers))[0]->thisIndex);
}

// method to read a variable and convert it to a python object
static PyObject *CkPy_read(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "O:read")) return NULL;
  int pyNumber = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = (*CsvAccess(pyWorkers))[pyNumber];
  CmiUnlock(CsvAccess(pyLock));
  return pyWorker->read(args);
}

// method to convert a python object into a variable and write it
static PyObject *CkPy_write(PyObject *self, PyObject *args) {
  PyObject *where, *what;
  if (!PyArg_ParseTuple(args, "OO:write",&where,&what)) return NULL;
  int pyNumber = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = (*CsvAccess(pyWorkers))[pyNumber];
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
  char pyString[25];

  // update the reference number, used to access the current chare
  CmiLock(CsvAccess(pyLock));
  int pyReference = CsvAccess(pyNumber)++;
  CsvAccess(pyNumber) &= ~(1<<31);
  (*CsvAccess(pyWorkers))[pyReference] = this;
  //printf("map number:%d\n",CsvAccess(pyWorkers)->size());
  CmiUnlock(CsvAccess(pyLock));
  sprintf(pyString, "charmNumber=%d", pyReference);

  // create the new interpreter
  PyEval_AcquireLock();
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);

  // insert into the dictionary a variable with the reference number
  PyObject *mod = PyImport_AddModule("__main__");
  PyObject *dict = PyModule_GetDict(mod);
  PyRun_String(pyString,Py_file_input,dict,dict);

  // run the program
  PyRun_SimpleString((char *)msg->data);

  // distroy map element in pyWorkers and terminate interpreter
  Py_EndInterpreter(pts);
  PyEval_ReleaseLock();
  CmiLock(CsvAccess(pyLock));
  CsvAccess(pyWorkers)->erase(pyReference);
  CmiUnlock(CsvAccess(pyLock));
}

void PythonObject::iterate (CkCcsRequestMsg *msg) {
  char pyString[25];

  // update the reference number, used to access the current chare
  CmiLock(CsvAccess(pyLock));
  int pyReference = CsvAccess(pyNumber)++;
  CsvAccess(pyNumber) &= ~(1<<31);
  (*CsvAccess(pyWorkers))[pyReference] = this;
  //printf("map number:%d\n",CsvAccess(pyWorkers)->size());
  CmiUnlock(CsvAccess(pyLock));
  sprintf(pyString, "charmNumber=%d", pyReference);

  // create the new interpreter
  PyEval_AcquireLock();
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);

  // insert into the dictionary a variable with the reference number
  PyObject *mod = PyImport_AddModule("__main__");
  PyObject *dict = PyModule_GetDict(mod);
  PyRun_String(pyString,Py_file_input,dict,dict);

  // compile the program
  char *userCode = (char *)msg->data;
  struct _node* programNode = PyParser_SimpleParseString(userCode, Py_file_input);
  if (programNode==NULL) { CkPrintf("Program error\n"); return; }
  PyCodeObject *program = PyNode_Compile(programNode, "");
  if (program==NULL) { CkPrintf("Program error\n"); return; }
  PyObject *code = PyEval_EvalCode(program, dict, dict);
  if (code==NULL) { CkPrintf("Program error\n"); return; }

  // load the user defined method
  char *userMethod = userCode + strlen(userCode) + 1;
  PyObject *item = PyDict_GetItemString(dict, userMethod);
  if (item==NULL) { CkPrintf("Method not found\n"); return; }

  // create the container for the data
  PyRun_String("class Particle:\n\tpass\n\n", Py_file_input, dict, dict);
  PyObject *part = PyRun_String("Particle()", Py_eval_input, dict, dict);
  PyObject *arg = PyTuple_New(1);
  PyTuple_SetItem(arg, 0, part);

  // construct the iterator calling the user defined method in the interiting class
  void *userIterator = (void *)(userMethod + strlen(userMethod) + 1);
  int more = buildIterator(part, userIterator);

  // iterate over all the provided iterators from the user class
  PyObject *result;
  while (more) {
    result = PyObject_CallObject(item, arg);
    more = nextIteratorUpdate(part, result, userIterator);
  }

  // distroy map element in pyWorkers and terminate interpreter
  Py_EndInterpreter(pts);
  PyEval_ReleaseLock();
  CmiLock(CsvAccess(pyLock));
  CsvAccess(pyWorkers)->erase(pyReference);
  CmiUnlock(CsvAccess(pyLock));
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
  PyObject *ck = Py_InitModule("ck", CkPy_MethodsDefault);

  PyEval_ReleaseLock();
}

#include "PythonCCS.def.h"
