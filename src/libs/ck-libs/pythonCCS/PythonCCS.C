#include "PythonCCS.h"

CsvDeclare(CmiNodeLock, pyLock);
CsvDeclare(PythonTable *, pyWorkers);
CsvDeclare(int, pyNumber);
CtvDeclare(PyObject *, pythonReturnValue);

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

/*
static PyObject *CkPy_myindex(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, ":myindex")) return NULL;
  int pyNumber = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  ArrayElement1D *pyArray = dynamic_cast<ArrayElement1D>((*CsvAccess(pyWorkers))[pyNumber]);
  CmiUnlock(CsvAccess(pyLock));
  if (pyArray) return Py_BuildValue("i", pyArray->thisIndex);
  else { Py_INCREF(Py_None);return Py_None;}
  //return Py_BuildValue("i", (*CsvAccess(pyWorkers))[0]->thisIndex);
}
*/

// method to read a variable and convert it to a python object
static PyObject *CkPy_read(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "O:read")) return NULL;
  int pyNumber = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = ((*CsvAccess(pyWorkers))[pyNumber]).object;
  CmiUnlock(CsvAccess(pyLock));
  return pyWorker->read(args);
}

// method to convert a python object into a variable and write it
static PyObject *CkPy_write(PyObject *self, PyObject *args) {
  PyObject *where, *what;
  if (!PyArg_ParseTuple(args, "OO:write",&where,&what)) return NULL;
  int pyNumber = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = ((*CsvAccess(pyWorkers))[pyNumber]).object;
  CmiUnlock(CsvAccess(pyLock));
  pyWorker->write(where, what);
  Py_INCREF(Py_None);return Py_None;
}

PyMethodDef PythonObject::CkPy_MethodsCustom[] = {
  {NULL,      NULL}        /* Sentinel */
};

PyMethodDef CkPy_MethodsDefault[] = {
  {"printstr", CkPy_printstr , METH_VARARGS},
  {"mype", CkPy_mype, METH_VARARGS},
  {"numpes", CkPy_numpes, METH_VARARGS},
  {"read", CkPy_read, METH_VARARGS},
  {"write", CkPy_write, METH_VARARGS},
  {NULL,      NULL}        /* Sentinel */
};

void PythonObject::execute (CkCcsRequestMsg *msg) {

  // update the reference number, used to access the current chare
  CmiLock(CsvAccess(pyLock));
  int pyReference = CsvAccess(pyNumber)++;
  CsvAccess(pyNumber) &= ~(1<<31);
  ((*CsvAccess(pyWorkers))[pyReference]).object = this;
  CmiUnlock(CsvAccess(pyLock));

  // send back this number to the client
  CcsSendDelayedReply(msg->reply, 1, (void *)&pyReference);

  // create the new interpreter
  PyEval_AcquireLock();
  PyThreadState *pts = Py_NewInterpreter();

  Py_InitModule("ck", CkPy_MethodsDefault);
  Py_InitModule("charm", getMethods());

  // insert into the dictionary a variable with the reference number
  PyObject *mod = PyImport_AddModule("__main__");
  PyObject *dict = PyModule_GetDict(mod);

  PyDict_SetItemString(dict,"charmNumber",PyInt_FromLong(pyReference));
  PyRun_String("import ck",Py_file_input,dict,dict);
  PyRun_String("import charm",Py_file_input,dict,dict);

  // run the program
  //PythonArray1D *pyArray = dynamic_cast<PythonArray1D*>(this);
  CthResume(CthCreate((CthVoidFn)_callthr_executeThread, new CkThrCallArg(msg,this), 0));
  //executeThread(msg);
  //CkIndex_PythonArray1D::_call_executeThread_CkCcsRequestMsg(msg,pyArray);
  //getMyHandle()[pyArray->thisIndex].executeThread(msg);
  //getMyHandle().executeThread(msg);
  //PyRun_SimpleString((char *)msg->data);

  // distroy map element in pyWorkers and terminate interpreter
  /*
  Py_EndInterpreter(pts);
  PyEval_ReleaseLock();

  CmiLock(CsvAccess(pyLock));
  CsvAccess(pyWorkers)->erase(pyReference);
  CmiUnlock(CsvAccess(pyLock));
  delete msg;
  */
}

void PythonObject::_callthr_executeThread(CkThrCallArg *impl_arg) {
  void *impl_msg = impl_arg->msg;
  PythonObject *impl_obj = (PythonObject *) impl_arg->obj;
  delete impl_arg;
  impl_obj->executeThread((CkCcsRequestMsg*)impl_msg);
}

void PythonObject::executeThread(CkCcsRequestMsg *msg) {
  // get the information about the running python thread and my reference number
  PyThreadState *mine = PyThreadState_Get();
  int pyReference = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));

  // store the self thread for future suspention
  CmiLock(CsvAccess(pyLock));
  ((*CsvAccess(pyWorkers))[pyReference]).thread=CthSelf();
  CmiUnlock(CsvAccess(pyLock));

  PyRun_SimpleString((char *)msg->data);

  Py_EndInterpreter(mine);
  PyEval_ReleaseLock();

  CmiLock(CsvAccess(pyLock));
  CsvAccess(pyWorkers)->erase(pyReference);
  CmiUnlock(CsvAccess(pyLock));
  delete msg;
  // since this function should always be executed in a different thread,
  // when it is done, destroy the current thread
  CthFree(CthSelf());
}

void PythonObject::iterate (CkCcsRequestMsg *msg) {

  // update the reference number, used to access the current chare
  CmiLock(CsvAccess(pyLock));
  int pyReference = CsvAccess(pyNumber)++;
  CsvAccess(pyNumber) &= ~(1<<31);
  ((*CsvAccess(pyWorkers))[pyReference]).object = this;
  CmiUnlock(CsvAccess(pyLock));

  // send back this number to the client
  CcsSendDelayedReply(msg->reply, 1, (void *)&pyReference);

  // create the new interpreter
  PyEval_AcquireLock();
  PyThreadState *pts = Py_NewInterpreter();
  Py_InitModule("ck", CkPy_MethodsDefault);

  // insert into the dictionary a variable with the reference number
  PyObject *mod = PyImport_AddModule("__main__");
  PyObject *dict = PyModule_GetDict(mod);

  PyDict_SetItemString(dict,"charmNumber",PyInt_FromLong(pyReference));

  // compile the program
  char *userCode = (char *)msg->data;
  struct _node* programNode = PyParser_SimpleParseString(userCode, Py_file_input);
  if (programNode==NULL) {
    CkPrintf("Program error\n");
    // distroy map element in pyWorkers and terminate interpreter
    Py_EndInterpreter(pts);
    PyEval_ReleaseLock();
    CmiLock(CsvAccess(pyLock));
    CsvAccess(pyWorkers)->erase(pyReference);
    CmiUnlock(CsvAccess(pyLock));
    delete msg;
    return;
  }
  PyCodeObject *program = PyNode_Compile(programNode, "");
  if (program==NULL) {
    CkPrintf("Program error\n");
    PyNode_Free(programNode);
    // distroy map element in pyWorkers and terminate interpreter
    Py_EndInterpreter(pts);
    PyEval_ReleaseLock();
    CmiLock(CsvAccess(pyLock));
    CsvAccess(pyWorkers)->erase(pyReference);
    CmiUnlock(CsvAccess(pyLock));
    delete msg;
    return;
  }
  PyObject *code = PyEval_EvalCode(program, dict, dict);
  if (code==NULL) {
    CkPrintf("Program error\n");
    PyNode_Free(programNode);
    Py_DECREF(program);
    // distroy map element in pyWorkers and terminate interpreter
    Py_EndInterpreter(pts);
    PyEval_ReleaseLock();
    CmiLock(CsvAccess(pyLock));
    CsvAccess(pyWorkers)->erase(pyReference);
    CmiUnlock(CsvAccess(pyLock));
    delete msg;
    return;
  }

  // load the user defined method
  char *userMethod = userCode + strlen(userCode) + 1;
  PyObject *item = PyDict_GetItemString(dict, userMethod);
  if (item==NULL) {
    CkPrintf("Method not found\n");
    PyNode_Free(programNode);
    Py_DECREF(program);
    Py_DECREF(code);
    // distroy map element in pyWorkers and terminate interpreter
    Py_EndInterpreter(pts);
    PyEval_ReleaseLock();
    CmiLock(CsvAccess(pyLock));
    CsvAccess(pyWorkers)->erase(pyReference);
    CmiUnlock(CsvAccess(pyLock));
    delete msg;
    return;
  }

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
    if (!result) {
      CkPrintf("Python Call error\n");
      break;
    }
    more = nextIteratorUpdate(part, result, userIterator);
    Py_DECREF(result);
  }

  Py_DECREF(part);
  Py_DECREF(arg);
  PyNode_Free(programNode);
  Py_DECREF(program);
  Py_DECREF(code);

  // distroy map element in pyWorkers and terminate interpreter
  Py_EndInterpreter(pts);
  PyEval_ReleaseLock();
  CmiLock(CsvAccess(pyLock));
  CsvAccess(pyWorkers)->erase(pyReference);
  CmiUnlock(CsvAccess(pyLock));
  delete msg;
}

static void initializePythonDefault(void) {
  CsvInitialize(int, pyNumber);
  CsvAccess(pyNumber) = 0;
  CsvInitialize(PythonTable *,pyWorkers);
  CsvAccess(pyWorkers) = new PythonTable();
  CsvInitialize(CmiNodeLock, pyLock);
  CsvAccess(pyLock) = CmiCreateLock();
  CtvInitialize(PyObject *,pythonReturnValue);

  Py_Initialize();
  PyEval_InitThreads();

  PyEval_ReleaseLock();
}

// a bunch of routines to ease the user code
void PythonObject::pythonSetString(PyObject *arg, char *descr, char *value) {
  PyObject *tmp = PyString_FromString(value);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetString(PyObject *arg, char *descr, char *value, int len) {
  PyObject *tmp = PyString_FromStringAndSize(value, len);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetString(PyObject *arg, char *descr, char **result) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *result = PyString_AsString(tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetInt(PyObject *arg, char *descr, long value) {
  PyObject *tmp = PyInt_FromLong(value);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetInt(PyObject *arg, char *descr, long *result) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *result = PyInt_AsLong(tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetLong(PyObject *arg, char *descr, long value) {
  PyObject *tmp = PyLong_FromLong(value);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetLong(PyObject *arg, char *descr, unsigned long value) {
  PyObject *tmp = PyLong_FromUnsignedLong(value);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetLong(PyObject *arg, char *descr, double value) {
  PyObject *tmp = PyLong_FromDouble(value);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetLong(PyObject *arg, char *descr, long *result) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *result = PyLong_AsLong(tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetLong(PyObject *arg, char *descr, unsigned long *result) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *result = PyLong_AsUnsignedLong(tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetLong(PyObject *arg, char *descr, double *result) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *result = PyLong_AsDouble(tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetFloat(PyObject *arg, char *descr, double value) {
  PyObject *tmp = PyFloat_FromDouble(value);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetFloat(PyObject *arg, char *descr, double *result) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *result = PyFloat_AsDouble(tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonSetComplex(PyObject *arg, char *descr, double real, double imag) {
  PyObject *tmp = PyComplex_FromDoubles(real, imag);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetComplex(PyObject *arg, char *descr, double *real, double *imag) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *real = PyComplex_RealAsDouble(tmp);
  *imag = PyComplex_ImagAsDouble(tmp);
  Py_DECREF(tmp);
}

PyObject *PythonObject::pythonGetArg(int handle) {
  CmiLock(CsvAccess(pyLock));
  PyObject *result = ((*CsvAccess(pyWorkers))[handle]).arg;
  CmiUnlock(CsvAccess(pyLock));
  return result;
}

void PythonObject::pythonPrepareReturn(int handle) {
  pythonAwake(handle);
}

void PythonObject::pythonReturn(int handle) {
  CmiLock(CsvAccess(pyLock));
  CthThread handleThread = ((*CsvAccess(pyWorkers))[handle]).thread;
  CmiUnlock(CsvAccess(pyLock));
  CthAwaken(handleThread);
}

void PythonObject::pythonReturn(int handle, PyObject* data) {
  CmiLock(CsvAccess(pyLock));
  CthThread handleThread = ((*CsvAccess(pyWorkers))[handle]).thread;
  *((*CsvAccess(pyWorkers))[handle]).result = data;
  CmiUnlock(CsvAccess(pyLock));
  CthAwaken(handleThread);
}

void PythonObject::pythonAwake(int handle) {
  CmiLock(CsvAccess(pyLock));
  PyThreadState *handleThread = ((*CsvAccess(pyWorkers))[handle]).pythread;
  CmiUnlock(CsvAccess(pyLock));
  PyEval_AcquireLock();
  PyThreadState_Swap(handleThread);
}

void PythonObject::pythonSleep(int handle) {
  PyEval_ReleaseLock();
}

#include "PythonCCS.def.h"
