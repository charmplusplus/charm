#include "PythonCCS.h"

CsvDeclare(CmiNodeLock, pyLock);
CsvDeclare(PythonTable *, pyWorkers);
CsvDeclare(CmiUInt4, pyNumber);
CtvDeclare(PyObject *, pythonReturnValue);

// One-time per-processor setup routine
// main interface for python to access common charm methods
static PyObject *CkPy_printstr(PyObject *self, PyObject *args) {
  char *stringToPrint;
  if (!PyArg_ParseTuple(args, "s:printstr", &stringToPrint)) return NULL;
  CkPrintf("%s\n",stringToPrint);
  Py_INCREF(Py_None);return Py_None; //Return-nothing idiom
}

static PyObject *CkPy_print(PyObject *self, PyObject *args) {
  char *stringToPrint;
  if (!PyArg_ParseTuple(args, "s:print", &stringToPrint)) return NULL;
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = ((*CsvAccess(pyWorkers))[pyReference]).object;
  if (((*CsvAccess(pyWorkers))[pyReference]).clientReady > 0) {
    // return the string to the client
    // since there is a client waiting, it means there must not be
    // pending strings to be returned ("printed" is empty)
    //CkPrintf("printing data to the client\n");
    CcsDelayedReply client = ((*CsvAccess(pyWorkers))[pyReference]).client;
    CcsSendDelayedReply(client, strlen(stringToPrint)+1, stringToPrint);
    ((*CsvAccess(pyWorkers))[pyReference]).clientReady = 0;
  } else {
    // add the string to those in list to be returned
    ((*CsvAccess(pyWorkers))[pyReference]).printed += std::string(stringToPrint);
  }
  CmiUnlock(CsvAccess(pyLock));
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
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  ArrayElement1D *pyArray = dynamic_cast<ArrayElement1D>((*CsvAccess(pyWorkers))[pyReference]);
  CmiUnlock(CsvAccess(pyLock));
  if (pyArray) return Py_BuildValue("i", pyArray->thisIndex);
  else { Py_INCREF(Py_None);return Py_None;}
  //return Py_BuildValue("i", (*CsvAccess(pyWorkers))[0]->thisIndex);
}
*/

// method to read a variable and convert it to a python object
static PyObject *CkPy_read(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "O:read")) return NULL;
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = ((*CsvAccess(pyWorkers))[pyReference]).object;
  CmiUnlock(CsvAccess(pyLock));
  return pyWorker->read(args);
}

// method to convert a python object into a variable and write it
static PyObject *CkPy_write(PyObject *self, PyObject *args) {
  PyObject *where, *what;
  if (!PyArg_ParseTuple(args, "OO:write",&where,&what)) return NULL;
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));
  CmiLock(CsvAccess(pyLock));
  PythonObject *pyWorker = ((*CsvAccess(pyWorkers))[pyReference]).object;
  CmiUnlock(CsvAccess(pyLock));
  pyWorker->write(where, what);
  Py_INCREF(Py_None);return Py_None;
}

PyMethodDef PythonObject::CkPy_MethodsCustom[] = {
  {NULL,      NULL}        /* Sentinel */
};

PyMethodDef CkPy_MethodsDefault[] = {
  {"printstr", CkPy_printstr, METH_VARARGS},
  {"printclient", CkPy_print, METH_VARARGS},
  {"mype", CkPy_mype, METH_VARARGS},
  {"numpes", CkPy_numpes, METH_VARARGS},
  {"read", CkPy_read, METH_VARARGS},
  {"write", CkPy_write, METH_VARARGS},
  {NULL,      NULL}        /* Sentinel */
};

void PythonObject::execute (CkCcsRequestMsg *msg) {
  PythonAbstract *pyAbstract = (PythonAbstract *)msg->data;
  PythonPrint *pyPrint=0;
  PythonExecute *pyMsg=0;
  if (pyAbstract->magic == sizeof(PythonPrint)) {
    pyPrint = (PythonPrint *)msg->data;
  } else if (pyAbstract->magic == sizeof(PythonExecute)) {
    pyMsg = (PythonExecute *)msg->data;
  } else {
    CkPrintf("Wrong request arrived!\n");
    return;
  }

  // ATTN: be sure that in all possible paths pyLock is released!
  CmiLock(CsvAccess(pyLock));
  CmiUInt4 pyReference;

  // check if this is just a request for prints
  if (pyPrint) {
    PythonTable::iterator iter = CsvAccess(pyWorkers)->find((CmiUInt4)*pyPrint->interpreter);
    if (iter == CsvAccess(pyWorkers)->end()) {
      // Malformed request!
      CkPrintf("PythonCCS: print request on invalid interpreter\n");
      pyReference = 0;
      CmiUnlock(CsvAccess(pyLock));
      CcsSendDelayedReply(msg->reply, sizeof(CmiUInt4), (void *)&pyReference);
    } else {
      if (iter->second.printed.length() > 0) {
	// send back to the client the string
	const char *str = iter->second.printed.data();
	//CkPrintf("sending data to the client\n");
	CcsSendDelayedReply(msg->reply, strlen(str)+1, str);
	if (iter->second.clientReady == -1) {
	  // after the client flush the printed buffer, delete the entry
	  CsvAccess(pyWorkers)->erase((CmiUInt4)*pyPrint->interpreter);
	}
	CmiUnlock(CsvAccess(pyLock));
      } else {
	// nothing printed, store the client request if it will be waiting
	if (pyPrint->isWait()) {
	  iter->second.client = msg->reply;
	  iter->second.clientReady = 1;
	}
	CmiUnlock(CsvAccess(pyLock));
      }
    }
    delete msg;
    return;
  }

  // re-establish the pointers in the structure
  pyMsg->unpack();

  if ((CmiUInt4)*pyMsg->interpreter > 0) {
    // the user specified an interpreter, check if it is free
    PythonTable::iterator iter;
    if ((iter=CsvAccess(pyWorkers)->find((CmiUInt4)*pyMsg->interpreter))!=CsvAccess(pyWorkers)->end() && !iter->second.inUse) {
      // the interpreter already exists and it is not in use
      //CkPrintf("interpreter present and not in use\n");
      pyReference = (CmiUInt4)*pyMsg->interpreter;
      iter->second.inUse = true;
      // send back this number to the client, which is an ack
      CcsSendDelayedReply(msg->reply, sizeof(CmiUInt4), (void *)&pyReference);
      CmiUnlock(CsvAccess(pyLock));
      PyEval_AcquireLock();
    } else {
      // ops, either the iterator does not exist or is already in use, return an
      // error to the client, we don't want to create a new interpreter if the
      // old is in use, because this can corrupt the semantics of the user code.
      //if (iter!=CsvAccess(pyWorkers)->end()) CkPrintf("asked for an interpreter not present\n");
      //else CkPrintf("interpreter already in use\n");
      pyReference = ~0;
      CcsSendDelayedReply(msg->reply, sizeof(CmiUInt4), (void *)&pyReference);
      CmiUnlock(CsvAccess(pyLock));
      return;  // stop the execution
    }
  } else {
    // the user didn't specify an interpreter, create a new one
    //CkPrintf("creating new interpreter\n");

    // update the reference number, used to access the current chare
    pyReference = CsvAccess(pyNumber)++;
    CsvAccess(pyNumber) &= ~(1<<31);
    ((*CsvAccess(pyWorkers))[pyReference]).object = this;
    ((*CsvAccess(pyWorkers))[pyReference]).inUse = true;
    CmiUnlock(CsvAccess(pyLock));

    // send back this number to the client
    //CkPrintf("sending interpreter to the client\n");
    CcsSendDelayedReply(msg->reply, sizeof(CmiUInt4), (void *)&pyReference);

    // create the new interpreter
    PyEval_AcquireLock();
    PyThreadState *pts = Py_NewInterpreter();

    Py_InitModule("ck", CkPy_MethodsDefault);
    if (pyMsg->isHighLevel()) Py_InitModule("charm", getMethods());

    // insert into the dictionary a variable with the reference number
    PyObject *mod = PyImport_AddModule("__main__");
    PyObject *dict = PyModule_GetDict(mod);

    PyDict_SetItemString(dict,"charmNumber",PyInt_FromLong(pyReference));
    PyRun_String("import ck",Py_file_input,dict,dict);
    if (pyMsg->isHighLevel()) PyRun_String("import charm",Py_file_input,dict,dict);
  }

  // run the program
  if (pyMsg->isHighLevel()) {
    CthResume(CthCreate((CthVoidFn)_callthr_executeThread, new CkThrCallArg(msg,this), 0));
  } else {
    executeThread(msg);
  }
}

// created in a new thread, call the executeThread method
void PythonObject::_callthr_executeThread(CkThrCallArg *impl_arg) {
  void *impl_msg = impl_arg->msg;
  PythonObject *impl_obj = (PythonObject *) impl_arg->obj;
  delete impl_arg;
  impl_obj->executeThread((CkCcsRequestMsg*)impl_msg);
}

void PythonObject::executeThread(CkCcsRequestMsg *msg) {
  // get the information about the running python thread and my reference number
  PyThreadState *mine = PyThreadState_Get();
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(PyModule_GetDict(PyImport_AddModule("__main__")),"charmNumber"));

  PythonExecute *pyMsg = (PythonExecute *)msg->data;

  // store the self thread for future suspention if high level execution
  if (pyMsg->isHighLevel()) {
    CmiLock(CsvAccess(pyLock));
    ((*CsvAccess(pyWorkers))[pyReference]).thread=CthSelf();
    CmiUnlock(CsvAccess(pyLock));
  }

  // decide whether it is iterative or not
  if (!pyMsg->isIterate()) {
    PyRun_SimpleString(pyMsg->code);
  } else {
    // compile the program
    char *userCode = pyMsg->code;
    struct _node* programNode = PyParser_SimpleParseString(userCode, Py_file_input);
    if (programNode==NULL) {
      CkPrintf("Program error\n");
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
      delete msg;
      return;
    }
    PyCodeObject *program = PyNode_Compile(programNode, "");
    if (program==NULL) {
      CkPrintf("Program error\n");
      PyNode_Free(programNode);
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
      delete msg;
      return;
    }
    PyObject *mod = PyImport_AddModule("__main__");
    PyObject *dict = PyModule_GetDict(mod);
    PyObject *code = PyEval_EvalCode(program, dict, dict);
    if (code==NULL) {
      CkPrintf("Program error\n");
      PyNode_Free(programNode);
      Py_DECREF(program);
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
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
      cleanup(pyMsg, mine, pyReference);
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

  } // end decision if it is iterate or not

  cleanup(pyMsg, mine, pyReference);
  delete msg;

}

// this function takes care of destroying the interpreter, deleting the
// reference into the map table, depending on the flags specified
void PythonObject::cleanup (PythonExecute *pyMsg, PyThreadState *pts, CmiUInt4 pyReference) {
  if (!pyMsg->isPersistent()) {
    Py_EndInterpreter(pts);
    ((*CsvAccess(pyWorkers))[pyReference]).clientReady = -1;
  }
  PyEval_ReleaseLock();

  if (!pyMsg->isPersistent() && !pyMsg->isKeepPrint()) {
    // destroy the entry in the map
    //CkPrintf("destroyed interpreter\n");
    CmiLock(CsvAccess(pyLock));
    CsvAccess(pyWorkers)->erase(pyReference);
    CmiUnlock(CsvAccess(pyLock));
  }
}

/*
void PythonObject::iterate (CkCcsRequestMsg *msg) {

  // update the reference number, used to access the current chare
  CmiLock(CsvAccess(pyLock));
  CmiUInt4 pyReference = CsvAccess(pyNumber)++;
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
*/

void PythonObject::getPrint(CkCcsRequestMsg *msg) {

}

static void initializePythonDefault(void) {
  CsvInitialize(CmiUInt, pyNumber);
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
