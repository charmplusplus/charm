#include "PythonCCS.h"

//CsvDeclare(CmiNodeLock, pyLock);
//CsvDeclare(PythonTable *, pyWorkers);
//CsvDeclare(CmiUInt4, pyNumber);
CtvDeclare(PyObject *, pythonReturnValue);

CProxy_PythonCCS pythonCcsProxy;

// One-time per-processor setup routine
// main interface for python to access common charm methods
static PyObject *CkPy_printstr(PyObject *self, PyObject *args) {
  char *stringToPrint;
  if (!PyArg_ParseTuple(args, "s:printstr", &stringToPrint)) return NULL;
  CkPrintf("%s\n",stringToPrint);
  Py_INCREF(Py_None);return Py_None; //Return-nothing idiom
}

static inline void Ck_printclient(PythonObject *object, CmiUInt4 ref, char* str) {
  //  CmiLock(CsvAccess(pyLock));
  PythonStruct &worker = object->pyWorkers[ref];
  //PythonObject *pyWorker = ((*CsvAccess(pyWorkers))[pyReference]).object;
  //if (((*CsvAccess(pyWorkers))[ref]).clientReady > 0) {
  if (worker.clientReady > 0) {
    // return the string to the client
    // since there is a client waiting, it means there must not be
    // pending strings to be returned ("printed" is empty)
    //CkPrintf("printing data to the client\n");
    CcsSendDelayedReply(worker.client, strlen(str), str);
    worker.printed.erase();
    worker.clientReady = 0;
  } else {
    // add the string to those in list to be returned if it is keepPrint
    if (worker.isKeepPrint) {
      worker.printed += std::string(str);
    }
    // else just drop the line
  }
  //CmiUnlock(CsvAccess(pyLock));
}

static PyObject *CkPy_print(PyObject *self, PyObject *args) {
  char *stringToPrint;
  if (!PyArg_ParseTuple(args, "s:printclient", &stringToPrint)) return NULL;
  PyObject *dict = PyModule_GetDict(PyImport_AddModule("__main__"));
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(dict,"__charmNumber__"));
  PythonObject *object = (PythonObject*)PyLong_AsVoidPtr(PyDict_GetItemString(dict,"__charmObject__"));
  Ck_printclient(object, pyReference, stringToPrint);
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
  PyObject *dict = PyModule_GetDict(PyImport_AddModule("__main__"));
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(dict,"__charmNumber__"));
  PythonObject *object = (PythonObject*)PyLong_AsVoidPtr(PyDict_GetItemString(dict,"__charmObject__"));
  //CmiLock(CsvAccess(pyLock));
  ArrayElement1D *pyArray1 = dynamic_cast<ArrayElement1D*>(object);
  ArrayElement2D *pyArray2 = dynamic_cast<ArrayElement2D*>(object);
  ArrayElement3D *pyArray3 = dynamic_cast<ArrayElement3D*>(object);
  ArrayElement4D *pyArray4 = dynamic_cast<ArrayElement4D*>(object);
  ArrayElement5D *pyArray5 = dynamic_cast<ArrayElement5D*>(object);
  ArrayElement6D *pyArray6 = dynamic_cast<ArrayElement6D*>(object);
  //CmiUnlock(CsvAccess(pyLock));
  if (pyArray1) return Py_BuildValue("(i)", pyArray1->thisIndex);
  else if (pyArray2) return Py_BuildValue("(ii)", pyArray2->thisIndex.x, pyArray2->thisIndex.y);
  else if (pyArray3) return Py_BuildValue("(iii)", pyArray3->thisIndex.x, pyArray3->thisIndex.y, pyArray3->thisIndex.z);
  else if (pyArray4) return Py_BuildValue("(iiii)", pyArray4->thisIndex.w, pyArray4->thisIndex.x, pyArray4->thisIndex.y, pyArray4->thisIndex.z);
  else if (pyArray5) return Py_BuildValue("(iiiii)", pyArray5->thisIndex.v, pyArray5->thisIndex.w, pyArray5->thisIndex.x, pyArray5->thisIndex.y, pyArray5->thisIndex.z);
  else if (pyArray6) return Py_BuildValue("(iiiiii)", pyArray6->thisIndex.x1, pyArray6->thisIndex.y1, pyArray6->thisIndex.z1, pyArray6->thisIndex.x2, pyArray6->thisIndex.y2, pyArray6->thisIndex.z2);
  else { Py_INCREF(Py_None);return Py_None;}
  //return Py_BuildValue("i", (*CsvAccess(pyWorkers))[0]->thisIndex);
}

// method to read a variable and convert it to a python object
static PyObject *CkPy_read(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, "O:read")) return NULL;
  PyObject *dict = PyModule_GetDict(PyImport_AddModule("__main__"));
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(dict,"__charmNumber__"));
  PythonObject *object = (PythonObject*)PyLong_AsVoidPtr(PyDict_GetItemString(dict,"__charmObject__"));
  //CmiLock(CsvAccess(pyLock));
  //CmiUnlock(CsvAccess(pyLock));
  return object->read(args);
}

// method to convert a python object into a variable and write it
static PyObject *CkPy_write(PyObject *self, PyObject *args) {
  PyObject *where, *what;
  if (!PyArg_ParseTuple(args, "OO:write",&where,&what)) return NULL;

  /* Problem solving:

     when the calling parameters (where and what) are formed by only one value,
     the parsing done here above will return two objects which are not tuples
     (and no parenthesis on the caller will change this). For this reason, we
     have to recreate two new tuples with the given argements to pass to the
     write function of the object.
  */
  PyObject *whereT, *whatT;
  if (PyTuple_Check(where)) whereT = where;
  else {
    whereT = PyTuple_New(1);
    PyTuple_SET_ITEM(whereT, 0, where);
  }
  if (PyTuple_Check(what)) whatT = what;
  else {
    whatT = PyTuple_New(1);
    PyTuple_SET_ITEM(whatT, 0, what);
  }
  PyObject *dict = PyModule_GetDict(PyImport_AddModule("__main__"));
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(dict,"__charmNumber__"));
  PythonObject *object = (PythonObject*)PyLong_AsVoidPtr(PyDict_GetItemString(dict,"__charmObject__"));
  object->write(whereT, whatT);
  Py_DECREF(whereT);
  Py_DECREF(whatT);
  Py_INCREF(Py_None);return Py_None;
}

PyMethodDef PythonObject::CkPy_MethodsCustom[] = {
  {NULL,      NULL}        /* Sentinel */
};

char* PythonObject::CkPy_MethodsCustomDoc = "";

PyMethodDef CkPy_MethodsDefault[] = {
  {"printstr", CkPy_printstr, METH_VARARGS},
  {"printclient", CkPy_print, METH_VARARGS},
  {"mype", CkPy_mype, METH_VARARGS},
  {"numpes", CkPy_numpes, METH_VARARGS},
  {"myindex", CkPy_myindex, METH_VARARGS},
  {"read", CkPy_read, METH_VARARGS},
  {"write", CkPy_write, METH_VARARGS},
  {NULL,      NULL}        /* Sentinel */
};

void PythonObject::replyIntValue(CcsDelayedReply *reply, CmiUInt4 *value) {
  CkPrintf("[%d] PythonObject::replyIntValue\n",CkMyPe());
  PythonReplyInt forward;
  forward.reply = *reply;
  forward.value = *value;
  CkCallback cb(CkIndex_PythonCCS::forwardInt(0), pythonCcsProxy);
  ArrayElement *array = dynamic_cast<ArrayElement *>(this);
  Group *group = dynamic_cast<Group *>(this);
  if (array != NULL) array->contribute(sizeof(PythonReplyInt), &forward, CkReduction::bitvec_and, cb);
  else if (group != NULL) group->contribute(sizeof(PythonReplyInt), &forward, CkReduction::bitvec_and, cb);
  else CcsSendDelayedReply(*reply, sizeof(CmiUInt4), (void *)value);
}

void PythonObject::pyRequest (CkCcsRequestMsg *msg) {
  PythonAbstract *pyAbstract = (PythonAbstract *)msg->data;
  pyAbstract->unpack();

  if (pyAbstract->isExecute()) {
    execute(msg, &msg->reply);
    // the message is not deleted here, it will deleted by the function itself
    // deleting it here creates problems with the high level scripting where
    // a new thread is created. the alternative will be to memcopy the message
  } else if (pyAbstract->isPrint()) {
    print((PythonPrint *)msg->data, &msg->reply);
    delete msg;
  } else if (pyAbstract->isFinished()) {
    finished((PythonFinished *)msg->data, &msg->reply);
    delete msg;
  } else {
    CkPrintf("Wrong request arrived!\n");
    delete msg;
  }
}


void PythonObject::print (PythonPrint *pyMsg, CcsDelayedReply *reply) {
  // ATTN: be sure that in all possible paths pyLock is released!
  //CmiLock(CsvAccess(pyLock));
  CmiUInt4 returnValue;
  pyMsg->unpack();

  PythonTable::iterator iter = pyWorkers.find(pyMsg->interpreter);
  if (iter == pyWorkers.end()) {
    // Malformed request!
    //CkPrintf("PythonCCS: print request on invalid interpreter\n");
    //CmiUnlock(CsvAccess(pyLock));
    returnValue = htonl(0);
    replyIntValue(reply, &returnValue);
  } else {
    // it is a correct print request, parse it

    if (pyMsg->isKill()) {
      // kill the pending client print request
      if (iter->second.clientReady == 1) {
	returnValue = htonl(0);
	replyIntValue(&iter->second.client, &returnValue);
      }
      //CmiUnlock(CsvAccess(pyLock));
      // do no return anything to the calling socket
      CcsNoDelayedReply(*reply);
      return;
    }

    // is something already printed?
    if (iter->second.printed.length() > 0) {
      // send back to the client the string
      const char *str = iter->second.printed.c_str();
      int length = strlen(str);
      //CkPrintf("sending data to the client\n");
      PythonReplyString *forward = new (length) PythonReplyString();
      forward->reply = *reply;
      memcpy(forward->data, str, length);
      CkCallback cb(CkIndex_PythonCCS::forwardString(0), pythonCcsProxy);
      ArrayElement *array = dynamic_cast<ArrayElement *>(this);
      Group *group = dynamic_cast<Group *>(this);
      if (array != NULL) array->contribute(sizeof(CcsDelayedReply)+length, forward, PythonCCS::reduceString, cb);
      else if (group != NULL) group->contribute(sizeof(CcsDelayedReply)+length, forward, PythonCCS::reduceString, cb);
      else CcsSendDelayedReply(*reply, length, str);
      iter->second.printed.erase();
      if (iter->second.clientReady == -1) {
	// after the client flush the printed buffer, delete the entry
	pyWorkers.erase(pyMsg->interpreter);
      }
    } else {
      // nothing printed, store the client request if it will be waiting
      if (pyMsg->isWait()) {
	// check if someone else has requested prints, if yes, kill the other
	if (iter->second.clientReady == 1) {
	  returnValue = htonl(0);
	  replyIntValue(&iter->second.client, &returnValue);
	}
	iter->second.client = *reply;
	iter->second.clientReady = 1;
      } else {
	// return null
	returnValue = htonl(0);
	replyIntValue(reply, &returnValue);
      }
    }
    //CmiUnlock(CsvAccess(pyLock));
  }
}

void PythonObject::finished (PythonFinished *pyMsg, CcsDelayedReply *reply) {
   // ATTN: be sure that in all possible paths pyLock is released!
  //CmiLock(CsvAccess(pyLock));
  CmiUInt4 pyReference = pyMsg->interpreter;
  CmiUInt4 returnValue;
  pyMsg->unpack();

  PythonTable::iterator iter = pyWorkers.find(pyMsg->interpreter);
  if (iter == pyWorkers.end() || !iter->second.inUse) {
    //ckout <<"answering Finished yes"<<endl;
    returnValue = htonl(pyReference);
    replyIntValue(reply, &returnValue);
    //CmiUnlock(CsvAccess(pyLock));
    return;
  }

  // the client is in use
  if (pyMsg->isWait()) {
    // is there another client waiting for termination?
    if (iter->second.finishReady) {
      // kill the previous requester
      returnValue = htonl(0);
      replyIntValue(&iter->second.finish, &returnValue);
    }
    //ckout <<"queueing Finished request"<<endl;
    iter->second.finish = *reply;
    iter->second.finishReady = 1;
  } else {
    //ckout <<"answering Finished no"<<endl;
    returnValue = htonl(0);
    replyIntValue(reply, &returnValue);
  }
  //CmiUnlock(CsvAccess(pyLock));
}

void PythonObject::execute (CkCcsRequestMsg *msg, CcsDelayedReply *reply) {
  // ATTN: be sure that in all possible paths pyLock is released!
  PythonExecute *pyMsg = (PythonExecute *)msg->data;
  //PyEval_AcquireLock();
  //CmiLock(CsvAccess(pyLock));
  CmiUInt4 pyReference;
  CmiUInt4 returnValue;

  // re-establish the pointers in the structure
  pyMsg->unpack();

  if (pyMsg->interpreter > 0) {
    // the user specified an interpreter, check if it is free
    PythonTable::iterator iter;
    if ((iter=pyWorkers.find(pyMsg->interpreter))!=pyWorkers.end() && !iter->second.inUse && iter->second.clientReady!=-1) {
      // the interpreter already exists and it is neither in use, nor dead
      //CkPrintf("interpreter present and not in use\n");
      pyReference = pyMsg->interpreter;
      PyEval_AcquireLock();
      PyThreadState_Swap(iter->second.pythread);
    } else {
      // ops, either the iterator does not exist or is already in use, return an
      // error to the client, we don't want to create a new interpreter if the
      // old is in use, because this can corrupt the semantics of the user code.
      //if (iter==CsvAccess(pyWorkers)->end()) CkPrintf("asked for an interpreter not present\n");
      //else CkPrintf("interpreter already in use\n");
      returnValue = htonl(0);
      replyIntValue(reply, &returnValue);
      //CmiUnlock(CsvAccess(pyLock));
      //PyEval_ReleaseLock();
      return;  // stop the execution
    }
  } else {
    // the user didn't specify an interpreter, create a new one
    //CkPrintf("creating new interpreter\n");

    // update the reference number, used to access the current chare
    pyReference = ++pyNumber;
    pyNumber &= ~(1<<31);
    //pyWorkers[pyReference].object = this;
    pyWorkers[pyReference].clientReady = 0;

    // create the new interpreter
    PyEval_AcquireLock();
    PyThreadState *pts = Py_NewInterpreter();

    CkAssert(pts != NULL);
    pyWorkers[pyReference].pythread = pts;

    Py_InitModule("ck", CkPy_MethodsDefault);
    if (pyMsg->isHighLevel()) Py_InitModule("charm", getMethods());

    // insert into the dictionary a variable with the reference number
    PyObject *mod = PyImport_AddModule("__main__");
    PyObject *dict = PyModule_GetDict(mod);

    PyDict_SetItemString(dict,"__charmNumber__",PyInt_FromLong(pyReference));
    PyDict_SetItemString(dict,"__charmObject__",PyLong_FromVoidPtr(this));
    PyRun_String("import ck\nimport sys\n"
		 "ck.__doc__ = \"Ck module: basic charm routines\\n"
		 "printstr(str) -- print a string on the server\\n"
		 "printclient(str) -- print a string on the client\\n"
		 "mype() -- return an integer for MyPe()\\n"
		 "numpes() -- return an integer for NumPes()\\n"
		 "myindex() -- return a tuple containing the array index (valid only for arrays)\\n"
		 "read(where) -- read a value on the chare (uses the \\\"read\\\" method of the chare)\\n"
		 "write(where, what) -- write a value back on the chare (uses the \\\"write\\\" method of the chare)\\n\"",
		 Py_file_input,dict,dict);
    if (pyMsg->isHighLevel()) {
      PyRun_String("import charm",Py_file_input,dict,dict);
      PyRun_String(getMethodsDoc(),Py_file_input,dict,dict);
    }

    PyRun_String("class __charmOutput__:\n"
		 "    def __init__(self, stdout):\n"
		 "        self.stdout = stdout\n"
		 "    def write(self, s):\n"
		 "        ck.printclient(s)\n"
		 "sys.stdout = __charmOutput__(sys.stdout)"
		 ,Py_file_input,dict,dict);

  }

  pyWorkers[pyReference].inUse = true;
  if (pyMsg->isKeepPrint()) {
    pyWorkers[pyReference].isKeepPrint = true;
  } else {
    pyWorkers[pyReference].isKeepPrint = false;
  }

  if (pyMsg->isWait()) {
    pyWorkers[pyReference].finish = *reply;
    pyWorkers[pyReference].finishReady = 1;
  } else {
    pyWorkers[pyReference].finishReady = 0;
    // send back this number to the client, which is an ack
    ckout<<"new interpreter created "<<pyReference<<endl;
    returnValue = htonl(pyReference);
    replyIntValue(reply, &returnValue);
  }

  //if (((*CsvAccess(pyWorkers))[pyReference]).object != this) ckout<<"object not this"<<endl;
  //else ckout<<"object ok"<<endl;

  //CmiUnlock(CsvAccess(pyLock));

  // run the program
  if (pyMsg->isHighLevel()) {
    CthResume(CthCreate((CthVoidFn)_callthr_executeThread, new CkThrCallArg(msg,this), 0));
    // msg is delete inside the newly created thread
  } else {
    executeThread(pyMsg);
    // delete the message, execute was delegated
    delete msg;
  }
}

// created in a new thread, call the executeThread method
// we know that the impl_msg contains a CkCcsRequestMsg which data is a
// PythonExecute, so we pass directly this latest parameter
void PythonObject::_callthr_executeThread(CkThrCallArg *impl_arg) {
  CkCcsRequestMsg *impl_msg = (CkCcsRequestMsg*)impl_arg->msg;
  PythonObject *impl_obj = (PythonObject *) impl_arg->obj;
  delete impl_arg;

  impl_obj->executeThread((PythonExecute*)impl_msg->data);
  delete impl_msg;
}

void PythonObject::executeThread(PythonExecute *pyMsg) {
  // get the information about the running python thread and my reference number
  //ckout << "options  "<<pyMsg->isPersistent()<<endl;
  PyThreadState *mine = PyThreadState_Get();
  PyObject *dict = PyModule_GetDict(PyImport_AddModule("__main__"));
  CmiUInt4 pyReference = PyInt_AsLong(PyDict_GetItemString(dict,"__charmNumber__"));

  // store the self thread for future suspention if high level execution
  if (pyMsg->isHighLevel()) {
    //CmiLock(CsvAccess(pyLock));
    pyWorkers[pyReference].thread=CthSelf();
    //CmiUnlock(CsvAccess(pyLock));
  }

  // decide whether it is iterative or not
  if (!pyMsg->isIterate()) {
    PyObject* python_output = PyRun_String(pyMsg->code.code, Py_file_input, dict, dict);
    //PyRun_SimpleString(pyMsg->code.code);
    //CkPrintf("python_output = %d\n",python_output);
    if (python_output == NULL) {
      // return the string error to the client
      PyObject *ptype, *pvalue, *ptraceback;
      PyErr_Fetch(&ptype, &pvalue, &ptraceback);
      PyObject *strP = PyObject_Str(ptype);
      char *str = PyString_AsString(strP);
      Ck_printclient(this, pyReference, str);
      //CkPrintf("%s\n",str);
      Py_DECREF(strP);
      Ck_printclient(this, pyReference, ": ");
      strP = PyObject_Str(pvalue);
      str = PyString_AsString(strP);
      Ck_printclient(this, pyReference, str);
      //CkPrintf("%s\n",str);
      Py_DECREF(strP);
      //PyObject_Print(ptype, stdout, 0);
      //CkPrintf("   %d %d %d %d\n",PyType_Check(ptype),PyString_Check(ptype),PyList_Check(ptype),PyTuple_Check(ptype));
      //PyObject_Print(pvalue, stdout, 0);
      //CkPrintf("   %d %d %d %d\n",PyType_Check(pvalue),PyString_Check(pvalue),PyList_Check(pvalue),PyTuple_Check(pvalue));
      //PyObject_Print(ptraceback, stdout, 0);
      //if (ptraceback) CkPrintf("   %d %d %d %d\n",PyType_Check(ptraceback),PyString_Check(ptraceback),PyList_Check(ptraceback),PyTuple_Check(ptraceback));
    }
  } else {
    CkPrintf("userCode: |%s|",pyMsg->code);
    CkPrintf("method: |%s|",pyMsg->methodName.methodName);
    // compile the program
    char *userCode = pyMsg->code.code;
    struct _node* programNode = PyParser_SimpleParseString(userCode, Py_file_input);
    if (programNode==NULL) {
      CkPrintf("Program Parse Error\n");
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
      return;
    }
    PyCodeObject *program = PyNode_Compile(programNode, "");
    if (program==NULL) {
      CkPrintf("Program Compile Error\n");
      PyNode_Free(programNode);
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
      return;
    }
    PyObject *mod = PyImport_AddModule("__main__");
    PyObject *dict = PyModule_GetDict(mod);
    PyObject *code = PyEval_EvalCode(program, dict, dict);
    if (code==NULL) {
      CkPrintf("Program Eval Error\n");
      PyNode_Free(programNode);
      Py_DECREF(program);
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
      return;
    }

    // load the user defined method
    char *userMethod = pyMsg->methodName.methodName;
    PyObject *item = PyDict_GetItemString(dict, userMethod);
    if (item==NULL) {
      CkPrintf("Method not found\n");
      PyNode_Free(programNode);
      Py_DECREF(program);
      Py_DECREF(code);
      // distroy map element in pyWorkers and terminate interpreter
      cleanup(pyMsg, mine, pyReference);
      return;
    }

    // create the container for the data
    PyRun_String("class CharmContainer:\n\tpass\n\n", Py_file_input, dict, dict);
    PyObject *part = PyRun_String("CharmContainer()", Py_eval_input, dict, dict);
    PyObject *arg = PyTuple_New(1);
    PyObject *oldArg = part;

    // construct the iterator calling the user defined method in the interiting class
    PythonIterator *userIterator = pyMsg->info.info;
    int more = buildIterator(part, userIterator);
    if (oldArg != part) Py_DECREF(oldArg);
    PyTuple_SetItem(arg, 0, part);

    // iterate over all the provided iterators from the user class
    PyObject *result;
    while (more) {
      result = PyObject_CallObject(item, arg);
      if (!result) {
	CkPrintf("Python Call error\n");
        //PyErr_Print();
	break;
      }
      oldArg = part;
      more = nextIteratorUpdate(part, result, userIterator);
      if (oldArg != part) {
        Py_DECREF(oldArg);
        PyTuple_SetItem(arg, 0, part);
      }
      Py_DECREF(result);
    }

    Py_DECREF(part);
    Py_DECREF(arg);
    PyNode_Free(programNode);
    Py_DECREF(program);
    Py_DECREF(code);

  } // end decision if it is iterate or not

  cleanup(pyMsg, mine, pyReference);

}

// this function takes care of destroying the interpreter, deleting the
// reference into the map table, depending on the flags specified
void PythonObject::cleanup (PythonExecute *pyMsg, PyThreadState *pts, CmiUInt4 pyReference) {
  CmiUInt4 returnValue;

  //ckout <<"cleanup called"<<endl;
  //CmiLock(CsvAccess(pyLock));
  // if there is someone waiting for finish, send ackowledge
  if (pyWorkers[pyReference].finishReady) {
    //ckout <<"answering the client finish"<<endl;
    returnValue = htonl(pyReference);
    replyIntValue(&pyWorkers[pyReference].finish, &returnValue);
    pyWorkers[pyReference].finishReady = 0;
  }

  //ckout << "options"<<pyMsg->isPersistent()<<endl;
  pyWorkers[pyReference].inUse = false;
  if (!pyMsg->isPersistent()) {
    Py_EndInterpreter(pts);
    pyWorkers[pyReference].clientReady = -1;
  }
  PyEval_ReleaseLock();

  if (!pyMsg->isPersistent() && pyWorkers[pyReference].printed.length()==0) {
    // destroy the entry in the map
    //CkPrintf("destroyed interpreter\n");
    pyWorkers.erase(pyReference);
  }
  //CmiUnlock(CsvAccess(pyLock));
}

void PythonObject::getPrint(CkCcsRequestMsg *msg) {

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

void PythonObject::pythonSetPointer(PyObject *arg, char *descr, void *ptr) {
  PyObject *tmp = PyLong_FromVoidPtr(ptr);
  PyObject_SetAttrString(arg, descr, tmp);
  Py_DECREF(tmp);
}

void PythonObject::pythonGetPointer(PyObject *arg, char *descr, void **ptr) {
  PyObject *tmp = PyObject_GetAttrString(arg, descr);
  *ptr = PyLong_AsVoidPtr(tmp);
  Py_DECREF(tmp);
}

PyObject *PythonObject::pythonGetArg(int handle) {
  //CmiLock(CsvAccess(pyLock));
  PyObject *result = pyWorkers[handle].arg;
  //CmiUnlock(CsvAccess(pyLock));
  return result;
}

void PythonObject::pythonReturn(int handle) {
  // The return value is now set to zero before calling the high-level function, so here
  // we don't need to do anything.
  //CmiLock(CsvAccess(pyLock));
  //CthThread handleThread = ((*CsvAccess(pyWorkers))[handle]).thread;
  //*((*CsvAccess(pyWorkers))[handle]).result = 0;
  //CmiUnlock(CsvAccess(pyLock));
  //CthAwaken(handleThread);
}

void PythonObject::pythonReturn(int handle, PyObject* data) {
  //CmiLock(CsvAccess(pyLock));
  //CthThread handleThread = ((*CsvAccess(pyWorkers))[handle]).thread;
  * pyWorkers[handle].result = data;
  //CmiUnlock(CsvAccess(pyLock));
  //CthAwaken(handleThread);
}

/*
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
*/

PythonCCS::PythonCCS(CkArgMsg *arg) {
  pythonCcsProxy = thishandle;
}

CkReductionMsg *pythonCombinePrint(int nMsg, CkReductionMsg **msgs) {
  // the final length is the sum of all the NULL-terminated strings, minus
  // the initial CcsDelayedReply structs and the terminating NULL characters
  // which are present only once in the final message
  int length = - (nMsg-1) * sizeof(CcsDelayedReply);
  for (int i=0; i<nMsg; ++i) {
    length += msgs[i]->getSize();
  }
  
  CkReductionMsg *result = CkReductionMsg::buildNew(length,NULL);
  
  PythonReplyString *data = (PythonReplyString*)(result->getData());
  data->reply = ((PythonReplyString*)msgs[0]->getData())->reply;
  char *cur=data->data;
  for (int i=0; i<nMsg; ++i) {
    int messageBytes=msgs[i]->getSize() - sizeof(CcsDelayedReply);
    memcpy((void *)cur,(void *)((PythonReplyString*)msgs[i]->getData())->data,messageBytes);
    cur+=messageBytes;
  }
  return result;
}

void PythonCCS::forwardString(CkReductionMsg *msg) {
  PythonReplyString *forward = (PythonReplyString *)msg->getData();
  CcsSendDelayedReply(forward->reply, msg->getSize()-sizeof(CcsDelayedReply), (void *)forward->data);
}

void PythonCCS::forwardInt(CkReductionMsg *msg) {
  CkPrintf("PythonCCS::forwardInt\n");
  PythonReplyInt *forward = (PythonReplyInt *)msg->getData();
  CcsSendDelayedReply(forward->reply, sizeof(CmiUInt4), (void *)&forward->value);
}

CkReduction::reducerType PythonCCS::reduceString;

static void initializePythonDefault(void) {
  //CsvInitialize(CmiUInt, pyNumber);
  //CsvAccess(pyNumber) = 0;
  //CsvInitialize(PythonTable *,pyWorkers);
  //CsvAccess(pyWorkers) = new PythonTable();
  //CsvInitialize(CmiNodeLock, pyLock);
  //CsvAccess(pyLock) = CmiCreateLock();
  CtvInitialize(PyObject *,pythonReturnValue);

  PythonCCS::reduceString = CkReduction::addReducer(pythonCombinePrint);
  
  Py_Initialize();
  PyEval_InitThreads();

  PyEval_ReleaseLock();
}

#include "PythonCCS.def.h"
