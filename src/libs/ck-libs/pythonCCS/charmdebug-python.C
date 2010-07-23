#include "charmdebug_python.decl.h"

//CpvExtern(CkHashtable_c, ccsTab);
class CpdPython : public CBase_CpdPython {
public:
  CpdPython (CkArgMsg *msg) {
    delete msg;
    //((CProxy_CpdPython)thishandle).registerPython("CpdPython");
    //CkCallback cb(CkIndex_CpdPython::pyRequest(0),thishandle);
    //CcsRegisterHandler("pycode", cb);
    CProxy_CpdPythonGroup group = CProxy_CpdPythonGroup::ckNew();
    group.registerPython("CpdPythonGroup");
    //CcsRegisterHandler("CpdPythonGroup", CkCallback(CkIndex_CpdPythonGroup::pyRequest(0),group));
    //CkPrintf("CpdPython registered\n");
    //char *string = "pycode";
    //CkAssert(CkHashtableGet(CpvAccess(ccsTab),(void *)&string) != NULL);
    CcsRegisterHandler("CpdPythonPersistent", CkCallback(CkIndex_CpdPythonGroup::registerPersistent(0), group));
  }
  //void get(int handle) {
  //  CkPrintf("CpdPython::get\n");
  //}
};

class CpdPythonArrayIterator : public CkLocIterator {
public:
  CkVec<CkMigratable*> elems;
  CkArray *arr;
  virtual void addLocation(CkLocation &loc) {
    elems.insertAtEnd(arr->lookup(loc.getIndex()));
  }
};

class CpdPythonGroup : public CBase_CpdPythonGroup, public CpdPersistentChecker {
  CpdPythonArrayIterator arriter;
  int nextElement;
  bool resultNotNone;
public:
  CpdPythonGroup() {
    //CkPrintf("[%d] CpdPythonGroup::constructor\n",CkMyPe());
  }

  int buildIterator(PyObject*&, void*);
  int nextIteratorUpdate(PyObject*&, PyObject*, void*);

  PyObject *getResultFromType(char, void*);
  void getArray(int handle);
  void getValue(int handle);
  void getCast(int handle);
  void getStatic(int handle);
  
  void getMessage(int handle);

  void cpdCheck(void*);
  void registerPersistent(CkCcsRequestMsg*);
};

int CpdPythonGroup::buildIterator(PyObject *&data, void *iter) {
  resultNotNone = false;
  int group = ntohl(*(int*)iter);
  if (group > 0) {
    CkGroupID id;
    id.idx = group;
    IrrGroup *ptr = _localBranch(id);
    if (ptr->isArrMgr()) {
      arriter.arr = (CkArray*)ptr;
      arriter.arr->getLocMgr()->iterate(arriter);
      if (arriter.elems.size() > 0) {
        data = PyLong_FromVoidPtr(arriter.elems[0]);
        nextElement = 1;
      } else {
        return 0;
      }
    } else {
      nextElement = 0;
      data = PyLong_FromVoidPtr(ptr);
      //CkPrintf("[%d] Building iterator for %i: %p\n", CkMyPe(), group, ptr);
      return 1;
    }
  } else {
    nextElement = 0;
    data = PyLong_FromVoidPtr(CpdGetCurrentObject());
    return 1;
  }
}

int CpdPythonGroup::nextIteratorUpdate(PyObject *&data, PyObject *result, void *iter) {
  //CkPrintf("[%d] Asked for next iterator\n",CkMyPe());
  if (result != Py_None) {
    PyObject *str = PyObject_Str(result);
    CkPrintf("Freezing the application: %s\n",PyString_AsString(str));
    Py_DECREF(str);
    resultNotNone = true;
    return 0;
  }
  if (nextElement > 0) {
    if (nextElement == arriter.elems.size()) {
      nextElement = 0;
      arriter.elems.removeAll();
      return 0;
    } else {
      data = PyLong_FromVoidPtr(arriter.elems[nextElement++]);
      return 1;
    }
  }
  //static int next = 0;
  //next = 1 - next;
  //return next;
  return 0;
}

PyObject *CpdPythonGroup::getResultFromType(char restype, void* ptr) {
  PyObject *result = NULL;
  switch (restype) {
  case 'c':
    result = PyLong_FromVoidPtr(ptr);
    break;
  case 'p':
    result = PyLong_FromVoidPtr(*(void**)ptr);
    break;
  case 'b':
    result = Py_BuildValue("b", *(char*)ptr);
    break;
  case 'h':
    result = Py_BuildValue("h", *(short*)ptr);
    break;
  case 'i':
    result = Py_BuildValue("i", *(int*)ptr);
    break;
  case 'l':
    result = Py_BuildValue("l", *(long*)ptr);
    break;
  case 'f':
    result = Py_BuildValue("f", *(float*)ptr);
    break;
  case 'd':
    result = Py_BuildValue("d", *(double*)ptr);
    break;
  case 's':
    result = Py_BuildValue("s", *(char**)ptr);
    break;
  }
  return result;
}

void CpdPythonGroup::getArray(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *obj;
  int num, size;
  char restype;
  if (PyArg_ParseTuple(arg, "Oici", &obj, &size, &restype, &num) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(obj);
  ptr += num * size;
  PyObject *result = getResultFromType(restype, ptr);
  pythonReturn(handle, result);
}

void CpdPythonGroup::getValue(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *obj;
  int offset;
  char restype;
  if (PyArg_ParseTuple(arg, "Oic", &obj, &offset, &restype) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(obj);
  ptr += offset;
  PyObject *result = getResultFromType(restype, ptr);
  pythonReturn(handle, result);
}

void CpdPythonGroup::getCast(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *obj;
  int offset;
  if (PyArg_ParseTuple(arg, "Oi", &obj, &offset) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(obj);
  ptr += offset;
  pythonReturn(handle, PyLong_FromVoidPtr(ptr));
}

void CpdPythonGroup::getStatic(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *location;
  char restype;
  if (PyArg_ParseTuple(arg, "Oc", &location, &restype) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(location);
  PyObject *result = getResultFromType(restype, ptr);
  pythonReturn(handle, result);
}

void CpdPythonGroup::getMessage(int handle) {
  void *msg = CpdGetCurrentMsg();
  if (msg != NULL) pythonReturn(handle, PyLong_FromVoidPtr(msg));
}

void CpdPythonGroup::cpdCheck(void *m) {
  CkCcsRequestMsg *msg = (CkCcsRequestMsg *)m;
  //CkPrintf("[%d] CpdPythonGroup::cpdCheck reached\n",CkMyPe());
  PythonExecute *pyMsg = (PythonExecute *)msg->data;
  CmiUInt4 pyReference = prepareInterpreter(pyMsg);
  if (pyReference == 0) {
    CkPrintf("[%d] CpdPythonGroup::cpdCheck error while preparing interpreter\n",CkMyPe());
  }
  pyWorkers[pyReference].inUse = true;
  CmiReference(UsrToEnv(msg));
  CthResume(CthCreate((CthVoidFn)_callthr_executeThread, new CkThrCallArg(msg,(PythonObject*)this), 0));
  if (resultNotNone) CpdFreeze();
}

void CpdPythonGroup::registerPersistent(CkCcsRequestMsg *msg) {
  PythonAbstract *pyAbstract = (PythonAbstract *)msg->data;
  pyAbstract->unpack();
  if (! pyAbstract->isExecute()) return;
  PythonExecute *pyMsg = (PythonExecute *)pyAbstract;
  pyMsg->unpack();
  CmiUInt4 pyReference = prepareInterpreter(pyMsg);
  PyEval_ReleaseLock();
  replyIntFn(this, &msg->reply, &pyReference);
  if (pyReference == 0) return;
  pyMsg->setInterpreter(pyReference);
  PythonIterator *iter = pyMsg->info.info;
  int n = ntohl(((int*)iter)[1]);
  DebugPersistentCheck dpc(this, msg);
  for (int i=0; i<n; ++i) {
    int ep = ntohl(((int*)iter)[i+2]);
    CkPrintf("registering method for EP %d\n",ep);
    if (ep > 0) CkpvAccess(_debugEntryTable)[ep].postProcess.push_back(dpc);
    else CkpvAccess(_debugEntryTable)[-ep].preProcess.push_back(dpc);
  }
  CkPrintf("[%d] Registering Persistent method (reference=%d)\n",CkMyPe(),pyReference);
}

#include "charmdebug_python.def.h"
