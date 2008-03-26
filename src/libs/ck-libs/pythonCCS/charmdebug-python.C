#include "charmdebug_python.decl.h"

//CpvExtern(CkHashtable_c, ccsTab);
class CpdPython : public CBase_CpdPython {
public:
  CpdPython (CkArgMsg *msg) {
    ((CProxy_CpdPython)thishandle).registerPython("CpdPython");
    //CkCallback cb(CkIndex_CpdPython::pyRequest(0),thishandle);
    //CcsRegisterHandler("pycode", cb);
    CProxy_CpdPythonGroup group = CProxy_CpdPythonGroup::ckNew();
    group.registerPython("CpdPythonGroup");
    //CcsRegisterHandler("CpdPythonGroup", CkCallback(CkIndex_CpdPythonGroup::pyRequest(0),group));
    //CkPrintf("CpdPython registered\n");
    //char *string = "pycode";
    //CkAssert(CkHashtableGet(CpvAccess(ccsTab),(void *)&string) != NULL);
  }
  void get(int handle) {
    CkPrintf("CpdPython::get\n");
  }
};

class CpdPythonGroup : public CBase_CpdPythonGroup {
public:
  CpdPythonGroup() {
    //CkPrintf("[%d] CpdPythonGroup::constructor\n",CkMyPe());
  }

  int buildIterator(PyObject*&, void*);
  int nextIteratorUpdate(PyObject*&, PyObject*, void*);
  
  void getArray(int handle);
  void getValue(int handle);
  void getCast(int handle);
  void getStatic(int handle);
};

int CpdPythonGroup::buildIterator(PyObject *&data, void *iter) {
  int group = ntohl(*(int*)iter);
  CkGroupID id;
  id.idx = group;
  void *ptr = CkLocalBranch(id);
  data = PyLong_FromVoidPtr(ptr);
  CkPrintf("[%d] Building iterator for %i: %p\n", CkMyPe(), group, ptr);
  return 1;
}

int CpdPythonGroup::nextIteratorUpdate(PyObject *&data, PyObject *result, void *iter) {
  CkPrintf("[%d] Asked for next iterator\n",CkMyPe());
  return 0;
}

void CpdPythonGroup::getArray(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *obj, *type;
  int num, size;
  if (PyArg_ParseTuple(arg, "OOii", &obj, &type, &num, &size) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(obj);
  ptr += num * size;
  pythonReturn(handle, PyLong_FromVoidPtr(ptr));
}

void CpdPythonGroup::getValue(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *obj, *type;
  int offset;
  char *name, restype;
  if (PyArg_ParseTuple(arg, "OOsic", &obj, &type, &name, &offset, &restype) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(obj);
  ptr += offset;
  PyObject *result = NULL;
  switch (restype) {
  case 'p':
    result = PyLong_FromVoidPtr(ptr);
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
  pythonReturn(handle, result);
}

void CpdPythonGroup::getCast(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *obj, *type, *newtype;
  int offset;
  if (PyArg_ParseTuple(arg, "OOOi", &obj, &type, &newtype, &offset) == 0) return;
  char *ptr = (char*)PyLong_AsVoidPtr(obj);
  ptr += offset;
  pythonReturn(handle, PyLong_FromVoidPtr(ptr));
}

void CpdPythonGroup::getStatic(int handle) {
  PyObject *arg = pythonGetArg(handle);
  PyObject *location;
  char *name, restype;
  CkPrintf("Parsing arguments\n");
  if (PyArg_ParseTuple(arg, "sOc", &name, &location, &restype) == 0) return;
  CkPrintf("Arguments parsed\n");
  char *ptr = (char*)PyLong_AsVoidPtr(location);
  CkPrintf("Pointer: %p",ptr);
  PyObject *result = NULL;
  switch (restype) {
  case 'p':
    result = PyLong_FromVoidPtr(ptr);
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
  pythonReturn(handle, result);
}

#include "charmdebug_python.def.h"
