#include "PythonCCS-client.h"

PythonExecute::PythonExecute(char *_code, bool _persistent, bool _highlevel, char _interp[4]) {
  magic = sizeof(*this);
  codeLength = strlen(_code);
  code = strdup(_code);
  methodNameLength = 0;
  methodName = 0;
  infoSize = 0;
  info = 0;
  flags = 0;
  if (_persistent) {
    flags |= FLAG_PERSISTENT;
    flags |= FLAG_KEEPPRINT;
  }
  if (_highlevel) flags |= FLAG_HIGHLEVEL;
  if (_interp) memcpy(interpreter, _interp, 4);
  else memset(interpreter, 0, 4);
}

PythonExecute::PythonExecute(char *_code, char *_method, PythonIterator *_info, bool _persistent, bool _highlevel, char _interp[4]) {
  magic = sizeof(*this);
  codeLength = strlen(_code);
  code = strdup(_code);
  methodNameLength = strlen(_method);
  methodName = strdup(_method);
  infoSize = _info->size();
  info = _info->pack();
  flags = 0;
  if (_persistent) {
    flags |= FLAG_PERSISTENT;
    flags |= FLAG_KEEPPRINT;
  }
  if (_highlevel) flags |= FLAG_HIGHLEVEL;
  flags |= FLAG_ITERATE;
  if (_interp) memcpy(interpreter, _interp, 4);
  else memset(interpreter, 0, 4);
}

PythonExecute::~PythonExecute() {
  if (code) free(code);
  if (methodName) free(methodName);
  if (info) free(info);
}

void PythonExecute::setCode(char *_set) {
  codeLength = strlen(_set);
  code = strdup(_set);
}

void PythonExecute::setMethodName(char *_set) {
  methodNameLength = strlen(_set);
  methodName = strdup(_set);
}

void PythonExecute::setIterator(PythonIterator *_set) {
  infoSize = _set->size();
  info = _set->pack();
}

void PythonExecute::setPersistent(bool _set) {
  if (_set) flags |= FLAG_PERSISTENT;
  else flags &= ~FLAG_PERSISTENT;
}

void PythonExecute::setIterate(bool _set) {
  if (_set) flags |= FLAG_ITERATE;
  else flags &= ~FLAG_ITERATE;
}

void PythonExecute::setHighLevel(bool _set) {
  if (_set) flags |= FLAG_HIGHLEVEL;
  else flags &= ~FLAG_HIGHLEVEL;
}

void PythonExecute::setKeepPrint(bool _set) {
  if (_set) flags |= FLAG_KEEPPRINT;
  else flags &= ~FLAG_KEEPPRINT;
}

int PythonExecute::size() {
  return sizeof(PythonExecute)+codeLength+1+methodNameLength+1+infoSize;
}

char *PythonExecute::toString() {
  void *memory = malloc(size());
  memcpy (memory, this, sizeof(PythonExecute));
  char *ptr = (char*)memory+sizeof(PythonExecute);
  if (codeLength) {
    memcpy (ptr, code, codeLength+1);
    ptr += codeLength+1;
  }
  if (methodNameLength) {
    memcpy (ptr, methodName, methodNameLength+1);
    ptr += methodNameLength+1;
  }
  if (infoSize) {
    memcpy (ptr, info, infoSize);
  }
  return (char*)memory;
}

void PythonExecute::unpack() {
  if (codeLength) code = (char*)this + sizeof(PythonExecute);
  if (methodNameLength) methodName = code + codeLength+1;
  if (infoSize) info = (PythonIterator*) (methodName + methodNameLength+1);
}

PythonPrint::PythonPrint(char _interp[4], bool Wait) {
  magic = sizeof(*this);
  memcpy(interpreter, _interp, 4);
  if (Wait) flags |= FLAG_WAIT;
}

void PythonPrint::setWait(bool _set) {
  if (_set) flags |= FLAG_WAIT;
  else flags &= ~FLAG_WAIT;
}

/* Testing routine */
/*
int main () {
PythonExecute a("abcdefl");
printf("size: %d %d\n",sizeof(PythonExecute),a.size());
//a.print();
printf("ok %s\n",a.toString()+a.size());
return 0;
}
*/
