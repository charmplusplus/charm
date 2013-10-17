#include "PythonCCS-client.h"

PythonExecute::PythonExecute(char *_code, bool _persistent, bool _highlevel, CmiUInt4 _interp) {
  magic = sizeof(*this) ^ localmagic;
  codeLength = strlen(_code);
  code.code = strdup(_code);
  methodNameLength = 0;
  methodName.methodName = 0;
  infoSize = 0;
  info.info = 0;
  flags = 0;
  if (_persistent) {
    flags |= FLAG_PERSISTENT;
    flags |= FLAG_KEEPPRINT;
  }
  if (_highlevel) flags |= FLAG_HIGHLEVEL;
  interpreter = _interp;
}

PythonExecute::PythonExecute(char *_code, char *_method, PythonIterator *_info, bool _persistent, bool _highlevel, CmiUInt4 _interp) {
  magic = sizeof(*this) ^ localmagic;
  codeLength = strlen(_code);
  code.code = strdup(_code);
  methodNameLength = strlen(_method);
  methodName.methodName = strdup(_method);
  infoSize = _info->size();
  info.info = (PythonIterator *)_info->pack();
  flags = 0;
  if (_persistent) {
    flags |= FLAG_PERSISTENT;
    flags |= FLAG_KEEPPRINT;
  }
  if (_highlevel) flags |= FLAG_HIGHLEVEL;
  flags |= FLAG_ITERATE;
  interpreter = _interp;
}

PythonExecute::~PythonExecute() {
  if (code.code) free(code.code);
  if (methodName.methodName) free(methodName.methodName);
  if (info.info) free(info.info);
}

void PythonExecute::setCode(char *_set) {
  codeLength = strlen(_set);
  code.code = strdup(_set);
}

void PythonExecute::setMethodName(char *_set) {
  methodNameLength = strlen(_set);
  methodName.methodName = strdup(_set);
}

void PythonExecute::setIterator(PythonIterator *_set) {
  infoSize = _set->size();
  info.info = (PythonIterator *)_set->pack();
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

void PythonExecute::setWait(bool _set) {
  if (_set) flags |= FLAG_WAIT;
  else flags &= ~FLAG_WAIT;
}

void PythonExecute::setNoCheck(bool _set) {
  if (_set) flags |= FLAG_NOCHECK;
  else flags &= ~FLAG_NOCHECK;
}

int PythonExecute::size() {
  return sizeof(PythonExecute)+codeLength+1+methodNameLength+1+infoSize;
}

char *PythonExecute::pack() {
  void *memory = malloc(size());
  memcpy (memory, (void*) this, sizeof(PythonExecute));
  char *ptr = (char*)memory+sizeof(PythonExecute);
  if (codeLength) {
    memcpy (ptr, code.code, codeLength+1);
    ptr += codeLength+1;
  }
  if (methodNameLength) {
    memcpy (ptr, methodName.methodName, methodNameLength+1);
    ptr += methodNameLength+1;
  }
  if (infoSize) {
    memcpy (ptr, info.info, infoSize);
  }
  // transform unsigned integers from host byte order to network byte order
  ((PythonAbstract*)memory)->magic = htonl(magic);
  //((PythonExecute*)memory)->interpreter = htonl(interpreter);
  ((PythonExecute*)memory)->codeLength = htonl(codeLength);
  ((PythonExecute*)memory)->methodNameLength = htonl(methodNameLength);
  ((PythonExecute*)memory)->infoSize = htonl(infoSize);

  return (char*)memory;
}

void PythonExecute::unpack() {
  // transform unsigned integers back to host byte order (from network byte order)
  interpreter = ntohl(interpreter);
  codeLength = ntohl(codeLength);
  methodNameLength = ntohl(methodNameLength);
  infoSize = ntohl(infoSize);
  if (codeLength) code.code = (char*)this + sizeof(PythonExecute);
  if (methodNameLength) methodName.methodName = code.code + codeLength+1;
  if (infoSize) info.info = (PythonIterator*) (methodName.methodName + methodNameLength+1);
}

void PythonAbstract::unpack() {
  magic = ntohl(magic);
}

bool PythonAbstract::isFinished() {
  return (magic == (sizeof(PythonFinished) ^ PythonFinished::localmagic));
}

bool PythonAbstract::isExecute() {
  return (magic == (sizeof(PythonExecute) ^ PythonExecute::localmagic));
}

bool PythonAbstract::isPrint() {
  return (magic == (sizeof(PythonPrint) ^ PythonPrint::localmagic));
}

PythonPrint::PythonPrint(CmiUInt4 _interp, bool Wait, bool Kill) {
  magic = sizeof(*this) ^ localmagic;
  interpreter = _interp;
  flags = 0;
  if (Wait) flags |= FLAG_WAIT;
  if (Kill) flags |= FLAG_KILL;
}

void PythonPrint::setWait(bool _set) {
  if (_set) flags |= FLAG_WAIT;
  else flags &= ~FLAG_WAIT;
}

void PythonPrint::setKill(bool _set) {
  if (_set) flags |= FLAG_KILL;
  else flags &= ~FLAG_KILL;
}

PythonFinished::PythonFinished(CmiUInt4 _interp, bool Wait) {
  magic = sizeof(*this) ^ localmagic;
  interpreter = _interp;
  flags = 0;
  if (Wait) flags |= FLAG_WAIT;
}

void PythonFinished::setWait(bool _set) {
  if (_set) flags |= FLAG_WAIT;
  else flags &= ~FLAG_WAIT;
}

void PythonExecute::print() {
  printf("magic %d, interpreter %d, flags 0x%x\n",magic ,interpreter, flags);
}

void PythonPrint::print() {
  printf("magic %d, interpreter %d, flags 0x%x\n",magic, interpreter, flags);
}

/* Testing routine */
/*int main () {
PythonExecute a("abcdefl");
printf("size: %d %d\n",sizeof(PythonExecute),a.size());
//a.print();
printf("ok %s, %d\n",a.pack()+sizeof(a), strlen("abcdefl"));
return 0;
}*/
