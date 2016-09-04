/*This class is a wrapper class for binding
 * the data specific to one zero copy operation*/

#ifndef __CKRDMAWRAPPER_H
#define __CKRDMAWRAPPER_H

#include "ckcallback.h"

#define CkSendBuffer(...) CkRdmaWrapper(__VA_ARGS__)

class CkRdmaWrapper{
  public:
  const void* ptr;
  CkCallback *callback;

  //this field is used for handling rdma acks called by comm thread
  //the comm thread calls the callbackgroup to call the callback on the
  //appropriate pe.
  int srcPe;
  size_t cnt;

  void pup(PUP::er &p){
    pup_bytes(&p, this, sizeof(CkRdmaWrapper));
  }

  CkRdmaWrapper() : ptr(NULL), callback(NULL) {
    srcPe = -1;
  }
  explicit CkRdmaWrapper(const void *address) : ptr(address){
    srcPe = CkMyPe();
    callback = new CkCallback(CkCallback::ignore);
  }
  CkRdmaWrapper(const void *address, CkCallback cb) : ptr(address) {
    srcPe = CkMyPe();
    callback = new CkCallback(cb);
  }
};

struct CkRdmaPostStruct{
  void *ptr;
  size_t cnt;
};

struct CkRdmaPostHandle{
  void *msg; //msg, not envelope
  int nstructs;
  CkRdmaPostStruct structs[0];
};

#endif
