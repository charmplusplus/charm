/*This class is a wrapper class for binding
 * the data specific to one rdma operation*/

#ifndef __CKRDMAWRAPPER_H
#define __CKRDMAWRAPPER_H

#include "ckcallback.h"

#define rdma(...) CkRdmaWrapper(__VA_ARGS__)

class CkRdmaWrapper{
  public:
  void* ptr;
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
  CkRdmaWrapper(void *address) : ptr(address){
    srcPe = CkMyPe();
    callback = new CkCallback(CkCallback::ignore);
  }
  CkRdmaWrapper(void *address, CkCallback cb) : ptr(address) {
    srcPe = CkMyPe();
    callback = new CkCallback(cb);
  }
};
#endif
