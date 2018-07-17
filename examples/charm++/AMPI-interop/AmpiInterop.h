
typedef void (*LibMpiFn)(void*,void*);

#include "mpi.h"
#include "ampi-interoperate.h"

struct MpiCallData {

  static constexpr size_t size = sizeof(LibMpiFn)+2*sizeof(void*);

  LibMpiFn fn;
  void *in;
  void *out;
  CkCallback cb;
  unsigned char pup_dat[size];

  MpiCallData() {
  }

  ~MpiCallData() {
  }

  void pup(PUP::er &p) {
    if (p.isPacking()) {
      memcpy(pup_dat, &fn, sizeof(LibMpiFn));
      memcpy(pup_dat+sizeof(LibMpiFn), &in, sizeof(void*));
      memcpy(pup_dat+sizeof(LibMpiFn)+sizeof(void*), &out, sizeof(void*));
    }
    PUParray(p, pup_dat, size);
    if (p.isUnpacking()) {
      memcpy(&fn, pup_dat, sizeof(LibMpiFn));
      memcpy(&in, pup_dat+sizeof(LibMpiFn), sizeof(void*));
      memcpy(&out, pup_dat+sizeof(LibMpiFn)+sizeof(void*), sizeof(void*));
    }
    p|cb;
  }
};

#include "AmpiInterop.decl.h"
extern CProxy_AmpiInterop ampiInteropProxy;
void AmpiInteropInit();
