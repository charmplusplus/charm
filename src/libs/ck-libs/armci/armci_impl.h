#ifndef _ARMCI_IMPL_H
#define _ARMCI_IMPL_H

#include "tcharmc.h"
#include "tcharm.h"

#include "armci.decl.h"
#include "armci.h"

// structure definitions and forward declarations (for reductions)

typedef void* pointer;

typedef struct peAddr {
  int pe;
  pointer ptr;
} addressPair;

extern CkArrayID armciVPAid;

void mallocClient(void *param, int datasize, void *data);

// virtual processor class declaration
// ARMCI is supposed to be platform neutral, so calling this a thread did
// not seem like a proper abstraction.
class ArmciVirtualProcessor : public TCharmClient1D {
  CmiIsomallocBlockList *memBlock;
  CProxy_ArmciVirtualProcessor thisProxy;
  AddressMessage *addressReply;
 protected:
  virtual void setupThreadPrivate(CthThread forThread);
 public:
  ArmciVirtualProcessor(const CProxy_TCharm &_thr_proxy);
  ArmciVirtualProcessor(CkMigrateMessage *m);
  void getAddresses(AddressMessage *msg);
  void putData(int local, int nbytes, char *data, int sourceVP);
  void putAck(int sourceVP);
  void putFromGet(int local, int remote, int nbytes, int sourceVP);
  void putDataFromGet(int local, int nbytes, char *data);

  // non-entry methods. Mainly interfaces to API interface methods.
  int requestAddresses(pointer ptr_arr[], int bytes);
  int put(pointer local, pointer remote, int bytes, int destVP);
  int get(pointer remote, pointer local, int bytes, int destVP);

  virtual void pup(PUP::er &p);
};

class AddressMessage : public CMessage_AddressMessage {
 public:
  pointer *addresses;
  friend class CMessage_AddressMessage;
};

// pointer to the current tcshmem thread. Needed to regain context after
// getting called by user.
CtvStaticDeclare(ArmciVirtualProcessor *, _armci_ptr);

#endif
