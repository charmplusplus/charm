#ifndef _ARMCI_IMPL_H
#define _ARMCI_IMPL_H

#include "tcharmc.h"
#include "tcharm.h"

//Types needed for remote method parameters:
typedef void* pointer;
PUPmarshallBytes(pointer); //Pointers get sent as raw bytes

#include "armci.decl.h"
#include "armci.h"

// structure definitions and forward declarations (for reductions)
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
  ~ArmciVirtualProcessor();
  
  void getAddresses(AddressMessage *msg);
  void putData(pointer local, int nbytes, char *data, int sourceVP);
  void putAck(int sourceVP);
  void putFromGet(pointer local, pointer remote, int nbytes, int sourceVP);
  void putDataFromGet(pointer local, int nbytes, char *data);

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
CtvExtern(ArmciVirtualProcessor *, _armci_ptr);

#endif
