#include "armci_impl.h"

// Virtual Processor (thread) implementation

ArmciVirtualProcessor::ArmciVirtualProcessor(const CProxy_TCharm &_thr_proxy)
  : TCharmClient1D(_thr_proxy) {
  tcharmClientInit();
  memBlock = CmiIsomallocBlockListNew();
  thisProxy = CProxy_ArmciVirtualProcessor(thisArrayID);
  addressReply = NULL;
  thread->ready();
}

ArmciVirtualProcessor::ArmciVirtualProcessor(CkMigrateMessage *m) 
  : TCharmClient1D(m) 
{
  memBlock = NULL; //Paranoia-- makes sure we initialize this in pup
  thread = NULL;
  addressReply = NULL;
}

ArmciVirtualProcessor::~ArmciVirtualProcessor()
{
  CmiIsomallocBlockListDelete(memBlock);
  if (addressReply) {delete addressReply;}
}

void ArmciVirtualProcessor::setupThreadPrivate(CthThread forThread) {
  CtvAccessOther(forThread, _armci_ptr) = this;
}

void ArmciVirtualProcessor::getAddresses(AddressMessage *msg) {
  addressReply = msg;
  thread->resume();
}

void ArmciVirtualProcessor::putData(pointer local, int nbytes, char *data,
				    int sourceVP) {
  memcpy(local, data, nbytes);
  thisProxy[sourceVP].putAck(thisIndex);
}

void ArmciVirtualProcessor::putAck(int sourceVP) {
  thread->resume();
}

void ArmciVirtualProcessor::putFromGet(pointer local, pointer remote, int nbytes,
				       int sourceVP) {
  char *buffer;
  buffer = new char[nbytes];
  buffer = (char *)memcpy(buffer, local, nbytes);
  thisProxy[sourceVP].putDataFromGet(remote, nbytes, buffer);
}

// this is essentially the same as putData except that no acknowledgement
// is required and the thread suspended while waiting for the data is
// awoken.
void ArmciVirtualProcessor::putDataFromGet(pointer local, int nbytes, char *data) {
  memcpy(local, data, nbytes);
  thread->resume();
}

void ArmciVirtualProcessor::pup(PUP::er &p) {
  TCharmClient1D::pup(p);
  CmiIsomallocBlockListPup(&p, &memBlock);
  p|thisProxy;
  CkPupMessage(p, (void **)&addressReply, 1);
}

// NOT an entry method. This is an object-interface to the API interface.
int ArmciVirtualProcessor::requestAddresses(pointer ptr_arr[], int bytes) {
  int thisPE = armci_me;
  int numPE = armci_nproc;

  // reset the reply field
  addressReply = NULL;

  // my thread allocates memory using malloc (actually isomalloc)
  pointer ptr = (void *)CmiIsomallocBlockListMalloc(memBlock, bytes);
  addressPair *pair = new addressPair;
  pair->pe = thisPE;
  pair->ptr = ptr;

  // do a reduction to get everyone else's data.
  contribute(sizeof(addressPair), pair, CkReduction::concat);
  // wait for the reply to arrive.
  thread->suspend();

  // copy the acquired data to the user-allocated array.
  for (int i=0; i<numPE; i++) {
    ptr_arr[i] = addressReply->addresses[i];
  }
  delete addressReply;
  addressReply = NULL;

  // need to find out and return a proper error code if something bad happens
  return 0;
}

// implemented as a blocking put for now
int ArmciVirtualProcessor::put(pointer local, pointer remote,
			       int nbytes, int destVP) {
  char *buffer;
  buffer = new char[nbytes];
  buffer = (char *)memcpy(buffer, local, nbytes);

  thisProxy[destVP].putData(remote, nbytes, buffer, thisIndex);

  // blocking call. Wait for acknowledgement from target
  thread->suspend();

  // need to find out and return a proper error code if something bad happens
  return 0;
}

int ArmciVirtualProcessor::get(pointer remote, pointer local,
			       int nbytes, int destVP) {
  thisProxy[destVP].putFromGet(remote, local, nbytes, thisIndex);
  // wait for reply
  thread->suspend();

  // need to find out and return a proper error code if something bad happens
  return 0;
}
