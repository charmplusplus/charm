#include "armci_impl.h"

// This is the way to adapt a library's preferred start interface with the
// one provided by TCharm (eg. argc,argv vs void).
extern "C" void armciLibStart(void) {
  int argc=CkGetArgc();
  char **argv=CkGetArgv();
  armciStart(argc,argv);
}

// Default startup routine (can be overridden by user's own)
// This will be registered with TCharm's startup routine
// in the Node initialization function.
static void ArmciDefaultSetup(void) {
  // Create the base threads on TCharm using user-defined start routine.
  TCHARM_Create(TCHARM_Get_num_chunks(), armciLibStart);
}

CtvDeclare(ArmciVirtualProcessor *, _armci_ptr);

// Node initialization (made by initcall of the module armci)
void armciProcInit(void) {
  CtvInitialize(ArmciVirtualProcessor, _armci_ptr);
  CtvAccess(_armci_ptr) = NULL;

  // Register the library's default startup routine to TCharm
  TCHARM_Set_fallback_setup(ArmciDefaultSetup);
};

ArmciVirtualProcessor::ArmciVirtualProcessor(const CProxy_TCharm &_thr_proxy)
  : TCharmClient1D(_thr_proxy) {
  tcharmClientInit();
  memBlock = CmiIsomallocBlockListNew();
  thisProxy = CProxy_ArmciVirtualProcessor(thisArrayID);
  addressReply = NULL;
  // Save ourselves for the waiting ARMCI_Init
  thread->semaPut(ARMCI_TCHARM_SEMAID,this);
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
  armci_nproc = thread->getNumElements();
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

void *stridedCopy(void *base, void *buffer_ptr,
		  int *stride, int *count, 
		  int dim_id, bool flatten) {
  if (dim_id == 0) {
    if (flatten) {
      memcpy(buffer_ptr, base, count[dim_id]);
    } else {
      memcpy(base, buffer_ptr, count[dim_id]);
    }
    return (void *)((char *)buffer_ptr + count[dim_id]);
  } else {
    for (int i=0; i<count[dim_id]; i++) {
      buffer_ptr = stridedCopy(base, buffer_ptr, stride, 
			       count, dim_id-1, flatten);
      base = (void *)((char *)base + stride[dim_id-1]);
    }
    return buffer_ptr;
  }
}

// malloc reduction client
void mallocClient(void *param, int datasize, void *data) {
  // get the array proxy from the supplied aid value (at setup)
  CProxy_ArmciVirtualProcessor vpProxy = 
    CProxy_ArmciVirtualProcessor(*(CkArrayID *)param);
  int numBlocks = datasize/sizeof(addressPair);
  addressPair *dataBlocks = (addressPair *)data;

  AddressMessage *msg = new(numBlocks, 0) AddressMessage;
  // constructing the ordered set of shared pointers
  for (int i=0; i<numBlocks; i++) {
    msg->addresses[dataBlocks[i].pe] = dataBlocks[i].ptr;
    /*
    ckout << "dest" << i << ": " << dataBlocks[i].pe 
	  << "->" << dataBlocks[i].ptr << endl;
    */
  }
  // broadcast the results to everyone.
  vpProxy.getAddresses(msg);
}

#include "armci.def.h"

