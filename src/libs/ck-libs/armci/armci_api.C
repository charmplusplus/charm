// APIs exposed in armci.h. These should be called from within the driver
// code "armciStart". Applications running the armci library MUST use
// the runtime option -memory isomalloc.

#include "armci_impl.h"

// API implementations

int armci_nproc;

// initialization api

// these are no-ops because of this implementation's associations with the
// TCharm initialization process.
int ARMCI_Init(void) {
  return 0;
}

int ARMCI_Finalize(void) {
  return 0;
}

// basic copy operations

// src is local memory, dst is remote address
int ARMCI_Put(void *src, void *dst, int bytes, int proc) {
  TCHARM_API_TRACE("ARMCI_Put", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  return vp->put(src, dst, bytes, proc);
}

// src is remote memory addr, dst is local address
int ARMCI_Get(void *src, void *dst, int bytes, int proc) {
  TCHARM_API_TRACE("ARMCI_Get", "armci");
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  return vp->get(src, dst, bytes, proc);
}

// strided copy operations

int ARMCI_PutS(void *src_ptr, int src_stride_ar[], 
	       void *dst_ptr, int dst_stride_ar[],
	       int count[], int stride_levels, int proc) {
  char *buffer;
  
  return 0;
}
	       

// global completion operations

// these are no-ops because Put is blocking
int ARMCI_Fence(int proc) {
  return 0;
}

int ARMCI_FenceAll(void) {
  return 0;
}

// memory operations

// malloc is a collective operation. The user is expected to allocate
// and manage ptr_arr.
int ARMCI_Malloc(void *ptr_arr[], int bytes) {
  TCHARM_API_TRACE("ARMCI_Malloc", "armci");
  // shift work off to entry method for split-phase communication.
  ArmciVirtualProcessor *vp = CtvAccess(_armci_ptr);
  // requestAddresses is called to bridge the split-phase gap at the
  // virtual processor object.
  return vp->requestAddresses(ptr_arr, bytes);
}

// CmiIsomalloc does not return a value and no indication is given about
// the success nor failure of the operation. Hence, it is assumed always
// that free works.
int ARMCI_Free(void *address) {
  TCHARM_API_TRACE("ARMCI_Free", "armci");
  CmiIsomallocBlockListFree(address);
  return 0;
}

// cleanup operations

void ARMCI_Cleanup(void) {
  // do nothing?
}

void ARMCI_Error(char *message, int code) {
  ckerr << "armci error: " << message << " | code = " << code << endl;
}
