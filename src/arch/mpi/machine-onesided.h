//List of rdma receiver buffer information that is used for waiting for completion
void MPIPostOneBuffer(const void *buffer, void *ref, int size, int pe, int tag, int type);

int getNewMPITag(void){

  /* A local variable is used to avoid a race condition that can occur when
   * the global variable rdmaTag is updated by another thread after the first
   * thread releases the lock but before it sends the variable back */
  int newRdmaTag;
#if CMK_SMP
  CmiLock(rdmaTagLock);
#endif

  rdmaTag++;

  /* Reset generated tag when equal to the implementation dependent upper bound.
   * This condition also ensures correct resetting of the generated tag if tagUb is INT_MAX */
  if(rdmaTag == tagUb)
    rdmaTag = RDMA_BASE_TAG; //reseting can fail if previous tags are in use

  //copy the updated value into the local variable to ensure consistent a tag value
  newRdmaTag = rdmaTag;
#if CMK_SMP
  CmiUnlock(rdmaTagLock);
#endif
  return newRdmaTag;
}

// Set the machine specific information for a nocopy pointer (Empty method to maintain API consistency)
void LrtsSetRdmaBufferInfo(void *info, const void *ptr, int size, unsigned short int mode){
}
