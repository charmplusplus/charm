
#include <converse.h>

#define ALIGNMENT        64
#define SMSG_SIZE        4096
#define N_SMSG_ELEM      512
#define LMSG_SIZE        16384
#define N_LMSG_ELEM      128

LRTSQueue sL2MemallocVec;
LRTSQueue bL2MemallocVec;

typedef struct CmiMemAllocHdr_bgq_t {
  int rank;
  int size;
  //Align the application buffer to 32 bytes
  char dummy[ALIGNMENT - sizeof(CmiChunkHeader) - 2*sizeof(int)];
} CmiMemAllocHdr_bgq;

static int _nodeStart;

void *CmiAlloc_bgq (int size) {
  CmiMemAllocHdr_bgq *hdr = NULL;
  char *buf;
  
  int myrank = Kernel_ProcessorID() - _nodeStart;

  if (size <= SMSG_SIZE) {
    hdr = LRTSQueuePop(&sL2MemallocVec[myrank]);
    if (hdr == NULL) 
      hdr = (CmiMemAllocHdr_bgq *)
	//malloc_nomigrate(SMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));      
	memalign(ALIGNMENT, SMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));      

    hdr->size = SMSG_SIZE;
  }
  else if (size <= LMSG_SIZE) {
    hdr = LRTSQueuePop(&bL2MemallocVec[myrank]);
    if (hdr == NULL) 
      hdr = (CmiMemAllocHdr_bgq *)
	//malloc_nomigrate(LMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));      
	memalign(ALIGNMENT, LMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));  
    hdr->size = LMSG_SIZE;
  }
  else {
    hdr = (CmiMemAllocHdr_bgq *)
      //malloc_nomigrate(size + sizeof(CmiMemAllocHdr_bgq));      
      memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));
    hdr->size = size;
  }

  hdr->rank = myrank;
  buf = (char*)hdr + sizeof(CmiMemAllocHdr_bgq);

  return buf;
}

void CmiFree_bgq (void *buf) {
  CmiMemAllocHdr_bgq *hdr = (CmiMemAllocHdr_bgq *)((char*)buf - sizeof(CmiMemAllocHdr_bgq));  
  int rc = L2A_EAGAIN;
  
  if (hdr->size == SMSG_SIZE) 
    rc = LRTSQueuePush(&(sL2MemallocVec[hdr->rank]), hdr);
  else if (hdr->size == LMSG_SIZE)
    rc = LRTSQueuePush(&(bL2MemallocVec[hdr->rank]), hdr);

  if (rc == L2A_EAGAIN)
    free_nomigrate(hdr);
}


void CmiMemAllocInit_bgq (void   * l2mem,
			  size_t   l2memsize) 
{
  int i = 0;
  int node_size = 64/Kernel_ProcessCount();
  _nodeStart = node_size * Kernel_MyTcoord();
  //We want to align headers to 32 bytes
  CmiAssert(sizeof(CmiMemAllocHdr_bgq)+sizeof(CmiChunkHeader) == ALIGNMENT);

  CmiAssert (l2memsize >= 2 * node_size * sizeof(L2AtomicState));
  sL2MemallocVec = (LRTSQueue)malloc_nomigrate(sizeof(struct _l2atomicq)*node_size);
  bL2MemallocVec = (LRTSQueue)malloc_nomigrate(sizeof(struct _l2atomicq)*node_size);

  for (i = 0; i < node_size; ++i) {
    LRTSQueueInit ((char *)l2mem + 2*i*sizeof(L2AtomicState),
		       sizeof(L2AtomicState),
		       &(sL2MemallocVec[i]),
		       0, /*No Overflow*/
		       N_SMSG_ELEM /*512 entries in short q*/);

    LRTSQueueInit ((char *)l2mem + (2*i+1)*sizeof(L2AtomicState),
		       sizeof(L2AtomicState),
		       &(bL2MemallocVec[i]),
		       0,
		       N_LMSG_ELEM /*128 entries in long q*/);
  }
}

