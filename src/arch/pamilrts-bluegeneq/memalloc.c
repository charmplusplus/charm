
#include <converse.h>

#define ALIGNMENT        64
#define ALIGNMENT2       128
#define SMSG_SIZE        4096
#define N_SMSG_ELEM      512
#define MAX_SMSG_ELEM     4096
#define LMSG_SIZE        16384
#define N_LMSG_ELEM      128
#define MAX_LMSG_ELEM     2048

typedef struct CmiMemAllocHdr_bgq_t {
  int rank;
  int size;
  int tobuf;
  //Align the application buffer to 32 bytes
  char dummy[ALIGNMENT - sizeof(CmiChunkHeader) - 3*sizeof(int)];
} CmiMemAllocHdr_bgq;

typedef struct _memstruct {
    L2AtomicQueue memQ;
    int allocated_msg;
    //char pad[ALIGNMENT2 - sizeof(L2AtomicQueue) - sizeof(int)];
} L2MemStruct;

static int _nodeStart;
L2MemStruct *sL2MemallocVec;
L2MemStruct *bL2MemallocVec;

void *CmiAlloc_bgq (int size) {
  CmiMemAllocHdr_bgq *hdr = NULL;
  char *buf;
  
  int myrank = Kernel_ProcessorID() - _nodeStart;

  if (size <= SMSG_SIZE) {
    hdr = LRTSQueuePop(&(sL2MemallocVec[myrank].memQ));
    if (hdr == NULL) {
    if(sL2MemallocVec[myrank].allocated_msg > MAX_SMSG_ELEM) {
        hdr = (CmiMemAllocHdr_bgq *)memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));      
        hdr->tobuf = 0;
      } else {
        hdr = (CmiMemAllocHdr_bgq *) memalign(ALIGNMENT, SMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));      
        sL2MemallocVec[myrank].allocated_msg++;
        hdr->size = SMSG_SIZE;
        hdr->tobuf = 1;
      }
    }
  }
  else if (size <= LMSG_SIZE) {
    hdr = LRTSQueuePop(&(bL2MemallocVec[myrank].memQ));
    if (hdr == NULL) {      
      if(bL2MemallocVec[myrank].allocated_msg > MAX_LMSG_ELEM) {
        hdr = (CmiMemAllocHdr_bgq *)memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));      
        hdr->tobuf = 0;
      } else {
        hdr = (CmiMemAllocHdr_bgq *) memalign(ALIGNMENT, LMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));  
        bL2MemallocVec[myrank].allocated_msg++;
        hdr->size = LMSG_SIZE;
        hdr->tobuf = 1;
      }
    }
  }
  else {
    hdr = (CmiMemAllocHdr_bgq *) memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));
    hdr->size = size;
    hdr->tobuf  = 0;
  }

  hdr->rank = myrank;
  buf = (char*)hdr + sizeof(CmiMemAllocHdr_bgq);

  return buf;
}

void CmiFree_bgq (void *buf) {
  CmiMemAllocHdr_bgq *hdr = (CmiMemAllocHdr_bgq *)((char*)buf - sizeof(CmiMemAllocHdr_bgq));  
  int rc = L2A_EAGAIN;
  
   if (hdr->tobuf && hdr->size == SMSG_SIZE) 
     rc = LRTSQueuePush(&(sL2MemallocVec[hdr->rank].memQ), hdr);
   else if (hdr->tobuf && hdr->size == LMSG_SIZE)
     rc = LRTSQueuePush(&(bL2MemallocVec[hdr->rank].memQ), hdr);
 
   //queues are full or large buf
   if (rc == L2A_EAGAIN) {
     if(hdr->tobuf) {
      if(hdr->size == SMSG_SIZE)
        sL2MemallocVec[hdr->rank].allocated_msg--;
      else 
        bL2MemallocVec[hdr->rank].allocated_msg--;
     }
    free_nomigrate(hdr);
   }
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
  sL2MemallocVec = (L2MemStruct *)malloc_nomigrate(sizeof(L2MemStruct)*node_size);
  bL2MemallocVec = (L2MemStruct *)malloc_nomigrate(sizeof(L2MemStruct)*node_size);

  for (i = 0; i < node_size; ++i) {
    LRTSQueueInit ((char *)l2mem + 2*i*sizeof(L2AtomicState),
		       sizeof(L2AtomicState),
		       &(sL2MemallocVec[i].memQ),
		       0, /*No Overflow*/
		       N_SMSG_ELEM /*512 entries in short q*/);

    LRTSQueueInit ((char *)l2mem + (2*i+1)*sizeof(L2AtomicState),
		       sizeof(L2AtomicState),
		       &(bL2MemallocVec[i].memQ),
		       0,
           N_LMSG_ELEM /*128 entries in long q*/);
    sL2MemallocVec[i].allocated_msg = 0;
    bL2MemallocVec[i].allocated_msg = 0;
  }
}

