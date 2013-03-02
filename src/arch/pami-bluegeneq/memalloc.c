
#include <converse.h>

#define ALIGNMENT        64
#define SMSG_SIZE        4096
#define N_SMSG_ELEM      512
#define MAX_SMSG_ELEM     1024
#define LMSG_SIZE        16384
#define N_LMSG_ELEM      128
#define MAX_LMSG_ELEM     256

L2AtomicQueue *sL2MemallocVec;
L2AtomicQueue *bL2MemallocVec;

typedef struct CmiMemAllocHdr_bgq_t {
  int rank;
  int size;
  //Align the application buffer to 32 bytes
  int tobuf;
  char dummy[ALIGNMENT - sizeof(CmiChunkHeader) - 3*sizeof(int)];
} CmiMemAllocHdr_bgq;

typedef struct _AllocCount {
    int allocated_smsg;
    int allocated_lmsg;
    char pad[ALIGNMENT -  2*sizeof(int)];
} AllocCount;

AllocCount allocs[64];

static int _nodeStart;

void *CmiAlloc_bgq (int size) {
  CmiMemAllocHdr_bgq *hdr = NULL;
  char *buf;
  
  int myrank = Kernel_ProcessorID() - _nodeStart;
  if (size <= SMSG_SIZE) {
    hdr = L2AtomicDequeue (&sL2MemallocVec[myrank]);
    if (hdr == NULL) {
      if(allocs[myrank].allocated_smsg > MAX_SMSG_ELEM) {
        hdr = (CmiMemAllocHdr_bgq *)memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));      
        hdr->tobuf = 0;
      } else {
        hdr = (CmiMemAllocHdr_bgq *) memalign(ALIGNMENT, SMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));      
        allocs[myrank].allocated_smsg++;
        hdr->size = SMSG_SIZE;
        hdr->tobuf = 1;
      }
    }
  }
  else if (size <= LMSG_SIZE) {
    hdr = L2AtomicDequeue (&bL2MemallocVec[myrank]);
    if (hdr == NULL) {      
      if(allocs[myrank].allocated_lmsg > MAX_LMSG_ELEM) {
        hdr = (CmiMemAllocHdr_bgq *)memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));      
        hdr->tobuf = 0;
      } else {
        hdr = (CmiMemAllocHdr_bgq *) memalign(ALIGNMENT, LMSG_SIZE + sizeof(CmiMemAllocHdr_bgq));  
        allocs[myrank].allocated_lmsg++;
        hdr->size = LMSG_SIZE;
        hdr->tobuf = 1;
      }
    }
  }
  else {
    hdr = (CmiMemAllocHdr_bgq *)
      //malloc_nomigrate(size + sizeof(CmiMemAllocHdr_bgq));      
      memalign(ALIGNMENT, size + sizeof(CmiMemAllocHdr_bgq));
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
    rc = L2AtomicEnqueue (&sL2MemallocVec[hdr->rank], hdr);
  else if (hdr->tobuf && hdr->size == LMSG_SIZE)
    rc = L2AtomicEnqueue (&bL2MemallocVec[hdr->rank], hdr);

  //queues are full or large buf
  if (rc == L2A_EAGAIN) {
    if(hdr->tobuf) {
      if(hdr->size == SMSG_SIZE)
        allocs[hdr->rank].allocated_smsg--;
      else 
        allocs[hdr->rank].allocated_lmsg--;
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
  sL2MemallocVec = (L2AtomicQueue *)malloc_nomigrate(sizeof(L2AtomicQueue)*node_size);
  bL2MemallocVec = (L2AtomicQueue *)malloc_nomigrate(sizeof(L2AtomicQueue)*node_size);

  for (i = 0; i < node_size; ++i) {
    L2AtomicQueueInit ((char *)l2mem + 2*i*sizeof(L2AtomicState),
		       sizeof(L2AtomicState),
		       &sL2MemallocVec[i],
		       0, /*No Overflow*/
		       N_SMSG_ELEM /*512 entries in short q*/);

    L2AtomicQueueInit ((char *)l2mem + (2*i+1)*sizeof(L2AtomicState),
		       sizeof(L2AtomicState),
		       &bL2MemallocVec[i],
		       0,
		       N_LMSG_ELEM /*128 entries in long q*/);
    allocs[i].allocated_smsg = 0;
    allocs[i].allocated_lmsg = 0;
  }
}
