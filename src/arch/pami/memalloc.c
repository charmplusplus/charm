
#include <converse.h>

#define ALIGNMENT        32
#define SMSG_SIZE        4096
#define N_SMSG_ELEM      4096
#define MMSG_SIZE        16384
#define N_MMSG_ELEM      2048
#define LLMSG_SIZE       65536
#define N_LLMSG_ELEM     1024

#if CMK_BLUEGENEQ
#include <spi/include/kernel/location.h>
#endif

PPCAtomicQueue *sPPCMemallocVec;
PPCAtomicQueue *mPPCMemallocVec;
PPCAtomicQueue *llPPCMemallocVec;

typedef struct CmiMemAllocHdr_ppcq_t {
  int rank;
  int size;
  //Align the application buffer to 32 bytes
  char dummy[ALIGNMENT - sizeof(CmiChunkHeader) - 2*sizeof(int)];
} CmiMemAllocHdr_ppcq;

static int _nodeStart;
extern int  Cmi_nodestart; /* First processor in this address space */

void *CmiAlloc_ppcq (int size) {
  CmiMemAllocHdr_ppcq *hdr = NULL;
  char *buf;
#if CMK_TRACE_ENABLED
  double start = CmiWallTimer();
#endif

#if CMK_BLUEGENEQ
  //Comm threads are hidden on BG/Q
  int myrank = Kernel_ProcessorID() - _nodeStart;
#else
  int myrank = CmiMyRank();
#endif

  if (size <= SMSG_SIZE) {
    hdr = PPCAtomicDequeue (&sPPCMemallocVec[myrank]);
    if (hdr == NULL)
      hdr = (CmiMemAllocHdr_ppcq *)
        malloc_nomigrate(SMSG_SIZE + sizeof(CmiMemAllocHdr_ppcq));
    hdr->size = SMSG_SIZE;
  }
  else if (size <= MMSG_SIZE) {
    hdr = PPCAtomicDequeue (&mPPCMemallocVec[myrank]);
    if (hdr == NULL)
      hdr = (CmiMemAllocHdr_ppcq *)
        malloc_nomigrate(MMSG_SIZE + sizeof(CmiMemAllocHdr_ppcq));
    hdr->size = MMSG_SIZE;
  }
  else if (size <= LLMSG_SIZE) {
    hdr = PPCAtomicDequeue (&llPPCMemallocVec[myrank]);
    if (hdr == NULL)
      hdr = (CmiMemAllocHdr_ppcq *)
        malloc_nomigrate(LLMSG_SIZE + sizeof(CmiMemAllocHdr_ppcq));
    hdr->size = LLMSG_SIZE;
  }
  else {
    hdr = (CmiMemAllocHdr_ppcq *)
      malloc_nomigrate(size + sizeof(CmiMemAllocHdr_ppcq));
    hdr->size = size;
  }

  hdr->rank = myrank;
  buf = (char*)hdr + sizeof(CmiMemAllocHdr_ppcq);

#if CMK_TRACE_ENABLED
  traceUserBracketEvent(30001, start, CmiWallTimer());
#endif

  return buf;
}

void CmiFree_ppcq (void *buf) {
  CmiMemAllocHdr_ppcq *hdr = (CmiMemAllocHdr_ppcq *)((char*)buf - sizeof(CmiMemAllocHdr_ppcq));
  int rc = CMI_PPCQ_EAGAIN;

#if CMK_TRACE_ENABLED
  double start = CmiWallTimer();
#endif

  if (hdr->size == SMSG_SIZE)
    rc = PPCAtomicEnqueue (&sPPCMemallocVec[hdr->rank], hdr);
  else if (hdr->size == MMSG_SIZE)
    rc = PPCAtomicEnqueue (&mPPCMemallocVec[hdr->rank], hdr);
  else if (hdr->size == LLMSG_SIZE)
    rc = PPCAtomicEnqueue (&llPPCMemallocVec[hdr->rank], hdr);

  if (rc == CMI_PPCQ_EAGAIN)
    //queues are full or large buf
    free_nomigrate(hdr);

#if CMK_TRACE_ENABLED
  traceUserBracketEvent(30002, start, CmiWallTimer());
#endif
}

void CmiMemAllocInit_ppcq (void   * atomic_mem,
			   size_t   atomic_memsize)
{
  int i = 0;
#if CMK_BLUEGENEQ
  int node_size = 64/Kernel_ProcessCount();
  _nodeStart = node_size * Kernel_MyTcoord();
#else
  int node_size = CmiMyNodeSize();
  _nodeStart = Cmi_nodestart;
#endif

  //We want to align headers to 32 bytes
  CmiAssert(sizeof(CmiMemAllocHdr_ppcq)+sizeof(CmiChunkHeader) == ALIGNMENT);

  CmiAssert (atomic_memsize >= 3 * node_size * sizeof(PPCAtomicState));
  sPPCMemallocVec = (PPCAtomicQueue *)malloc_nomigrate(sizeof(PPCAtomicQueue)*node_size);
  mPPCMemallocVec = (PPCAtomicQueue *)malloc_nomigrate(sizeof(PPCAtomicQueue)*node_size);
  llPPCMemallocVec = (PPCAtomicQueue *)malloc_nomigrate(sizeof(PPCAtomicQueue)*node_size);

  for (i = 0; i < node_size; ++i) {
    PPCAtomicQueueInit ((char *)atomic_mem + 3*i*sizeof(PPCAtomicState),
			sizeof(PPCAtomicState),
			&sPPCMemallocVec[i],
			0, /*No Overflow*/
			N_SMSG_ELEM );

    PPCAtomicQueueInit ((char *)atomic_mem + (3*i+1)*sizeof(PPCAtomicState),
			sizeof(PPCAtomicState),
			&mPPCMemallocVec[i],
			0,
			N_MMSG_ELEM );

    PPCAtomicQueueInit ((char *)atomic_mem + (3*i+2)*sizeof(PPCAtomicState),
			sizeof(PPCAtomicState),
			&llPPCMemallocVec[i],
			0,
			N_LLMSG_ELEM );
  }
}
