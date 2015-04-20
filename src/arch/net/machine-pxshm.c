/** @file
                        size = CmiMsgHeaderGetLength(msg);
 * pxshm --> posix shared memory based network layer for communication
 * between processes on the same node
 * This is not going to be the primary mode of communication
 * but only for messages below a certain size between
 * processes on the same node
 * for non-smp version only
 * * @ingroup NET
 * contains only pxshm code for
 * - CmiInitPxshm()
 * - DeliverViaPxShm()
 * - CommunicationServerPxshm()
 * - CmiMachineExitPxshm()


There are three options here for synchronization:
      PXSHM_FENCE is the default. It uses memory fences
      PXSHM_OSSPINLOCK will cause OSSpinLock's to be used (available on OSX)
      PXSHM_LOCK will cause POSIX semaphores to be used

  created by
        Sayantan Chakravorty, sayantan@gmail.com ,21st March 2007
*/

/**
 * @addtogroup NET
 * @{
 */

#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

/**************
   Determine which type of synchronization to use
*/
#if PXSHM_OSSPINLOCK
#include <libkern/OSAtomic.h>
#elif PXSHM_LOCK
#include <semaphore.h>
#else
/* Default to using fences */
#define PXSHM_FENCE 1
#endif

#define MEMDEBUG(x) // x

#define PXSHM_STATS 0

/*** The following code was copied verbatim from pcqueue.h file ***/
#if PXSHM_FENCE
#ifdef POWER_PC
#define CmiMemoryWriteFence(startPtr, nBytes) asm volatile("eieio" ::: "memory")
#else
#define CmiMemoryWriteFence(startPtr, nBytes)                                  \
  asm volatile("sfence" ::: "memory")
//#define CmiMemoryWriteFence(startPtr,nBytes)
#endif
#else
#define CmiMemoryWriteFence(startPtr, nBytes)
#endif

#if PXSHM_FENCE
#ifdef POWER_PC
#define CmiMemoryReadFence(startPtr, nBytes) asm volatile("eieio" ::: "memory")
#else
#define CmiMemoryReadFence(startPtr, nBytes) asm volatile("lfence" ::: "memory")
//#define CmiMemoryReadFence(startPtr,nBytes)
#endif
#else
#define CmiMemoryReadFence(startPtr, nBytes)
#endif

/***************************************************************************************/

enum entities { SENDER, RECEIVER };

/************************
 * 	Implementation currently assumes that
 * 	1) all nodes have the same number of processors
 *  2) in the nodelist all processors in a node are listed in sequence
 *   0 1 2 3      4 5 6 7
 *   -------      -------
 *    node 1       node 2
 ************************/

#define NAMESTRLEN 50
#define PREFIXSTRLEN 30

#define SHMBUFLEN 1000000

#define SENDQSTARTSIZE 128

/// This struct is used as the first portion of a shared memory region, followed
/// by data
typedef struct {
  int count; // number of messages
  int bytes; // number of bytes

#if PXSHM_OSSPINLOCK
  OSSpinLock lock;
#endif

#if PXSHM_FENCE
  volatile int flagSender;
  volatile int flagReceiver;
  volatile int turn;
#endif

} sharedBufHeader;

typedef struct {
#if PXSHM_LOCK
  sem_t *mutex;
#endif
  sharedBufHeader *header;
  char *data;
} sharedBufData;

typedef struct {
  int size; // total size of data array
  int begin; // position of first element
  int end; // position of next element
  int numEntries; // number of entries

  OutgoingMsg *data;

} PxshmSendQ;

typedef struct {
  int nodesize;
  int noderank;
  int nodestart, nodeend; // proc numbers for the start and end of this node
  char prefixStr[PREFIXSTRLEN];
  char **recvBufNames;
  char **sendBufNames;

  sharedBufData *recvBufs;
  sharedBufData *sendBufs;

  PxshmSendQ **sendQs;

#if PXSHM_STATS
  int sendCount;
  int validCheckCount;
  int lockRecvCount;
  double validCheckTime;
  double sendTime;
  double commServerTime;
#endif

} PxshmContext;

PxshmContext *pxshmContext = NULL; // global context

void calculateNodeSizeAndRank();
void setupSharedBuffers();
void initAllSendQs();

/******************
 * 	Initialization routine
 * 	currently just testing start up
 * ****************/
void CmiInitPxshm(char **argv)
{
  MACHSTATE(3, "CminitPxshm start");
  pxshmContext = (PxshmContext *) malloc(sizeof(PxshmContext));

  if (Cmi_charmrun_pid <= 0) {
    CmiAbort("pxshm must be run with charmrun");
  }
  CmiDeprecateArgInt(argv, "+nodesize", "Number of cores in this node",
                     "Charmrun> Deprecation warning: charmrun now figures "
                     "out the nodesize on its own.");

  calculateNodeSizeAndRank();
  if (pxshmContext->nodesize == 1) {
    return;
  }

  MACHSTATE1(3, "CminitPxshm  %d calculateNodeSizeAndRank",
             pxshmContext->nodesize);

  snprintf(&(pxshmContext->prefixStr[0]), PREFIXSTRLEN - 1, "charm_pxshm_%d",
           Cmi_charmrun_pid);

  MACHSTATE2(3, "CminitPxshm %s %d pre setupSharedBuffers",
             pxshmContext->prefixStr, pxshmContext->nodesize);

  setupSharedBuffers();

  MACHSTATE2(3, "CminitPxshm %s %d setupSharedBuffers", pxshmContext->prefixStr,
             pxshmContext->nodesize);

  initAllSendQs();

  MACHSTATE2(3, "CminitPxshm %s %d initAllSendQs", pxshmContext->prefixStr,
             pxshmContext->nodesize);

  MACHSTATE2(3, "CminitPxshm %s %d done", pxshmContext->prefixStr,
             pxshmContext->nodesize);

#if PXSHM_STATS
  pxshmContext->sendCount = 0;
  pxshmContext->sendTime = 0.0;
  pxshmContext->validCheckCount = 0;
  pxshmContext->validCheckTime = 0.0;
  pxshmContext->commServerTime = 0;
  pxshmContext->lockRecvCount = 0;
#endif
};

/**************
 * shutdown shmem objects and semaphores
 *
 * *******************/
void tearDownSharedBuffers();

void CmiExitPxshm()
{
  int i = 0;

  if (pxshmContext->nodesize != 1) {
    tearDownSharedBuffers();

    for (i = 0; i < pxshmContext->nodesize; i++) {
      if (i != pxshmContext->noderank) {
        break;
      }
      free(pxshmContext->recvBufNames[i]);
      free(pxshmContext->sendBufNames[i]);
    }
    free(pxshmContext->recvBufNames);
    free(pxshmContext->sendBufNames);

    free(pxshmContext->recvBufs);
    free(pxshmContext->sendBufs);
  }
#if PXSHM_STATS
  CmiPrintf("[%d] sendCount %d sendTime %6lf validCheckCount %d validCheckTime "
            "%.6lf commServerTime %6lf lockRecvCount %d \n",
            _Cmi_mynode, pxshmContext->sendCount, pxshmContext->sendTime,
            pxshmContext->validCheckCount, pxshmContext->validCheckTime,
            pxshmContext->commServerTime, pxshmContext->lockRecvCount);
#endif
  free(pxshmContext);
}

/******************
 *Should this message be sent using PxShm or not ?
 * ***********************/

inline int CmiValidPxshm(OutgoingMsg ogm, OtherNode node)
{
#if PXSHM_STATS
  pxshmContext->validCheckCount++;
#endif

  /*	if(pxshmContext->nodesize == 1){
                  return 0;
          }*/
  // replace by bitmap later
  if (ogm->dst >= pxshmContext->nodestart &&
      ogm->dst <= pxshmContext->nodeend && ogm->size < SHMBUFLEN) {
    return 1;
  } else {
    return 0;
  }
};

inline int PxshmRank(int dst) { return dst - pxshmContext->nodestart; }
inline void pushSendQ(PxshmSendQ *q, OutgoingMsg msg);
inline int sendMessage(OutgoingMsg ogm, sharedBufData *dstBuf,
                       PxshmSendQ *dstSendQ);
inline int flushSendQ(int dstRank);

/***************
 *
 *Send this message through shared memory
 *if you cannot get lock, put it in the sendQ
 *Before sending messages pick them from sendQ
 *
 * ****************************/

void CmiSendMessagePxshm(OutgoingMsg ogm, OtherNode node, int rank,
                         unsigned int broot)
{
  sharedBufData *dstBuf;

#if PXSHM_STATS
  double _startSendTime = CmiWallTimer();
#endif

  int dstRank = PxshmRank(ogm->dst);
  MEMDEBUG(CmiMemoryCheck());

  DgramHeaderMake(ogm->data, rank, ogm->src, Cmi_charmrun_pid, 1, broot);

  MACHSTATE4(3, "Send Msg Pxshm ogm %p size %d dst %d dstRank %d", ogm,
             ogm->size, ogm->dst, dstRank);

  CmiAssert(dstRank >= 0 && dstRank != pxshmContext->noderank);

  dstBuf = &(pxshmContext->sendBufs[dstRank]);

#if PXSHM_OSSPINLOCK
  if (!OSSpinLockTry(&dstBuf->header->lock)) {
#elif PXSHM_LOCK
  if (sem_trywait(dstBuf->mutex) < 0) {
#elif PXSHM_FENCE
  dstBuf->header->flagSender = 1;
  dstBuf->header->turn = RECEIVER;
  CmiMemoryReadFence(0, 0);
  CmiMemoryWriteFence(0, 0);
  if (dstBuf->header->flagReceiver && dstBuf->header->turn == RECEIVER) {
    dstBuf->header->flagSender = 0;
#endif

    /**failed to get the lock
    insert into q and retain the message*/

    pushSendQ(pxshmContext->sendQs[dstRank], ogm);
    ogm->refcount++;
    MEMDEBUG(CmiMemoryCheck());
    return;
  } else {

    /***
     * We got the lock for this buffer
     * first write all the messages in the sendQ and then write this guy
     * */
    if (pxshmContext->sendQs[dstRank]->numEntries == 0) {
      // send message user event
      int ret = sendMessage(ogm, dstBuf, pxshmContext->sendQs[dstRank]);
      MACHSTATE(3, "Pxshm Send succeeded immediately");
    } else {
      int sent;
      ogm->refcount +=
          2; /*this message should not get deleted when the queue is flushed*/
      pushSendQ(pxshmContext->sendQs[dstRank], ogm);
      MACHSTATE3(3, "Pxshm ogm %p pushed to sendQ length %d refcount %d", ogm,
                 pxshmContext->sendQs[dstRank]->numEntries, ogm->refcount);
      sent = flushSendQ(dstRank);
      ogm->refcount--; /*if it has been sent, can be deleted by caller, if not
                          will be deleted when queue is flushed*/
      MACHSTATE1(3, "Pxshm flushSendQ sent %d messages", sent);
    }
/* unlock the recvbuffer*/

#if PXSHM_OSSPINLOCK
    OSSpinLockUnlock(&dstBuf->header->lock);
#elif PXSHM_LOCK
    sem_post(dstBuf->mutex);
#elif PXSHM_FENCE
    CmiMemoryReadFence(0, 0);
    CmiMemoryWriteFence(0, 0);
    dstBuf->header->flagSender = 0;
#endif
  }
#if PXSHM_STATS
  pxshmContext->sendCount++;
  pxshmContext->sendTime += (CmiWallTimer() - _startSendTime);
#endif
  MEMDEBUG(CmiMemoryCheck());
};

inline void emptyAllRecvBufs();
inline void flushAllSendQs();

/**********
 * Extract all the messages from the recvBuffers you can
 * Flush all sendQs
 * ***/
inline void CommunicationServerPxshm()
{

#if PXSHM_STATS
  double _startCommServerTime = CmiWallTimer();
#endif

  MEMDEBUG(CmiMemoryCheck());
  emptyAllRecvBufs();
  flushAllSendQs();

#if PXSHM_STATS
  pxshmContext->commServerTime += (CmiWallTimer() - _startCommServerTime);
#endif

  MEMDEBUG(CmiMemoryCheck());
};

static void CmiNotifyStillIdlePxshm(CmiIdleState *s)
{
  CommunicationServerPxshm();
}

static void CmiNotifyBeginIdlePxshm(CmiIdleState *s) { CmiNotifyStillIdle(s); }

void calculateNodeSizeAndRank()
{
  pxshmContext->nodesize = 1;
  MACHSTATE(3, "calculateNodeSizeAndRank start");
  // CmiGetArgIntDesc(argv, "+nodesize", &(pxshmContext->nodesize),"Number of
  // cores in this node (for non-smp case).Used by the shared memory
  // communication layer");
  pxshmContext->nodesize = _Cmi_myphysnode_numprocesses;
  MACHSTATE1(3, "calculateNodeSizeAndRank argintdesc %d",
             pxshmContext->nodesize);

  pxshmContext->noderank = _Cmi_mynode % (pxshmContext->nodesize);

  MACHSTATE1(3, "calculateNodeSizeAndRank noderank %d", pxshmContext->noderank);

  pxshmContext->nodestart = _Cmi_mynode - pxshmContext->noderank;

  MACHSTATE(3, "calculateNodeSizeAndRank nodestart ");

  pxshmContext->nodeend = pxshmContext->nodestart + pxshmContext->nodesize - 1;

  if (pxshmContext->nodeend >= _Cmi_numnodes) {
    pxshmContext->nodeend = _Cmi_numnodes - 1;
    pxshmContext->nodesize =
        (pxshmContext->nodeend - pxshmContext->nodestart) + 1;
  }

  MACHSTATE3(3, "calculateNodeSizeAndRank nodestart %d nodesize %d noderank %d",
             pxshmContext->nodestart, pxshmContext->nodesize,
             pxshmContext->noderank);
}

void allocBufNameStrings(char ***bufName);
void createShmObjectsAndSems(sharedBufData **bufs, char **bufNames);
/***************
 * 	calculate the name of the shared objects and semaphores
 *
 * 	name scheme
 * 	shared memory: charm_pxshm_<recvernoderank>_<sendernoderank>
 *  semaphore    : charm_pxshm_<recvernoderank>_<sendernoderank>.sem for
 *semaphore for that shared object
 *                the semaphore name used by us is the same as the shared memory
 *object name
 *                the posix library adds the semaphore tag // in linux at least
 *. other machines might need more portable code
 *
 * 	open these shared objects and semaphores
 * *********/
void setupSharedBuffers()
{
  int i = 0;

  allocBufNameStrings(&(pxshmContext->recvBufNames));

  MACHSTATE(3, "allocBufNameStrings for recvBufNames done");
  MEMDEBUG(CmiMemoryCheck());

  allocBufNameStrings((&pxshmContext->sendBufNames));

  MACHSTATE(3, "allocBufNameStrings for sendBufNames done");

  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      snprintf(pxshmContext->recvBufNames[i], NAMESTRLEN - 1, "%s_%d_%d",
               pxshmContext->prefixStr,
               pxshmContext->noderank + pxshmContext->nodestart,
               i + pxshmContext->nodestart);
      MACHSTATE2(3, "recvBufName %s with rank %d",
                 pxshmContext->recvBufNames[i], i)
      snprintf(pxshmContext->sendBufNames[i], NAMESTRLEN - 1, "%s_%d_%d",
               pxshmContext->prefixStr, i + pxshmContext->nodestart,
               pxshmContext->noderank + pxshmContext->nodestart);
      MACHSTATE2(3, "sendBufName %s with rank %d",
                 pxshmContext->sendBufNames[i], i);
    }
  }

  createShmObjectsAndSems(&(pxshmContext->recvBufs),
                          pxshmContext->recvBufNames);
  createShmObjectsAndSems(&(pxshmContext->sendBufs),
                          pxshmContext->sendBufNames);

  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      CmiAssert(pxshmContext->sendBufs[i].header->count == 0);
      pxshmContext->sendBufs[i].header->count = 0;
      pxshmContext->sendBufs[i].header->bytes = 0;
    }
  }
}

void allocBufNameStrings(char ***bufName)
{
  int i, count;

  int totalAlloc = sizeof(char) * NAMESTRLEN * (pxshmContext->nodesize - 1);
  char *tmp = malloc(totalAlloc);

  MACHSTATE2(3, "allocBufNameStrings tmp %p totalAlloc %d", tmp, totalAlloc);

  *bufName = (char **) malloc(sizeof(char *) * pxshmContext->nodesize);

  for (i = 0, count = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      (*bufName)[i] = &(tmp[count * NAMESTRLEN * sizeof(char)]);
      count++;
    } else {
      (*bufName)[i] = NULL;
    }
  }
}

void createShmObject(char *name, int size, char **pPtr);

void createShmObjectsAndSems(sharedBufData **bufs, char **bufNames)
{
  int i = 0;

  *bufs =
      (sharedBufData *) malloc(sizeof(sharedBufData) * pxshmContext->nodesize);

  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      createShmObject(bufNames[i], SHMBUFLEN + sizeof(sharedBufHeader),
                      (char **) &((*bufs)[i].header));
      (*bufs)[i].data =
          ((char *) ((*bufs)[i].header)) + sizeof(sharedBufHeader);
#if PXSHM_OSSPINLOCK
      (*bufs)[i].header->lock =
          0; // by convention(see man page) 0 means unlocked
#elif PXSHM_LOCK
      (*bufs)[i].mutex = sem_open(bufNames[i], O_CREAT, S_IRUSR | S_IWUSR, 1);
#endif
    } else {
      (*bufs)[i].header = NULL;
      (*bufs)[i].data = NULL;
#if PXSHM_LOCK
      (*bufs)[i].mutex = NULL;
#endif
    }
  }
}

void createShmObject(char *name, int size, char **pPtr)
{
  int fd = -1;
  int flags; // opening flags for shared object
  int open_repeat_count = 0;

  flags =
      O_RDWR |
      O_CREAT; // open file in read-write mode and create it if its not there

  while (fd < 0 && open_repeat_count < 100) {
    open_repeat_count++;
    fd = shm_open(name, flags, S_IRUSR | S_IWUSR); // create the shared object
                                                   // with permissions for only
                                                   // the user to read and write

    if (fd < 0 && open_repeat_count > 10) {
      fprintf(stderr, "Error(attempt=%d) from shm_open %s while opening %s \n",
              open_repeat_count, strerror(errno), name);
      fflush(stderr);
    }
  }

  CmiAssert(fd >= 0);

  ftruncate(fd, size); // set the size of the shared memory object

  *pPtr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  CmiAssert(*pPtr != NULL);

  close(fd);
}

void tearDownSharedBuffers()
{
  int i;
  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      if (shm_unlink(pxshmContext->recvBufNames[i]) < 0) {
        fprintf(stderr, "Error from shm_unlink %s \n", strerror(errno));
      }

#if PXSHM_LOCK
      sem_close(pxshmContext->recvBufs[i].mutex);
      sem_unlink(pxshmContext->recvBufNames[i]);
      sem_close(pxshmContext->sendBufs[i].mutex);
#endif
    }
  }
};

void initSendQ(PxshmSendQ *q, int size);

void initAllSendQs()
{
  int i = 0;
  pxshmContext->sendQs =
      (PxshmSendQ **) malloc(sizeof(PxshmSendQ *) * pxshmContext->nodesize);
  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      (pxshmContext->sendQs)[i] = (PxshmSendQ *) malloc(sizeof(PxshmSendQ));
      initSendQ((pxshmContext->sendQs)[i], SENDQSTARTSIZE);
    } else {
      (pxshmContext->sendQs)[i] = NULL;
    }
  }
};

/****************
 *copy this message into the sharedBuf
 If it does not succeed
 *put it into the sendQ
 *NOTE: This method is called only after obtaining the corresponding mutex
 * ********/
int sendMessage(OutgoingMsg ogm, sharedBufData *dstBuf, PxshmSendQ *dstSendQ)
{

  if (dstBuf->header->bytes + ogm->size <= SHMBUFLEN) {
    /**copy  this message to sharedBuf **/
    dstBuf->header->count++;
    memcpy(dstBuf->data + dstBuf->header->bytes, ogm->data, ogm->size);
    dstBuf->header->bytes += ogm->size;
    MACHSTATE4(3, "Pxshm send done ogm %p size %d dstBuf->header->count %d "
                  "dstBuf->header->bytes %d",
               ogm, ogm->size, dstBuf->header->count, dstBuf->header->bytes);
    return 1;
  }
  /***
   * Shared Buffer is too full for this message
   * **/
  printf("send buffer is too full\n");
  pushSendQ(dstSendQ, ogm);
  ogm->refcount++;
  MACHSTATE3(3, "Pxshm send ogm %p size %d queued refcount %d", ogm, ogm->size,
             ogm->refcount);
  return 0;
}

inline OutgoingMsg popSendQ(PxshmSendQ *q);

/****
 *Try to send all the messages in the sendq to this destination rank
 *NOTE: This method is called only after obtaining the corresponding mutex
 * ************/

inline int flushSendQ(int dstRank)
{
  sharedBufData *dstBuf = &(pxshmContext->sendBufs[dstRank]);
  PxshmSendQ *dstSendQ = pxshmContext->sendQs[dstRank];
  int count = dstSendQ->numEntries;
  int sent = 0;
  while (count > 0) {
    int ret;
    OutgoingMsg ogm = popSendQ(dstSendQ);
    ogm->refcount--;
    MACHSTATE4(3, "Pxshm trysending ogm %p size %d to dstRank %d refcount %d",
               ogm, ogm->size, dstRank, ogm->refcount);
    ret = sendMessage(ogm, dstBuf, dstSendQ);
    if (ret == 1) {
      sent++;
      GarbageCollectMsg(ogm);
    }
    count--;
  }
  return sent;
}

inline void emptyRecvBuf(sharedBufData *recvBuf);

inline void emptyAllRecvBufs()
{
  int i;
  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      sharedBufData *recvBuf = &(pxshmContext->recvBufs[i]);
      if (recvBuf->header->count > 0) {

#if PXSHM_STATS
        pxshmContext->lockRecvCount++;
#endif

#if PXSHM_OSSPINLOCK
        if (!OSSpinLockTry(&recvBuf->header->lock)) {
#elif PXSHM_LOCK
        if (sem_trywait(recvBuf->mutex) < 0) {
#elif PXSHM_FENCE
        recvBuf->header->flagReceiver = 1;
        recvBuf->header->turn = SENDER;
        CmiMemoryReadFence(0, 0);
        CmiMemoryWriteFence(0, 0);
        if ((recvBuf->header->flagSender && recvBuf->header->turn == SENDER)) {
          recvBuf->header->flagReceiver = 0;
#endif
        } else {

          MACHSTATE1(3, "emptyRecvBuf to be called for rank %d", i);
          emptyRecvBuf(recvBuf);

#if PXSHM_OSSPINLOCK
          OSSpinLockUnlock(&recvBuf->header->lock);
#elif PXSHM_LOCK
          sem_post(recvBuf->mutex);
#elif PXSHM_FENCE
          CmiMemoryReadFence(0, 0);
          CmiMemoryWriteFence(0, 0);
          recvBuf->header->flagReceiver = 0;
#endif
        }
      }
    }
  }
};

inline void flushAllSendQs()
{
  int i = 0;

  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank &&
        pxshmContext->sendQs[i]->numEntries > 0) {

#if PXSHM_OSSPINLOCK
      if (OSSpinLockTry(&pxshmContext->sendBufs[i].header->lock)) {
#elif PXSHM_LOCK
      if (sem_trywait(pxshmContext->sendBufs[i].mutex) >= 0) {
#elif PXSHM_FENCE
      pxshmContext->sendBufs[i].header->flagSender = 1;
      pxshmContext->sendBufs[i].header->turn = RECEIVER;
      CmiMemoryReadFence(0, 0);
      CmiMemoryWriteFence(0, 0);
      if (!(pxshmContext->sendBufs[i].header->flagReceiver &&
            pxshmContext->sendBufs[i].header->turn == RECEIVER)) {
#endif

        MACHSTATE1(3, "flushSendQ %d", i);
        flushSendQ(i);

#if PXSHM_OSSPINLOCK
        OSSpinLockUnlock(&pxshmContext->sendBufs[i].header->lock);
#elif PXSHM_LOCK
        sem_post(pxshmContext->sendBufs[i].mutex);
#elif PXSHM_FENCE
        CmiMemoryReadFence(0, 0);
        CmiMemoryWriteFence(0, 0);
        pxshmContext->sendBufs[i].header->flagSender = 0;
#endif
      } else {

#if PXSHM_FENCE
        pxshmContext->sendBufs[i].header->flagSender = 0;
#endif
      }
    }
  }
};

void static inline handoverPxshmMessage(char *newmsg, int total_size, int rank,
                                        int broot);

void emptyRecvBuf(sharedBufData *recvBuf)
{
  int numMessages = recvBuf->header->count;
  int i = 0;

  char *ptr = recvBuf->data;

  for (i = 0; i < numMessages; i++) {
    int size;
    int rank, srcpe, seqno, magic, i;
    unsigned int broot;
    char *msg = ptr;
    char *newMsg;

    DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);
    size = CmiMsgHeaderGetLength(msg);

    newMsg = (char *) CmiAlloc(size);
    memcpy(newMsg, msg, size);

    handoverPxshmMessage(newMsg, size, rank, broot);

    ptr += size;

    MACHSTATE3(
        3,
        "message of size %d recvd ends at ptr-data %d total bytes %d bytes %d",
        size, ptr - recvBuf->data, recvBuf->header->bytes);
  }
  /*
if(ptr - recvBuf->data != recvBuf->header->bytes){
          CmiPrintf("[%d] ptr - recvBuf->data  %d recvBuf->header->bytes %d
numMessages %d initialBytes %d \n",_Cmi_mynode, ptr - recvBuf->data,
recvBuf->header->bytes,numMessages,initialBytes);
  }*/
  CmiAssert(ptr - recvBuf->data == recvBuf->header->bytes);
  recvBuf->header->count = 0;
  recvBuf->header->bytes = 0;
}

void static inline handoverPxshmMessage(char *newmsg, int total_size, int rank,
                                        int broot)
{
  CmiAssert(rank == 0);
#if CMK_BROADCAST_SPANNING_TREE
  if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
      || rank == DGRAM_NODEBROADCAST
#endif
      ) {
    SendSpanningChildren(NULL, 0, total_size, newmsg, broot, rank);
  }
#elif CMK_BROADCAST_HYPERCUBE
  if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
      || rank == DGRAM_NODEBROADCAST
#endif
      ) {
    SendHypercube(NULL, 0, total_size, newmsg, broot, rank);
  }
#endif

  switch (rank) {
  case DGRAM_BROADCAST: {
    CmiPushPE(0, newmsg);
    break;
  }
  default: {

    CmiPushPE(rank, newmsg);
  }
  } /* end of switch */
}

/**************************
 *sendQ helper functions
 * ****************/

void initSendQ(PxshmSendQ *q, int size)
{
  q->data = (OutgoingMsg *) malloc(sizeof(OutgoingMsg) * size);

  q->size = size;
  q->numEntries = 0;

  q->begin = 0;
  q->end = 0;
}

void pushSendQ(PxshmSendQ *q, OutgoingMsg msg)
{
  if (q->numEntries == q->size) {
    // need to resize
    OutgoingMsg *oldData = q->data;
    int newSize = q->size << 1;
    q->data = (OutgoingMsg *) malloc(sizeof(OutgoingMsg) * newSize);
    // copy head to the beginning of the new array

    CmiAssert(q->begin == q->end);

    CmiAssert(q->begin < q->size);
    memcpy(&(q->data[0]), &(oldData[q->begin]),
           sizeof(OutgoingMsg) * (q->size - q->begin));

    if (q->end != 0) {
      memcpy(&(q->data[(q->size - q->begin)]), &(oldData[0]),
             sizeof(OutgoingMsg) * (q->end));
    }
    free(oldData);
    q->begin = 0;
    q->end = q->size;
    q->size = newSize;
  }
  q->data[q->end] = msg;
  (q->end)++;
  if (q->end >= q->size) {
    q->end -= q->size;
  }
  q->numEntries++;
}

OutgoingMsg popSendQ(PxshmSendQ *q)
{
  OutgoingMsg ret;
  if (0 == q->numEntries) {
    return NULL;
  }

  ret = q->data[q->begin];
  (q->begin)++;
  if (q->begin >= q->size) {
    q->begin -= q->size;
  }

  q->numEntries--;
  return ret;
}
