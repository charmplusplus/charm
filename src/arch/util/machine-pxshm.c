/** @file
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
#include <signal.h>

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

#define SENDQ_LIST 0

/*** The following code was copied verbatim from converse.h ***/
#if !CMK_SMP
#if PXSHM_FENCE

#undef CmiMemoryReadFence
#undef CmiMemoryWriteFence

#if CMK_C_SYNC_SYNCHRONIZE_PRIMITIVE
#define CmiMemoryReadFence() __sync_synchronize()
#define CmiMemoryWriteFence() __sync_synchronize()
#elif CMK_C_BUILTIN_IA32_XFENCE
#define CmiMemoryReadFence() __builtin_ia32_lfence()
#define CmiMemoryWriteFence() __builtin_ia32_sfence()
#elif CMK_GCC_X86_ASM
#define CmiMemoryReadFence() __asm__ __volatile__("lfence" ::: "memory")
#define CmiMemoryWriteFence() __asm__ __volatile__("sfence" ::: "memory")
#elif CMK_GCC_IA64_ASM
#define CmiMemoryReadFence() __asm__ __volatile__("mf" ::: "memory")
#define CmiMemoryWriteFence() __asm__ __volatile__("mf" ::: "memory")
#elif CMK_PPC_ASM
#define CmiMemoryReadFence() __asm__ __volatile__("sync" ::: "memory")
#define CmiMemoryWriteFence() __asm__ __volatile__("sync" ::: "memory")
#else
#error Cannot build PXSHM with non-SMP on a machine with no ASM for atomic fence
#endif /* CMK_C_SYNC_SYNCHRONIZE_PRIMITIVE */

#endif // end of PXSHM_FENCE
#endif // end of CMK_SMP

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

#define NAMESTRLEN 60
#define PREFIXSTRLEN 50

static int SHMBUFLEN = (1024 * 1024 * 4);
static int SHMMAXSIZE = (1024 * 1024);

static int SENDQSTARTSIZE = 256;

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
  CmiMemorySMPSeparation_t pad1;
  volatile int flagReceiver;
  CmiMemorySMPSeparation_t pad2;
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

typedef struct OutgoingMsgRec {
  char *data;
  int *refcount;
  int size;
} OutgoingMsgRec;

typedef struct {
  int size; // total size of data array
  int begin; // position of first element
  int end; // position of next element
  int numEntries; // number of entries
  int rank; // for dest rank
#if SENDQ_LIST
  int next; // next dstrank of non-empty queue
#endif
  OutgoingMsgRec *data;

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

#if SENDQ_LIST
static int sendQ_head_index = -1;
#endif

PxshmContext *pxshmContext = NULL; // global context

void calculateNodeSizeAndRank(char **);
void setupSharedBuffers();
void initAllSendQs();

void CmiExitPxshm();

static void cleanupOnAllSigs(int signo) { CmiExitPxshm(); }

/******************
 * 	Initialization routine
 * 	currently just testing start up
 * ****************/
void CmiInitPxshm(char **argv)
{
  char *env;
  MACHSTATE(3, "CminitPxshm start");

  pxshmContext = (PxshmContext *) calloc(1, sizeof(PxshmContext));

  CmiDeprecateArgInt(argv, "+nodesize", "Number of cores in this node",
                     "Charmrun> Deprecation warning: charmrun now figures "
                     "out the nodesize on its own.");

  calculateNodeSizeAndRank(argv);
  if (pxshmContext->nodesize == 1)
    return;

  MACHSTATE1(3, "CminitPxshm  %d calculateNodeSizeAndRank",
             pxshmContext->nodesize);

  env = getenv("CHARM_PXSHM_POOL_SIZE");
  if (env) {
    SHMBUFLEN = CmiReadSize(env);
  }
  env = getenv("CHARM_PXSHM_MESSAGE_MAX_SIZE");
  if (env) {
    SHMMAXSIZE = CmiReadSize(env);
  }
  if (SHMMAXSIZE > SHMBUFLEN)
    CmiAbort("Error> Pxshm pool size is set too small in env variable "
             "CHARM_PXSHM_POOL_SIZE");

  SENDQSTARTSIZE = 32 * pxshmContext->nodesize;

  if (_Cmi_mynode == 0)
    printf("Charm++> pxshm enabled: %d cores per node, buffer size: %.1fMB\n",
           pxshmContext->nodesize, SHMBUFLEN / 1024.0 / 1024.0);

#if CMK_CRAYXE || CMK_CRAYXC
  srand(getpid());
  int Cmi_charmrun_pid = rand();
  PMI_Bcast(&Cmi_charmrun_pid, sizeof(int));
  snprintf(&(pxshmContext->prefixStr[0]), PREFIXSTRLEN - 1, "charm_pxshm_%d",
           Cmi_charmrun_pid);
#elif CMK_NET_VERSION
  srand(Cmi_charmrun_pid);
  snprintf(&(pxshmContext->prefixStr[0]), PREFIXSTRLEN - 1, "charm_pxshm_%d",
           rand());
#else
#error PXSHM does not support the machine layer you are building on; please contact Charm++ develpers to report this.
#endif

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

  signal(SIGSEGV, cleanupOnAllSigs);
  signal(SIGFPE, cleanupOnAllSigs);
  signal(SIGILL, cleanupOnAllSigs);
  signal(SIGTERM, cleanupOnAllSigs);
  signal(SIGABRT, cleanupOnAllSigs);
  signal(SIGQUIT, cleanupOnAllSigs);
  signal(SIGBUS, cleanupOnAllSigs);
  signal(SIGINT, cleanupOnAllSigs);
  signal(SIGTRAP, cleanupOnAllSigs);

#if 0
        char name[64];
        gethostname(name,64);
        printf("[%d] name: %s\n", myrank, name);
#endif
};

/**************
 * shutdown shmem objects and semaphores
 *
 * *******************/
static int pxshm_freed = 0;
void tearDownSharedBuffers();
void freeSharedBuffers();

void CmiExitPxshm()
{
  if (pxshmContext == NULL)
    return;
  if (pxshmContext->nodesize != 1) {
    int i;
    if (!pxshm_freed)
      tearDownSharedBuffers();

    for (i = 0; i < pxshmContext->nodesize; i++) {
      if (i != pxshmContext->noderank) {
        break;
      }
    }
    free(pxshmContext->recvBufNames[i]);
    free(pxshmContext->sendBufNames[i]);

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
  pxshmContext = NULL;
}

/******************
 *Should this message be sent using PxShm or not ?
 * ***********************/

/* dstNode is node number */
inline static int CmiValidPxshm(int node, int size)
{
#if PXSHM_STATS
  pxshmContext->validCheckCount++;
#endif

  /*	if(pxshmContext->nodesize == 1){
                  return 0;
          }*/
  // replace by bitmap later
  // if(ogm->dst >= pxshmContext->nodestart && ogm->dst <= pxshmContext->nodeend
  // && ogm->size < SHMBUFLEN ){
  return (node >= pxshmContext->nodestart && node <= pxshmContext->nodeend &&
          size <= SHMMAXSIZE)
             ? 1
             : 0;
};

int PxshmRank(int dstnode) { return dstnode - pxshmContext->nodestart; }

inline void pushSendQ(PxshmSendQ *q, char *msg, int size, int *refcount);
inline int sendMessage(char *msg, int size, int *refcount,
                       sharedBufData *dstBuf, PxshmSendQ *dstSendQ);
int flushSendQ(PxshmSendQ *q);

int sendMessageRec(OutgoingMsgRec *omg, sharedBufData *dstBuf,
                          PxshmSendQ *dstSendQ)
{
  return sendMessage(omg->data, omg->size, omg->refcount, dstBuf, dstSendQ);
}

/***************
 *
 *Send this message through shared memory
 *if you cannot get lock, put it in the sendQ
 *Before sending messages pick them from sendQ
 *
 * ****************************/

void CmiSendMessagePxshm(char *msg, int size, int dstnode, int *refcount)
{
#if PXSHM_STATS
  double _startSendTime = CmiWallTimer();
#endif

  LrtsPrepareEnvelope(msg, size);

  int dstRank = PxshmRank(dstnode);
  MEMDEBUG(CmiMemoryCheck());

  /*
          MACHSTATE4(3,"Send Msg Pxshm ogm %p size %d dst %d dstRank
     %d",ogm,ogm->size,ogm->dst,dstRank);
          MACHSTATE4(3,"Send Msg Pxshm ogm %p size %d dst %d dstRank
     %d",ogm,ogm->size,ogm->dst,dstRank);
  */

  CmiAssert(dstRank >= 0 && dstRank != pxshmContext->noderank);

  sharedBufData *dstBuf = &(pxshmContext->sendBufs[dstRank]);
  PxshmSendQ *sendQ = pxshmContext->sendQs[dstRank];

#if PXSHM_OSSPINLOCK
  if (!OSSpinLockTry(&dstBuf->header->lock)) {
#elif PXSHM_LOCK
  if (sem_wait(dstBuf->mutex) != 0) {
// never comes here unless the mutex is buggy
#elif PXSHM_FENCE
  dstBuf->header->flagSender = 1;
  dstBuf->header->turn = RECEIVER;
  CmiMemoryReadFence();
  CmiMemoryWriteFence();
  // if(dstBuf->header->flagReceiver && dstBuf->header->turn == RECEIVER){
  if (dstBuf->header->flagReceiver) {
    dstBuf->header->flagSender = 0;
#endif
/**failed to get the lock
insert into q and retain the message*/
#if SENDQ_LIST
    if (sendQ->numEntries == 0 && sendQ->next == -2) {
      sendQ->next = sendQ_head_index;
      sendQ_head_index = dstRank;
    }
#endif
    pushSendQ(pxshmContext->sendQs[dstRank], msg, size, refcount);
    (*refcount)++;
    MEMDEBUG(CmiMemoryCheck());
    return;
  } else {

    /***
     * We got the lock for this buffer
     * first write all the messages in the sendQ and then write this guy
     * */
    if (pxshmContext->sendQs[dstRank]->numEntries == 0) {
      // send message user event
      int ret = sendMessage(msg, size, refcount, dstBuf,
                            pxshmContext->sendQs[dstRank]);
#if SENDQ_LIST
      if (sendQ->numEntries > 0 && sendQ->next == -2) {
        sendQ->next = sendQ_head_index;
        sendQ_head_index = dstRank;
      }
#endif
      MACHSTATE(3, "Pxshm Send succeeded immediately");
    } else {
      (*refcount) +=
          2; /*this message should not get deleted when the queue is flushed*/
      pushSendQ(pxshmContext->sendQs[dstRank], msg, size, refcount);
      //			MACHSTATE3(3,"Pxshm ogm %p pushed to sendQ
      //length %d refcount
      //%d",ogm,pxshmContext->sendQs[dstRank]->numEntries,ogm->refcount);
      int sent = flushSendQ(sendQ);
      (*refcount)--; /*if it has been sent, can be deleted by caller, if not
                        will be deleted when queue is flushed*/
      MACHSTATE1(3, "Pxshm flushSendQ sent %d messages", sent);
    }
/* unlock the recvbuffer*/

#if PXSHM_OSSPINLOCK
    OSSpinLockUnlock(&dstBuf->header->lock);
#elif PXSHM_LOCK
    sem_post(dstBuf->mutex);
#elif PXSHM_FENCE
    CmiMemoryReadFence();
    CmiMemoryWriteFence();
    dstBuf->header->flagSender = 0;
#endif
  }
#if PXSHM_STATS
  pxshmContext->sendCount++;
  pxshmContext->sendTime += (CmiWallTimer() - _startSendTime);
#endif
  MEMDEBUG(CmiMemoryCheck());
};

void emptyAllRecvBufs();
void flushAllSendQs();

/**********
 * Extract all the messages from the recvBuffers you can
 * Flush all sendQs
 * ***/
void CommunicationServerPxshm()
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

void calculateNodeSizeAndRank(char **argv)
{
  pxshmContext->nodesize = 1;
  MACHSTATE(3, "calculateNodeSizeAndRank start");
  // CmiGetArgIntDesc(argv, "+nodesize", &(pxshmContext->nodesize),"Number of
  // cores in this node (for non-smp case).Used by the shared memory
  // communication layer");
#if CMK_CRAYXT || CMK_CRAYXE || CMK_CRAYXC
  // On Cray machines, PXSHM is initialized twice: before and after initializing
  // the CPU topology mechanism. The information below is only accurate on the
  // second phase. In other words, other LRTS-based machine layers need to get
  // that information elsewhere, e.g. from charmrun.
  pxshmContext->nodesize =
      (CmiNumPesOnPhysicalNode(CmiPhysicalNodeID(CmiMyNode())) +
       CmiMyNodeSize() - 1) /
      CmiMyNodeSize();
#else
  pxshmContext->nodesize = _Cmi_myphysnode_numprocesses;
#endif
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
      // CmiAssert(pxshmContext->sendBufs[i].header->count == 0);
      pxshmContext->sendBufs[i].header->count = 0;
      pxshmContext->sendBufs[i].header->bytes = 0;
    }
  }

  return LrtsBarrier();
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
      (sharedBufData *) calloc(pxshmContext->nodesize, sizeof(sharedBufData));

  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      createShmObject(bufNames[i], SHMBUFLEN + sizeof(sharedBufHeader),
                      (char **) &((*bufs)[i].header));
      memset(((*bufs)[i].header), 0, SHMBUFLEN + sizeof(sharedBufHeader));
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

void freeSharedBuffers()
{
  int i;
  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      if (shm_unlink(pxshmContext->recvBufNames[i]) < 0) {
        fprintf(stderr, "Error from shm_unlink %s \n", strerror(errno));
      }
#if PXSHM_LOCK
      sem_unlink(pxshmContext->recvBufNames[i]);
#endif
    }
  }
};

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
      sem_close(pxshmContext->sendBufs[i].mutex);
      sem_unlink(pxshmContext->recvBufNames[i]);
      pxshmContext->recvBufs[i].mutex = NULL;
      pxshmContext->sendBufs[i].mutex = NULL;
#endif
    }
  }
};

void initSendQ(PxshmSendQ *q, int size, int rank);

void initAllSendQs()
{
  int i = 0;
  pxshmContext->sendQs =
      (PxshmSendQ **) malloc(sizeof(PxshmSendQ *) * pxshmContext->nodesize);
  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i != pxshmContext->noderank) {
      (pxshmContext->sendQs)[i] = (PxshmSendQ *) calloc(1, sizeof(PxshmSendQ));
      initSendQ((pxshmContext->sendQs)[i], SENDQSTARTSIZE, i);
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
int sendMessage(char *msg, int size, int *refcount, sharedBufData *dstBuf,
                PxshmSendQ *dstSendQ)
{

  if (dstBuf->header->bytes + size <= SHMBUFLEN) {
    /**copy  this message to sharedBuf **/
    dstBuf->header->count++;
    memcpy(dstBuf->data + dstBuf->header->bytes, msg, size);
    dstBuf->header->bytes += size;
    //		MACHSTATE4(3,"Pxshm send done ogm %p size %d dstBuf->header->count %d
    //dstBuf->header->bytes
    //%d",ogm,ogm->size,dstBuf->header->count,dstBuf->header->bytes);
    CmiFree(msg);
    return 1;
  }
  /***
   * Shared Buffer is too full for this message
   * **/
  // printf("[%d] send buffer is too full\n", CmiMyPe());
  pushSendQ(dstSendQ, msg, size, refcount);
  (*refcount)++;
  //	MACHSTATE3(3,"Pxshm send ogm %p size %d queued refcount
  //%d",ogm,ogm->size,ogm->refcount);
  return 0;
}

inline OutgoingMsgRec *popSendQ(PxshmSendQ *q);

/****
 *Try to send all the messages in the sendq to this destination rank
 *NOTE: This method is called only after obtaining the corresponding mutex
 * ************/

inline int flushSendQ(PxshmSendQ *dstSendQ)
{
  sharedBufData *dstBuf = &(pxshmContext->sendBufs[dstSendQ->rank]);
  int count = dstSendQ->numEntries;
  int sent = 0;
  while (count > 0) {
    OutgoingMsgRec *ogm = popSendQ(dstSendQ);
    (*ogm->refcount)--;
    MACHSTATE4(3, "Pxshm trysending ogm %p size %d to dstRank %d refcount %d",
               ogm, ogm->size, dstSendQ->rank, ogm->refcount);
    int ret = sendMessageRec(ogm, dstBuf, dstSendQ);
    if (ret == 1) {
      sent++;
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
        if (sem_wait(recvBuf->mutex) != 0) {
#elif PXSHM_FENCE
        recvBuf->header->flagReceiver = 1;
        recvBuf->header->turn = SENDER;
        CmiMemoryReadFence();
        CmiMemoryWriteFence();
        // if((recvBuf->header->flagSender && recvBuf->header->turn == SENDER)){
        if ((recvBuf->header->flagSender)) {
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
          CmiMemoryReadFence();
          CmiMemoryWriteFence();
          recvBuf->header->flagReceiver = 0;
#endif
        }
      }
    }
  }
};

inline void flushAllSendQs()
{
  int i;
#if SENDQ_LIST
  int index_prev = -1;

  i = sendQ_head_index;
  while (i != -1) {
    PxshmSendQ *sendQ = pxshmContext->sendQs[i];
    CmiAssert(i != pxshmContext->noderank);
    if (sendQ->numEntries > 0) {
#else
  for (i = 0; i < pxshmContext->nodesize; i++) {
    if (i == pxshmContext->noderank)
      continue;
    PxshmSendQ *sendQ = pxshmContext->sendQs[i];
    if (sendQ->numEntries > 0) {
#endif

#if PXSHM_OSSPINLOCK
      if (OSSpinLockTry(&pxshmContext->sendBufs[i].header->lock)) {
#elif PXSHM_LOCK
      if (sem_wait(pxshmContext->sendBufs[i].mutex) == 0) {
#elif PXSHM_FENCE
  pxshmContext->sendBufs[i].header->flagSender = 1;
  pxshmContext->sendBufs[i].header->turn = RECEIVER;
  CmiMemoryReadFence();
  CmiMemoryWriteFence();
  if (!(pxshmContext->sendBufs[i].header->flagReceiver &&
        pxshmContext->sendBufs[i].header->turn == RECEIVER)) {
#endif

        MACHSTATE1(3, "flushSendQ %d", i);
        flushSendQ(sendQ);

#if PXSHM_OSSPINLOCK
        OSSpinLockUnlock(&pxshmContext->sendBufs[i].header->lock);
#elif PXSHM_LOCK
        sem_post(pxshmContext->sendBufs[i].mutex);
#elif PXSHM_FENCE
    CmiMemoryReadFence();
    CmiMemoryWriteFence();
    pxshmContext->sendBufs[i].header->flagSender = 0;
#endif
      } else {

#if PXSHM_FENCE
        pxshmContext->sendBufs[i].header->flagSender = 0;
#endif
      }
    }
#if SENDQ_LIST
    if (sendQ->numEntries == 0) {
      if (index_prev != -1)
        pxshmContext->sendQs[index_prev]->next = sendQ->next;
      else
        sendQ_head_index = sendQ->next;
      i = sendQ->next;
      sendQ->next = -2;
    } else {
      index_prev = i;
      i = sendQ->next;
    }
#endif
  }
};

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

    size = CMI_MSG_SIZE(msg);

    newMsg = (char *) CmiAlloc(size);
    memcpy(newMsg, msg, size);

    handleOneRecvedMsg(size, newMsg);

    ptr += size;

    MACHSTATE3(
        3,
        "message of size %d recvd ends at ptr-data %d total bytes %d bytes %d",
        size, ptr - recvBuf->data, recvBuf->header->bytes);
  }
#if 1
  if (ptr - recvBuf->data != recvBuf->header->bytes) {
    CmiPrintf("[%d] ptr - recvBuf->data  %d recvBuf->header->bytes %d "
              "numMessages %d \n",
              _Cmi_mynode, ptr - recvBuf->data, recvBuf->header->bytes,
              numMessages);
  }
#endif
  CmiAssert(ptr - recvBuf->data == recvBuf->header->bytes);
  recvBuf->header->count = 0;
  recvBuf->header->bytes = 0;
}

/**************************
 *sendQ helper functions
 * ****************/

void initSendQ(PxshmSendQ *q, int size, int rank)
{
  q->data = (OutgoingMsgRec *) calloc(size, sizeof(OutgoingMsgRec));

  q->size = size;
  q->numEntries = 0;

  q->begin = 0;
  q->end = 0;

  q->rank = rank;
#if SENDQ_LIST
  q->next = -2;
#endif
}

void pushSendQ(PxshmSendQ *q, char *msg, int size, int *refcount)
{
  if (q->numEntries == q->size) {
    // need to resize
    OutgoingMsgRec *oldData = q->data;
    int newSize = q->size << 1;
    q->data = (OutgoingMsgRec *) calloc(newSize, sizeof(OutgoingMsgRec));
    // copy head to the beginning of the new array
    CmiAssert(q->begin == q->end);

    CmiAssert(q->begin < q->size);
    memcpy(&(q->data[0]), &(oldData[q->begin]),
           sizeof(OutgoingMsgRec) * (q->size - q->begin));

    if (q->end != 0) {
      memcpy(&(q->data[(q->size - q->begin)]), &(oldData[0]),
             sizeof(OutgoingMsgRec) * (q->end));
    }
    free(oldData);
    q->begin = 0;
    q->end = q->size;
    q->size = newSize;
  }
  OutgoingMsgRec *omg = &q->data[q->end];
  omg->size = size;
  omg->data = msg;
  omg->refcount = refcount;
  (q->end)++;
  if (q->end >= q->size) {
    q->end -= q->size;
  }
  q->numEntries++;
}

OutgoingMsgRec *popSendQ(PxshmSendQ *q)
{
  OutgoingMsgRec *ret;
  if (0 == q->numEntries) {
    return NULL;
  }

  ret = &q->data[q->begin];
  (q->begin)++;
  if (q->begin >= q->size) {
    q->begin -= q->size;
  }

  q->numEntries--;
  return ret;
}
