#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#define _POSIX1C
#define _NO_ANSIMODE

#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#include <limits.h>
#include <unistd.h>

#include "converse.h"
#include "fifo.h"

#define BLK_LEN  512

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int waiting;
  void     **blk;
  unsigned int blk_len;
  unsigned int first;
  unsigned int len;
  unsigned int maxlen;
} McQueue;

static McQueue *McQueueCreate(void);
static void McQueueAddToBack(McQueue *queue, void *element);
static void *McQueueRemoveFromFront(McQueue *queue);
static McQueue **MsgQueue;

CpvDeclare(void*, CmiLocalQueue);

int Cmi_argc;
int Cmi_numpes;
int Cmi_usched;
int Cmi_initret;
CmiStartFn Cmi_startFn;

pthread_key_t perThreadKey;

static void *threadInit(void *arg);

pthread_mutex_t memory_mutex;

void CmiMemLock() {pthread_mutex_lock(&memory_mutex);}
void CmiMemUnlock() {pthread_mutex_unlock(&memory_mutex);}

int barrier;
pthread_cond_t barrier_cond;
pthread_mutex_t barrier_mutex;

void CmiNodeBarrier(void)
{
  pthread_mutex_lock(&barrier_mutex);
  barrier++;
  if(barrier!=CmiNumPes())
    pthread_cond_wait(&barrier_cond, &barrier_mutex);
  else {
    barrier = 0;
    pthread_cond_broadcast(&barrier_cond);
  }
  pthread_mutex_unlock(&barrier_mutex);
}

CmiNodeLock CmiCreateLock(void)
{
  pthread_mutex_t *lock;
  lock = (pthread_mutex_t *) CmiAlloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(lock, (pthread_mutexattr_t *) 0);
  return lock;
}

void CmiLock(CmiNodeLock lock)
{
  pthread_mutex_lock(lock);
}

void CmiUnlock(CmiNodeLock lock)
{
  pthread_mutex_unlock(lock);
}

int CmiTryLock(CmiNodeLock lock)
{
  return pthread_mutex_trylock(lock);
}

void CmiDestroyLock(CmiNodeLock lock)
{
  pthread_mutex_destroy(lock);
}

int CmiMyPe()
{
  int mype = (int) pthread_getspecific(perThreadKey);
  return mype;
}

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
  CmiError(message);
  abort();
}

int CmiAsyncMsgSent(CmiCommHandle msgid)
{
  return 1;
}


typedef struct {
  char       **argv;
  int        mype;
} USER_PARAMETERS;

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int i,j;
  char **uargv;
  USER_PARAMETERS *usrparam;
  pthread_t *aThread;
 
  Cmi_numpes = 0; 
  Cmi_usched = usched;
  Cmi_initret = initret;
  Cmi_startFn = fn;
  Cmi_argc = argc;

  for(i=0;i<Cmi_argc;i++) {
    if (strcmp(argv[i], "+p") == 0) {
      sscanf(argv[i+1], "%d", &Cmi_numpes);
      break;
    } else {
      if (sscanf(argv[i], "+p%d", &Cmi_numpes) == 1) 
	break;
    }
  }

  if (Cmi_numpes <= 0)
  {
    CmiError("Error: requested number of processors is invalid %d\n",
              Cmi_numpes);
    abort();
  }


  pthread_mutex_init(&memory_mutex, (pthread_mutexattr_t *) 0);

  MsgQueue=(McQueue **)CmiAlloc(Cmi_numpes*sizeof(McQueue *));
  if (MsgQueue == (McQueue **)0) {
    CmiError("Cannot Allocate Memory...\n");
    abort();
  }
  for(i=0; i<Cmi_numpes; i++) 
    MsgQueue[i] = McQueueCreate();

  pthread_key_create(&perThreadKey, (void *) 0);
  barrier = 0;
  pthread_cond_init(&barrier_cond, (pthread_condattr_t *) 0);
  pthread_mutex_init(&barrier_mutex, (pthread_mutexattr_t *) 0);

  /* suggest to IRIX that we actually use the right number of processors */
  pthread_setconcurrency(Cmi_numpes);

  aThread = (pthread_t *) CmiAlloc(sizeof(pthread_t) * Cmi_numpes);
  for(i=1; i<Cmi_numpes; i++) {
    uargv = (char **) CmiAlloc(sizeof(char *) * (Cmi_argc+1));

    for (j=0;j<Cmi_argc;j++)
      uargv[j] = argv[j];
    uargv[j] = 0;

    usrparam = (USER_PARAMETERS *) CmiAlloc(sizeof(USER_PARAMETERS));
    usrparam->argv = uargv;
    usrparam->mype = i;

    pthread_create(&aThread[i],(pthread_attr_t *)0,threadInit,(void *)usrparam);
  }
  uargv = (char **) CmiAlloc(sizeof(char *) * (Cmi_argc+1));
  for (j=0;j<Cmi_argc;j++)
    uargv[j] = argv[j];
  uargv[j] = 0;
  usrparam = (USER_PARAMETERS *) CmiAlloc(sizeof(USER_PARAMETERS));
  usrparam->argv = uargv;
  usrparam->mype = 0;
  threadInit(usrparam);
}

void CmiTimerInit(void);

static void *threadInit(void *arg)
{
  USER_PARAMETERS *usrparam;
  usrparam = (USER_PARAMETERS *) arg;


  pthread_setspecific(perThreadKey, (void *) usrparam->mype);

  CthInit(usrparam->argv);
  ConverseCommonInit(usrparam->argv);
  CpvInitialize(void*, CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
  CmiTimerInit();
  if (Cmi_initret==0) {
    Cmi_startFn(Cmi_argc, usrparam->argv);
    if (Cmi_usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}


void ConverseExit(void)
{
  ConverseCommonExit();
  CmiNodeBarrier();
}


void CmiDeclareArgs(void)
{
}


void CmiNotifyIdle()
{
  McQueue *queue = MsgQueue[CmiMyPe()];
  struct timespec ts;
  pthread_mutex_lock(&(queue->mutex));
  if(!queue->len){
    queue->waiting++;
    ts.tv_sec = (time_t) 0;
    ts.tv_nsec = 10000000L;
    pthread_cond_timedwait(&(queue->cond), &(queue->mutex), &ts);
    queue->waiting--;
  }
  pthread_mutex_unlock(&(queue->mutex));
  return;
}

void *CmiGetNonLocal()
{
  return McQueueRemoveFromFront(MsgQueue[CmiMyPe()]);
}


void CmiSyncSendFn(int destPE, int size, char *msg)
{
  char *buf;

  buf=(void *)CmiAlloc(size);
  memcpy(buf,msg,size);
  McQueueAddToBack(MsgQueue[destPE],buf); 
}


CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  CmiSyncSendFn(destPE, size, msg); 
  return 0;
}


void CmiFreeSendFn(int destPE, int size, char *msg)
{
  if (CmiMyPe()==destPE) {
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  } else {
    McQueueAddToBack(MsgQueue[destPE],msg); 
  }
}

void CmiSyncBroadcastFn(int size, char *msg)
{
  int i;
  for(i=0; i<Cmi_numpes; i++)
    if (CmiMyPe() != i) CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size, msg);
  return 0;
}

void CmiFreeBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size,msg);
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg)
{
  int i;
  for(i=0; i<CmiNumPes(); i++) 
    CmiSyncSendFn(i,size,msg);
}


CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)
{
  CmiSyncBroadcastAllFn(size, msg);
  return 0; 
}


void CmiFreeBroadcastAllFn(int size, char *msg)
{
  int i;
  for(i=0; i<CmiNumPes(); i++) {
    if(CmiMyPe() != i) {
      CmiSyncSendFn(i,size,msg);
    }
  }
  FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
}

/* ****************************************************************** */
/*    The following internal functions implements FIFO queues for     */
/*    messages. These queues are shared among threads                 */
/* ****************************************************************** */

static void ** AllocBlock(unsigned int len)
{
  void **blk;

  blk=(void **)CmiAlloc(len*sizeof(void *));
  return blk;
}

static void 
SpillBlock(void **srcblk, void **destblk, unsigned int first, unsigned int len)
{
  memcpy(destblk, &(srcblk[first]), (len-first)*sizeof(void *));
  memcpy(&(destblk[len-first]),srcblk,first*sizeof(void *));
}

McQueue * McQueueCreate(void)
{
  McQueue *queue;

  queue = (McQueue *) CmiAlloc(sizeof(McQueue));
  pthread_mutex_init(&(queue->mutex), (pthread_mutexattr_t *) 0);
  pthread_cond_init(&(queue->cond), (pthread_condattr_t *) 0);
  queue->waiting = 0;
  queue->blk = AllocBlock(BLK_LEN);
  queue->blk_len = BLK_LEN;
  queue->first = 0;
  queue->len = 0;
  queue->maxlen = 0;
  return queue;
}

void McQueueAddToBack(McQueue *queue, void *element)
{
  pthread_mutex_lock(&(queue->mutex));
  if(queue->len==queue->blk_len) {
    void **blk;

    queue->blk_len *= 3;
    blk = AllocBlock(queue->blk_len);
    SpillBlock(queue->blk, blk, queue->first, queue->len);
    CmiFree(queue->blk);
    queue->blk = blk;
    queue->first = 0;
  }
  queue->blk[(queue->first+queue->len++)%queue->blk_len] = element;
  if(queue->len>queue->maxlen)
    queue->maxlen = queue->len;
  if(queue->waiting) {
    pthread_cond_broadcast(&(queue->cond));
  }
  pthread_mutex_unlock(&(queue->mutex));
}


void * McQueueRemoveFromFront(McQueue *queue)
{
  void *element = 0;
  pthread_mutex_lock(&(queue->mutex));
  if(queue->len) {
    element = queue->blk[queue->first++];
    queue->first = (queue->first+queue->blk_len)%queue->blk_len;
    queue->len--;
  }
  pthread_mutex_unlock(&(queue->mutex));
  return element;
}

/* Timer Routines */


CpvStaticDeclare(double,inittime_wallclock);
CpvStaticDeclare(double,inittime_virtual);

void CmiTimerInit(void)
{
  struct timespec temp;
  CpvInitialize(double, inittime_wallclock);
  CpvInitialize(double, inittime_virtual);
  clock_gettime(CLOCK_SGI_CYCLE, &temp);
  CpvAccess(inittime_wallclock) = (double) temp.tv_sec +
				  1e-9 * temp.tv_nsec;
  CpvAccess(inittime_virtual) = CpvAccess(inittime_wallclock);
}

double CmiWallTimer(void)
{
  struct timespec temp;
  double currenttime;

  clock_gettime(CLOCK_SGI_CYCLE, &temp);
  currenttime = (double) temp.tv_sec +
                1e-9 * temp.tv_nsec;
  return (currenttime - CpvAccess(inittime_wallclock));
}

double CmiCpuTimer(void)
{
  struct timespec temp;
  double currenttime;

  clock_gettime(CLOCK_SGI_CYCLE, &temp);
  currenttime = (double) temp.tv_sec +
                1e-9 * temp.tv_nsec;
  return (currenttime - CpvAccess(inittime_virtual));
}

double CmiTimer(void)
{
  return CmiCpuTimer();
}

