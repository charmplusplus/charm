
/*
for SMP versions:

CmiStateInit
CmiGetState
CmiGetStateN
CmiYield
CmiStartThreads

CmiIdleLock_init
CmiIdleLock_sleep
CmiIdleLock_addMessage
CmiIdleLock_checkMessage
*/

#include "machine-smp.h"

void CmiStateInit(int pe, int rank, CmiState state);

/************************ Win32 kernel SMP threads **************/
#if CMK_SHARED_VARS_NT_THREADS

CmiNodeLock CmiMemLock_lock;
static HANDLE comm_mutex;
#define CmiCommLockOrElse(x) /*empty*/
#define CmiCommLock() (WaitForSingleObject(comm_mutex, INFINITE))
#define CmiCommUnlock() (ReleaseMutex(comm_mutex))

static DWORD Cmi_state_key = 0xFFFFFFFF;
static CmiState     Cmi_state_vector = 0;

#ifdef CMK_OPTIMIZE
#  define CmiGetState() ((CmiState)TlsGetValue(Cmi_state_key))
#else
CmiState CmiGetState()
{
  CmiState result;
  result = (CmiState)TlsGetValue(Cmi_state_key);
  if(result == 0) PerrorExit("CmiGetState: TlsGetValue");
  return result;
}
#endif

CmiNodeLock CmiCreateLock(void)
{
  HANDLE hMutex = CreateMutex(NULL, FALSE, NULL);
  return hMutex;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  CloseHandle(lk);
}

void CmiYield(void) 
{ 
  Sleep(0);
}

#define CmiGetStateN(n) (Cmi_state_vector+(n))

static DWORD WINAPI comm_thread(LPVOID dummy)
{  
  if (Cmi_charmrun_fd!=-1)
    while (1) CommunicationServer(5);
  return 0;
}

static DWORD WINAPI call_startfn(LPVOID vindex)
{
  int index = (int)vindex;
 
  CmiState state = Cmi_state_vector + index;
  if(Cmi_state_key == 0xFFFFFFFF) PerrorExit("TlsAlloc");
  if(TlsSetValue(Cmi_state_key, (LPVOID)state) == 0) PerrorExit("TlsSetValue");

  ConverseRunPE(0);
  return 0;
}


/*Classic sense-reversing barrier algorithm.
FIXME: This should be the barrier implementation for 
all thread types.
*/
static HANDLE barrier_mutex;
static volatile int    barrier_wait[2] = {0,0};
static volatile int    barrier_which = 0;

void  CmiNodeBarrier(void)
{
  int doWait = 1;
  int which;

  WaitForSingleObject(barrier_mutex, INFINITE);
  which=barrier_which;
  barrier_wait[which]++;
  if (barrier_wait[which] == Cmi_mynodesize) {
    barrier_which = !which;
    barrier_wait[barrier_which] = 0;/*Reset new counter*/
    doWait = 0;
  }
  ReleaseMutex(barrier_mutex);

  if (doWait)
      while(barrier_wait[which] != Cmi_mynodesize)
		  sleep(0);/*<- could also just spin here*/
}

static void CmiStartThreads(char **argv)
{
  int     i;
  DWORD   threadID;
  HANDLE  thr;
  int     val = 0;

  CmiMemLock_lock=CmiCreateLock();
  comm_mutex = CmiCreateLock();
  barrier_mutex = CmiCreateLock();

  Cmi_state_key = TlsAlloc();
  if(Cmi_state_key == 0xFFFFFFFF) PerrorExit("TlsAlloc main");
  
  Cmi_state_vector =
    (CmiState)calloc(Cmi_mynodesize, sizeof(struct CmiStateStruct));
  
  for (i=0; i<Cmi_mynodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  
  for (i=1; i<Cmi_mynodesize; i++) {
    if((thr = CreateThread(NULL, 0, call_startfn, (LPVOID)i, 0, &threadID)) 
       == NULL) PerrorExit("CreateThread");
    CloseHandle(thr);
  }
  
  if(TlsSetValue(Cmi_state_key, (LPVOID)Cmi_state_vector) == 0) 
    PerrorExit("TlsSetValue");
  
  if((thr = CreateThread(NULL, 0, comm_thread, 0, 0, &threadID)) == NULL) 
     PerrorExit("CreateThread");
  CloseHandle(thr);
}

/***************** Pthreads kernel SMP threads ******************/
#elif CMK_SHARED_VARS_POSIX_THREADS_SMP

static pthread_key_t Cmi_state_key;
static CmiState     Cmi_state_vector;
CmiNodeLock CmiMemLock_lock;

#define CmiGetState() ((CmiState)pthread_getspecific(Cmi_state_key))

CmiNodeLock CmiCreateLock(void)
{
  CmiNodeLock lk = (CmiNodeLock)malloc(sizeof(pthread_mutex_t));
  _MEMCHECK(lk);
  pthread_mutex_init(lk,(pthread_mutexattr_t *)0);
  return lk;
}

void CmiDestroyLock(CmiNodeLock lk)
{
  pthread_mutex_destroy(lk);
  free(lk);
}

void CmiYield(void) { sched_yield(); }

int barrier;
pthread_cond_t barrier_cond;
pthread_mutex_t barrier_mutex;

void CmiNodeBarrier(void)
{
  pthread_mutex_lock(&barrier_mutex);
  barrier++;
  if(barrier != CmiMyNodeSize())
    pthread_cond_wait(&barrier_cond, &barrier_mutex);
  else{
    barrier = 0;
    pthread_cond_broadcast(&barrier_cond);
  }
  pthread_mutex_unlock(&barrier_mutex);
}


#define CmiGetStateN(n) (Cmi_state_vector+(n))

static CmiNodeLock comm_mutex;

#define CmiCommLockOrElse(x) /*empty*/

#if 1
/*Regular comm. lock*/
#  define CmiCommLock() CmiLock(comm_mutex)
#  define CmiCommUnlock() CmiUnlock(comm_mutex)
#else
/*Verbose debugging comm. lock*/
static int comm_mutex_isLocked=0;
void CmiCommLock(void) {
	if (comm_mutex_isLocked) 
		CmiAbort("CommLock: already locked!\n");
	CmiLock(comm_mutex);
	comm_mutex_isLocked=1;
}
void CmiCommUnlock(void) {
	if (!comm_mutex_isLocked)
		CmiAbort("CommUnlock: double unlock!\n");
	comm_mutex_isLocked=0;
	CmiUnlock(comm_mutex);
}
#endif

static void comm_thread(void)
{
  while (1) CommunicationServer(5);
}

static void *call_startfn(void *vindex)
{
  int index = (int)vindex;
  CmiState state = Cmi_state_vector + index;
  pthread_setspecific(Cmi_state_key, state);
  ConverseRunPE(0);
  return 0;
}

static void CmiStartThreads(char **argv)
{
  pthread_t pid;
  int i, ok;
  pthread_attr_t attr;

  CmiMemLock_lock=CmiCreateLock();
  comm_mutex=CmiCreateLock();

  pthread_key_create(&Cmi_state_key, 0);
  Cmi_state_vector =
    (CmiState)calloc(Cmi_mynodesize, sizeof(struct CmiStateStruct));
  for (i=0; i<Cmi_mynodesize; i++)
    CmiStateInit(i+Cmi_nodestart, i, CmiGetStateN(i));
  for (i=1; i<Cmi_mynodesize; i++) {
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    ok = pthread_create(&pid, &attr, call_startfn, (void *)i);
    if (ok<0) PerrorExit("pthread_create"); 
    pthread_attr_destroy(&attr);
  }
  pthread_setspecific(Cmi_state_key, Cmi_state_vector);
  ok = pthread_create(&pid, NULL, (void *(*)(void *))comm_thread, 0);
  if (ok<0) PerrorExit("pthread_create comm"); 
}

#endif


/***********************************************************
 * SMP Idle Locking
 *   In an SMP system, idle processors need to sleep on a
 * lock so that if a message for them arrives, they can be
 * woken up.
 **********************************************************/

#if CMK_SHARED_VARS_NT_THREADS

static void CmiIdleLock_init(CmiIdleLock *l) {
  l->hasMessages=0;
  l->isSleeping=0;
  l->sem=CreateSemaphore(NULL,0,1, NULL);
}

static void CmiIdleLock_sleep(CmiIdleLock *l,int msTimeout) {
  if (l->hasMessages) return;
  l->isSleeping=1;
  MACHSTATE(4,"Processor going to sleep {")
  WaitForSingleObject(l->sem,msTimeout);
  MACHSTATE(4,"} Processor awake again")
  l->isSleeping=0;
}

static void CmiIdleLock_addMessage(CmiIdleLock *l) {
  l->hasMessages=1;
  if (l->isSleeping) { /*The PE is sleeping on this lock-- wake him*/  
    MACHSTATE(4,"Waking sleeping processor")
    ReleaseSemaphore(l->sem,1,NULL);
  }
}
static void CmiIdleLock_checkMessage(CmiIdleLock *l) {
  l->hasMessages=0;
}

#elif CMK_SHARED_VARS_POSIX_THREADS_SMP

static void CmiIdleLock_init(CmiIdleLock *l) {
  l->hasMessages=0;
  l->isSleeping=0;
  pthread_mutex_init(&l->mutex,NULL);
  pthread_cond_init(&l->cond,NULL);
}

static void getTimespec(int msFromNow,struct timespec *dest) {
  struct timeval cur;
  int secFromNow;
  /*Get the current time*/
  gettimeofday(&cur,NULL);
  dest->tv_sec=cur.tv_sec;
  dest->tv_nsec=cur.tv_usec*1000;
  /*Add in the wait time*/
  secFromNow=msFromNow/1000;
  msFromNow-=secFromNow*1000;
  dest->tv_sec+=secFromNow;
  dest->tv_nsec+=1000*1000*msFromNow;
  /*Wrap around if we overflowed the nsec field*/
  while (dest->tv_nsec>1000000000u) {
    dest->tv_nsec-=1000000000;
    dest->tv_sec++;
  }
}

static void CmiIdleLock_sleep(CmiIdleLock *l,int msTimeout) {
  struct timespec wakeup;

  if (l->hasMessages) return;
  l->isSleeping=1;
  MACHSTATE(4,"Processor going to sleep {")
  pthread_mutex_lock(&l->mutex);
  getTimespec(msTimeout,&wakeup);
  while (!l->hasMessages)
    if (ETIMEDOUT==pthread_cond_timedwait(&l->cond,&l->mutex,&wakeup))
      break;
  pthread_mutex_unlock(&l->mutex);
  MACHSTATE(4,"} Processor awake again")
  l->isSleeping=0;
}
static void CmiIdleLock_wakeup(CmiIdleLock *l) {
  l->hasMessages=1; 
  MACHSTATE(4,"Waking sleeping processor")
  /*The PE is sleeping on this condition variable-- wake him*/
  pthread_mutex_lock(&l->mutex);
  pthread_cond_signal(&l->cond);
  pthread_mutex_unlock(&l->mutex);
}

static void CmiIdleLock_addMessage(CmiIdleLock *l) {
  if (l->isSleeping) CmiIdleLock_wakeup(l);
  l->hasMessages=1;
}
static void CmiIdleLock_checkMessage(CmiIdleLock *l) {
  l->hasMessages=0;
}
#else
#define CmiIdleLock_sleep(x) /*empty*/

static void CmiIdleLock_init(CmiIdleLock *l) {
  l->hasMessages=0;
}
static void CmiIdleLock_addMessage(CmiIdleLock *l) {
  l->hasMessages=1;
}
static void CmiIdleLock_checkMessage(CmiIdleLock *l) {
  l->hasMessages=0;
}
#endif

void CmiStateInit(int pe, int rank, CmiState state)
{
  state->pe = pe;
  state->rank = rank;
  state->recv = PCQueueCreate();
  state->localqueue = CdsFifo_Create();
  CmiIdleLock_init(&state->idle);
#if CMK_NODE_QUEUE_AVAILABLE
  CsvInitialize(CmiNodeLock, CmiNodeRecvLock);
  CsvInitialize(PCQueue, NodeRecv);
  if (rank==0) {
    CsvAccess(CmiNodeRecvLock) = CmiCreateLock();
    CsvAccess(NodeRecv) = PCQueueCreate();
  }
#endif
}


