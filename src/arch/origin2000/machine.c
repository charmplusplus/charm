#include <sys/types.h>
#include <sys/sysmp.h>
#include <sys/sysinfo.h>
#include <sys/prctl.h>
#include <sys/pda.h>
#include <ulocks.h>
#include <math.h>
#include "converse.h"

usptr_t *arena;
static barrier_t *barr;

extern void *FIFO_Create(void);
extern void FIFO_EnQueue(void *, void *);

#define BLK_LEN  512
typedef struct {
  usema_t  *sema;
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
#define g_malloc(s)  usmalloc(s,arena)
#define g_free(p)    usfree(p,arena)


CpvDeclare(void*, CmiLocalQueue);
int Cmi_mype;
int Cmi_numpes;
int Cmi_myrank;

static int nthreads;
static int requested_npe;

static void threadInit(void *arg);

int membusy;

void CmiMemLock() {membusy=1;}
void CmiMemUnlock() {membusy=0;}

void *CmiAlloc(int size)
{
  char *res;
  res =(char *) malloc(size+8);
  if (res==(char *)0) { 
    CmiError("%d:Memory allocation failed.",CmiMyPe()); 
    abort();
  }
  ((int *)res)[0]=size;
  return (void *)(res+8);
}

int CmiSize(void *blk)
{
  return ((int *)( ((char *) blk) - 8))[0];
}

void CmiFree(void *blk)
{
  free( ((char *)blk) - 8);
}


#if 0
void *CmiSvAlloc(int size)
{
  char *res;
  res =(char *) usmalloc(size+8,arena);
  if (res==0) {
    CmiError("Memory allocation failed.");
    abort();
  }
  ((int *)res)[0]=size;
  return (void *)(res+8);
}

void CmiSvFree(blk)
char *blk;
{
  usfree(blk-8, arena);
}
#endif

int CmiAsyncMsgSent(CmiCommHandle msgid)
{
  return 0;
}


typedef struct {
  void       *argv;
  CmiStartFn fn;
  int        argc;
  int        npe;
  int        mype;
  int        usched;
  int        initret;
} USER_PARAMETERS;

USER_PARAMETERS usrparam;

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int i;
 
  i =  0;
  requested_npe = 0; 
  while (argv[i] != 0) {
    if (strcmp(argv[i], "+p") == 0) {
      sscanf(argv[i+1], "%d", &requested_npe);
      break;
    } else {
      if (sscanf(argv[i], "+p%d", &requested_npe) == 1) break;
    }
    i++;
  }

  if (requested_npe <= 0)
  {
    CmiError("Error: requested number of processors is invalid %d\n",
              requested_npe);
    abort();
  }

  usconfig(CONF_INITUSERS, requested_npe+1);
  usconfig(CONF_ARENATYPE, US_SHAREDONLY);
  if(usconfig(CONF_INITSIZE,  16<<20)==(-1)) {
    CmiPrintf("Cannot set size of arena\n");
    abort();
  }
  arena = usinit("/dev/zero");
  if(arena == (usptr_t *) NULL) {
    CmiError("Cannot allocate arena\n");
    abort();
  }
  barr = new_barrier(arena);
  init_barrier(barr);
  nthreads = requested_npe;

  usrparam.argc = argc;
  usrparam.argv = argv;
  usrparam.npe = requested_npe;
  usrparam.fn = fn;
  usrparam.initret = initret;
  usrparam.usched = usched;

  MsgQueue=(McQueue **)g_malloc(requested_npe*sizeof(McQueue *));
  if (MsgQueue == (McQueue **)0) {
    CmiError("Cannot Allocate Memory...\n");
    abort();
  }
  for(i=0; i<requested_npe; i++) 
    MsgQueue[i] = McQueueCreate();

  for(i=0; i<requested_npe; i++) {
    usrparam.mype = i;
    sproc(threadInit, PR_SFDS, (void *)&usrparam);
  }
  for(i=0;i<requested_npe;i++)
    wait(0);
}

static void neighbour_init(int);

static void threadInit(void *arg)
{
  USER_PARAMETERS *usrparam;
  usrparam = (USER_PARAMETERS *) arg;


  CpvInitialize(void*, CmiLocalQueue);
  Cmi_mype  = usrparam->mype;
  Cmi_myrank = 0;
  Cmi_numpes =  usrparam->npe;
#ifdef DEBUG
  printf("thread %d/%d started \n", CmiMyPe(), CmiNumPes());
#endif

  ConverseCommonInit(usrparam->argv);
  CthInit(usrparam->argv);
  neighbour_init(Cmi_mype);
  CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
  CmiSpanTreeInit();
  CmiTimerInit();
  usadd(arena);
  if (usrparam->initret==0) {
    usrparam->fn(usrparam->argc, usrparam->argv);
    if (usrparam->usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}


void ConverseExit(void)
{
  ConverseCommonExit();
  barrier(barr, nthreads);
}


void CmiDeclareArgs(void)
{
}


void CmiNotifyIdle()
{
}

void *CmiGetNonLocal()
{
  return McQueueRemoveFromFront(MsgQueue[CmiMyPe()]);
}


void CmiSyncSendFn(int destPE, int size, char *msg)
{
  char *buf;

  buf=(void *)g_malloc(size+8);
  if(buf==(void *)0) {
    CmiError("Cannot allocate memory of size %d!\n", size);
    abort();
  }
  ((int *)buf)[0]=size;
  buf += 8;

#ifdef DEBUG
  printf("Message of size %d allocated\n", CmiSize(buf));
#endif
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
  if (Cmi_mype==destPE) {
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  } else {
    CmiSyncSendFn(destPE, size, msg);
    CmiFree(msg);
  }
}

void CmiSyncBroadcastFn(int size, char *msg)
{
  int i;
  for(i=0; i<CmiNumPes(); i++)
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

#if 0
void CmiNodeBarrier()
{
  barrier(barr,nthreads);
}
#endif

/* ********************************************************************** */
/* The following functions are required by the load balance modules       */
/* ********************************************************************** */

static int _MC_neighbour[4]; 
static int _MC_numofneighbour;

long CmiNumNeighbours(int node)
{
  if (node == CmiMyPe() ) 
   return  _MC_numofneighbour;
  else 
   return 0;
}


void CmiGetNodeNeighbours(int node, int *neighbours)
{
  int i;

  if (node == CmiMyPe() )
    for(i=0; i<_MC_numofneighbour; i++) 
      neighbours[i] = _MC_neighbour[i];
}


int CmiNeighboursIndex(int node, int neighbour)
{
  int i;

  for(i=0; i<_MC_numofneighbour; i++)
    if (_MC_neighbour[i] == neighbour) return i;
  return(-1);
}


static void neighbour_init(int p)
{
  int a,b,n;

  n = CmiNumPes();
  a = (int)sqrt((double)n);
  b = (int) ceil( ((double)n / (double)a) );

   
  _MC_numofneighbour = 0;
   
  /* east neighbour */
  if( (p+1)%b == 0 )
    n = p-b+1;
  else {
    n = p+1;
    if(n>=CmiNumPes()) 
      n = (a-1)*b; /* west-south corner */
  }
  if(neighbour_check(p,n) ) 
    _MC_neighbour[_MC_numofneighbour++] = n;

  /* west neigbour */
  if( (p%b) == 0) {
    n = p+b-1;
  if(n >= CmiNumPes()) 
    n = CmiNumPes()-1;
  } else
    n = p-1;
  if(neighbour_check(p,n) ) 
    _MC_neighbour[_MC_numofneighbour++] = n;

  /* north neighbour */
  if( (p/b) == 0) {
    n = (a-1)*b+p;
    if(n >= CmiNumPes()) 
      n = n-b;
  } else
    n = p-b;
  if(neighbour_check(p,n) ) 
    _MC_neighbour[_MC_numofneighbour++] = n;
    
  /* south neighbour */
  if( (p/b) == (a-1) )
    n = p%b;
  else {
    n = p+b;
    if(n >= CmiNumPes()) 
      n = n%b;
  } 
  if(neighbour_check(p,n) ) 
    _MC_neighbour[_MC_numofneighbour++] = n;
}

static neighbour_check(int p, int n)
{
    int i; 
    if(n==p) 
      return 0;
    for(i=0; i<_MC_numofneighbour; i++) 
      if (_MC_neighbour[i] == n) return 0;
    return 1; 
}

/* ****************************************************************** */
/*    The following internal functions implements FIFO queues for     */
/*    messages. These queues are shared among threads                 */
/* ****************************************************************** */

static void ** AllocBlock(unsigned int len)
{
  void ** blk;

  blk=(void **)g_malloc(len*sizeof(void *));
  if(blk==(void **)0) {
    CmiError("Cannot Allocate Memory!\n");
    abort();
  }
  return blk;
}

static void 
SpillBlock(void **srcblk, void **destblk, unsigned int first, unsigned int len)
{
  memcpy(destblk, &srcblk[first], (len-first)*sizeof(void *));
  memcpy(&destblk[len-first],srcblk,first*sizeof(void *));
}

McQueue * McQueueCreate(void)
{
  McQueue *queue;

  queue = (McQueue *) g_malloc(sizeof(McQueue));
  if(queue==(McQueue *)0) {
    CmiError("Cannot Allocate Memory!\n");
    abort();
  }
  queue->sema = usnewsema(arena, 1);
  usinitsema(queue->sema, 1);
  queue->blk = AllocBlock(BLK_LEN);
  queue->blk_len = BLK_LEN;
  queue->first = 0;
  queue->len = 0;
  queue->maxlen = 0;
  return queue;
}

void McQueueAddToBack(McQueue *queue, void *element)
{
#ifdef DEBUG
  printf("[%d] Waiting for lock\n",CmiMyPe());
#endif
  uspsema(queue->sema);
#ifdef DEBUG
  printf("[%d] Acquired lock\n",CmiMyPe());
#endif
  if(queue->len==queue->blk_len) {
    void **blk;

    queue->blk_len *= 3;
    blk = AllocBlock(queue->blk_len);
    SpillBlock(queue->blk, blk, queue->first, queue->len);
    g_free(queue->blk);
    queue->blk = blk;
    queue->first = 0;
  }
  queue->blk[(queue->first+queue->len++)%queue->blk_len] = element;
  if(queue->len>queue->maxlen)
    queue->maxlen = queue->len;
  usvsema(queue->sema);
#ifdef DEBUG
  printf("[%d] released lock\n",CmiMyPe());
#endif
}


void * McQueueRemoveFromFront(McQueue *queue)
{
  void *element;
  void *localmsg = (void *) 0;
  uspsema(queue->sema);
  element = (void *) 0;
  if(queue->len) {
    element = queue->blk[queue->first++];
    queue->first = (queue->first+queue->blk_len)%queue->blk_len;
    queue->len--;
  }
  usvsema(queue->sema);
  if(element) {
#ifdef DEBUG
    printf("[%d] received message of size %d\n", CmiMyPe(), CmiSize(element));
#endif
    localmsg = CmiAlloc(CmiSize(element));
    memcpy(localmsg, element, CmiSize(element));
    g_free((char *)element-8);
  }
  return localmsg;
}
