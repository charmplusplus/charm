/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <spp_prog_model.h>
#include <memory.h>
#include <cps.h>
#include <math.h>
#include "converse.h"
#include "fifo.h"


#define BLK_LEN		512
typedef struct {
  mem_sema_t sema;
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
#define g_malloc(s)  memory_class_malloc(s,NEAR_SHARED_MEM)


CpvDeclare(void*, CmiLocalQueue);
node_private int Cmi_numpes;

static node_private barrier_t barrier;

static void threadInit();

/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
  CmiError(message);
  exit(1);
}



void *CmiSvAlloc(size)
int size;
{
  char *res;
  res = (char *) memory_class_malloc(size+2*sizeof(int),NEAR_SHARED_MEM);
  if (res==0) CmiAbort("Memory allocation failed.");
  ((int *)res)[0]=size;
  ((int *)res)[1]=(-1);
  return (void *)(res+2*sizeof(int));
}

void CmiSvFree(blk)
char *blk;
{
  free(blk-2*sizeof(int));
}



CmiNotifyIdle()
{
}

CmiNodeLock CmiCreateLock(void)
{
  CmiNodeLock *plock = (CmiNodeLock *)malloc(sizeof(CmiNodeLock));
  cps_mutex_alloc(*plock);
  return *plock;
}

int CmiProbeLock(CmiNodeLock lock)
{
  if(cps_mutex_trylock(lock) == 0){
    cps_mutex_unlock(lock);
    return 1;
  } else {
    return 0;
  }
}


int CmiAsyncMsgSent(msgid)
CmiCommHandle msgid;
{
   return 1;
}

typedef struct {
   CmiStartFn fn;
   int usched;
} USER_PARAMETERS;


static node_private int Cmi_argc;
static node_private char** Cmi_argv;
static node_private USER_PARAMETERS Cmi_param;
static node_private CmiStartFn Cmi_fn;
static node_private int Cmi_usched;
static node_private int Cmi_initret;

void ConverseInit(int argc, char** argv, CmiStartFn fn, int usched, int initret)
{
    int i;
    spawn_sym_t request;

    /* figure out number of processors required */
    i =  0;
    Cmi_numpes = 0; 
    CmiGetArgInt(argv,"+p",&Cmi_numpes);
    if (Cmi_numpes <= 0)
      CmiAbort("Invalid number of processors\n");

    Cmi_argc = argc;
    Cmi_argv = argv;
    Cmi_fn = fn;
    Cmi_usched = usched;
    Cmi_initret = initret;

    request.node = CPS_SAME_NODE;
    request.min  = Cmi_numpes;
    request.max  = Cmi_numpes;
    request.threadscope = CPS_THREAD_PARALLEL;
   
    if(cps_barrier_alloc(&barrier)!=0)
      CmiAbort("Cannot Alocate Barrier\n");

    MsgQueue=(McQueue **)g_malloc(Cmi_numpes*sizeof(McQueue *));
    if (MsgQueue == (McQueue **)0) {
	CmiAbort("Cannot Allocate Memory...\n");
    }
    for(i=0; i<Cmi_numpes; i++) MsgQueue[i] = McQueueCreate();

    if (cps_ppcall(&request, threadInit ,(void *) 0) < 0) {
	CmiAbort("Cannot create threads...\n");
    } 
    cps_barrier_free(&barrier);

}

void ConverseExit(void)
{
   ConverseCommonExit();
   cps_barrier(&barrier,&Cmi_numpes);
}

static void threadInit(arg)
void *arg;
{
    char **argv=CmiCopyArgs(Cmi_argv);
    CpvInitialize(void*, CmiLocalQueue);

    ConverseCommonInit(argv);
    neighbour_init(CmiMyPe());
    CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
    CmiSpanTreeInit();
    CmiTimerInit();
    if (Cmi_initret==0) {
      Cmi_fn(CmiGetArgc(argv), argv);
      if (Cmi_usched==0) CsdScheduler(-1);
      ConverseExit();
    }
}

void *CmiGetNonLocal(void)
{
     int *buf, *msg;
     int size;
     msg = McQueueRemoveFromFront(MsgQueue[CmiMyPe()]);
     if(msg==0)
       return 0;
     size = *(msg-1);
     if((buf = (int *) CmiAlloc(size))==0)
       CmiAbort("Cannot allocate memory!\n");
     memcpy(buf, msg, size);
     free(msg-1);
     return buf;
}


void CmiSyncSendFn(int destPE, int size, char *msg)
{
   int *buf;

   buf=(int *)g_malloc(size+sizeof(int));
   if(buf==(void *)0) {
     CmiAbort("Cannot allocate memory!\n");
   }
   *buf++ = size;
   memcpy((char *)buf,(char *)msg,size);
   McQueueAddToBack(MsgQueue[destPE],buf); 
   CQdCreate(CpvAccess(cQdState), 1);
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
    CQdCreate(CpvAccess(cQdState), 1);
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
    for(i=0; i<CmiNumPes(); i++) CmiSyncSendFn(i,size,msg);
}


CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)
{
    CmiSyncBroadcastAllFn(size, msg);
    return 0; 
}


void CmiFreeBroadcastAllFn(int size, char *msg)
{
    int i;
    for(i=0; i<CmiNumPes(); i++)
       if (CmiMyPe() != i) CmiSyncSendFn(i,size,msg);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
    CQdCreate(CpvAccess(cQdState), 1);
}



void CmiNodeBarrier()
{
   if(cps_barrier(&barrier,&Cmi_numpes)!=0)
     CmiAbort("Error in Barrier\n");
}





/* ****************************************************************** */
/*    The following internal functions implements FIFO queues for     */
/*    messages. These queues are shared among threads                 */
/* ****************************************************************** */



static void ** AllocBlock(unsigned int len)
{
	void ** blk;

	blk=(void **)g_malloc(len*sizeof(void *));
	if(blk==(void **)0)
	{
		CmiAbort("Cannot Allocate Memory!\n");
	}
	return blk;
}

static void SpillBlock(void **srcblk, void **destblk, int first, int len)
{
	memcpy(destblk, &srcblk[first], (len-first)*sizeof(void *));
	memcpy(&destblk[len-first],srcblk,first*sizeof(void *));
}

static McQueue * McQueueCreate(void)
{
	McQueue *queue;
	int one = 1;

	queue = (McQueue *) g_malloc(sizeof(McQueue));
	if(queue==(McQueue *)0)
	{
		CmiError("Cannot Allocate Memory!\n");
		exit(1);
	}
	m_init32(&(queue->sema), &one);
	queue->blk = AllocBlock(BLK_LEN);
	queue->blk_len = BLK_LEN;
	queue->first = 0;
	queue->len = 0;
	queue->maxlen = 0;
	return queue;
}

static 
void
McQueueAddToBack(queue, element)
McQueue *queue;
void  *element;
{
	m_lock(&(queue->sema));
	if(queue->len==queue->blk_len)
	{
		void **blk;

		queue->blk_len *= 3;
		blk = AllocBlock(queue->blk_len);
		SpillBlock(queue->blk, blk, queue->first, queue->len);
		free(queue->blk);
		queue->blk = blk;
		queue->first = 0;
	}
	queue->blk[(queue->first+queue->len++)%queue->blk_len] = element;
	if(queue->len>queue->maxlen)
		queue->maxlen = queue->len;
	m_unlock(&(queue->sema));
}


static
void *
McQueueRemoveFromFront(queue)
McQueue *queue;
{
	void *element;
	m_lock(&(queue->sema));
	element = (void *) 0;
	if(queue->len)
	{
		element = queue->blk[queue->first++];
		queue->first = (queue->first+queue->blk_len)%queue->blk_len;
		queue->len--;
	}
	m_unlock(&(queue->sema));
	return element;
}
