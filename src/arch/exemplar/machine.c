/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.31  1998-03-20 16:08:11  milind
 * Fixed exemplar stuff.
 *
 * Revision 2.30  1998/02/13 23:55:43  pramacha
 * Removed CmiAlloc, CmiFree and CmiSize
 * Added CmiAbort
 *
 * Revision 2.29  1997/12/10 21:01:34  jyelon
 * *** empty log message ***
 *
 * Revision 2.28  1997/10/03 19:51:56  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 2.27  1997/08/01 22:38:50  milind
 * Fixed gatherflat problem.
 * Also fixed optimization flags.
 *
 * Revision 2.26  1997/07/29 16:09:48  milind
 * Added CmiNodeLock macros and functions to the machine layer for all except
 * solaris SMP.
 *
 * Revision 2.25  1997/07/29 16:00:06  milind
 * changed cmi_nodesize into cmi_mynodesize.
 *
 * Revision 2.24  1997/07/23 18:40:24  milind
 * Made charm++ to work on exemplar.
 *
 * Revision 2.23  1997/07/23 16:55:02  milind
 * Made Converse run on Exemplar.
 *
 * Revision 2.22  1997/07/22 18:16:05  milind
 * fixed some exemplar-related bugs.
 *
 * Revision 2.21  1996/11/23 02:25:36  milind
 * Fixed several subtle bugs in the converse runtime for convex
 * exemplar.
 *
 * Revision 2.20  1996/07/15 20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 2.19  1995/11/09 22:00:55  gursoy
 * fixed varsize related bug (CmiFreeSend...)
 *
 * Revision 2.18  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.17  1995/10/19  20:42:45  jyelon
 * added void to CmiNodeBarrier
 *
 * Revision 2.16  1995/10/19  20:40:17  jyelon
 * void * -> char *
 *
 * Revision 2.15  1995/10/19  20:38:08  jyelon
 * unsigned int -> int
 *
 * Revision 2.14  1995/10/13  20:05:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.13  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 2.12  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.11  1995/09/20  16:01:33  gursoy
 * this time really made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.10  1995/09/20  15:58:09  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.9  1995/09/14  18:50:10  milind
 * fixed a small bug - a typo
 *
 * Revision 2.8  1995/09/07  22:40:22  gursoy
 * Cmi_mype, Cmi_numpes and CmiLocalQueuea are accessed thru macros now
 *
 * Revision 2.7  1995/07/05  23:15:29  gursoy
 * minor change in +p code
 *
 * Revision 2.6  1995/07/05  23:07:28  gursoy
 * fixed +p option
 *
 * Revision 2.5  1995/07/03  17:57:37  gursoy
 * changed charm_main to user_main
 *
 * Revision 2.4  1995/06/28  15:31:54  gursoy
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/16  21:42:10  gursoy
 * fixed CmiSyncSend: size field is copied now
 *
 * Revision 2.2  1995/06/09  21:22:00  gursoy
 * Cpv macros moved to converse
 *
 * Revision 2.1  1995/06/09  16:43:47  gursoy
 * Initial implementation of CmiMyRank
 *
 * Revision 2.0  1995/06/08  16:35:12  gursoy
 * Reorganized directory structure
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <spp_prog_model.h>
#include <memory.h>
#include <cps.h>
#include <math.h>
#include "converse.h"


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

void CmiAbort(char *message)
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
   return 0;
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
    for(i=1;i<argc;i++) {
      if(strcmp(argv[i], "+p") == 0) {
        sscanf(argv[i + 1], "%d", &Cmi_numpes);
        break;
      } else if(sscanf(argv[i], "+p%d", &Cmi_numpes) == 1) 
          break;
    }

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
    CpvInitialize(void*, CmiLocalQueue);

    CthInit(Cmi_argv);
    ConverseCommonInit(Cmi_argv);
    neighbour_init(CmiMyPe());
    CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
    CmiSpanTreeInit();
    CmiTimerInit();
    if (Cmi_initret==0) {
      Cmi_fn(Cmi_argc, Cmi_argv);
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
}



void CmiNodeBarrier()
{
   if(cps_barrier(&barrier,&Cmi_numpes)!=0)
     CmiAbort("Error in Barrier\n");
}





/* ********************************************************************** */
/* The following functions are required by the load balance modules       */
/* ********************************************************************** */

static thread_private int _MC_neighbour[4]; 
static thread_private int _MC_numofneighbour;

long CmiNumNeighbours(node)
int node;
{
    if (node == CmiMyPe() ) 
     return  _MC_numofneighbour;
    else 
     return 0;
}


CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
    int i;

    if (node == CmiMyPe() )
       for(i=0; i<_MC_numofneighbour; i++) neighbours[i] = _MC_neighbour[i];

}


int CmiNeighboursIndex(node, neighbour)
int node, neighbour;
{
    int i;

    for(i=0; i<_MC_numofneighbour; i++)
       if (_MC_neighbour[i] == neighbour) return i;
    return(-1);
}


static neighbour_init(p)
int p;
{
    int a,b,n;

    n = CmiNumPes();
    a = (int)sqrt((double)n);
    b = (int) ceil( ((double)n / (double)a) );

   
    _MC_numofneighbour = 0;
   
    /* east neighbour */
    if ( (p+1)%b == 0 )
           n = p-b+1;
    else {
           n = p+1;
           if (n>=CmiNumPes()) n = (a-1)*b; /* west-south corner */
    }
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* west neigbour */
    if ( (p%b) == 0) {
          n = p+b-1;
          if (n >= CmiNumPes()) n = CmiNumPes()-1;
       }
    else
          n = p-1;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* north neighbour */
    if ( (p/b) == 0) {
          n = (a-1)*b+p;
          if (n >= CmiNumPes()) n = n-b;
       }
    else
          n = p-b;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;
    
    /* south neighbour */
    if ( (p/b) == (a-1) )
           n = p%b;
    else {
           n = p+b;
           if (n >= CmiNumPes()) n = n%b;
    } 
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

}

static neighbour_check(p,n)
int p,n;
{
    int i; 
    if (n==p) return 0;
    for(i=0; i<_MC_numofneighbour; i++) if (_MC_neighbour[i] == n) return 0;
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
