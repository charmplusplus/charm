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
 * Revision 2.30  1998-02-13 23:55:43  pramacha
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


#define FreeFn		free
#define BLK_LEN		512
typedef struct
	{
		mem_sema_t sema;
		void     **blk;
		unsigned int blk_len;
		unsigned int first;
		unsigned int len;
		unsigned int maxlen;
}McQueue;

static McQueue *McQueueCreate(void);
static void McQueueAddToBack(McQueue *queue, void *element);
static void *McQueueRemoveFromFront(McQueue *queue);
static McQueue **MsgQueue;
#define g_malloc(s)  memory_class_malloc(s,NEAR_SHARED_MEM)


CpvDeclare(void*, CmiLocalQueue);
int Cmi_numpes;
int Cmi_mynodesize;

static node_private barrier_t barrier;
static node_private barrier_t *barr;
static node_private int *nthreads;
static node_private int requested_npe;

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
  res =(char *)memory_class_malloc(size+8,NEAR_SHARED_MEM);
  if (res==0) CmiError("Memory allocation failed.");
  ((int *)res)[0]=size;
  return (void *)(res+8);
}

void CmiSvFree(blk)
char *blk;
{
  free(blk-8);
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
   int argc;
   void *argv;
   int  npe;
   CmiStartFn fn;
   int usched;
} USER_PARAMETERS;



void ConverseInit(int argc, char** argv, CmiStartFn fn, int usched, int initret)
{
    int i;
    USER_PARAMETERS usrparam;
    void *arg = & usrparam;
 
    spawn_sym_t request;


    /* figure out number of processors required */
    i =  0;
    requested_npe = 0; 
    while (argv[i] != 0)
    {
         if (strcmp(argv[i], "+p") == 0)
           {
                 sscanf(argv[i + 1], "%d", &requested_npe);
                 break;
           }
         else if (sscanf(argv[i], "+p%d", &requested_npe) == 1) break;
         i++;
    }


    if (requested_npe <= 0)
    {
       CmiError("Error: requested number of processors is invalid %d\n",requested_npe);
       exit();
    }


    printf("Requested NP = %d\n", requested_npe);
    usrparam.argc = argc;
    usrparam.argv = (void *) argv;
    usrparam.npe  = requested_npe;
    usrparam.fn = fn;
    usrparam.usched = usched;
    request.node = CPS_ANY_NODE;
    request.min  = requested_npe;
    request.max  = requested_npe;
    request.threadscope = CPS_THREAD_PARALLEL;

   
    nthreads = &requested_npe;
    barr     = &barrier; 

    cps_barrier_alloc(barr);

    MsgQueue=(McQueue **)g_malloc(requested_npe*sizeof(McQueue *));
    if (MsgQueue == (McQueue **)0) {
	CmiError("Cannot Allocate Memory...\n");
	exit(1);
    }
    for(i=0; i<requested_npe; i++) MsgQueue[i] = McQueueCreate();

    if (cps_ppcall(&request, threadInit ,arg) < 0) {
	CmiError("Cannot created threads...\n");
	exit(1);
    } 
    cps_barrier_free(barr);

}

void CmiInitMc(char **);

void ConverseExit(void)
{
   ConverseCommonExit();
   cps_barrier(barr,nthreads);
}

static void threadInit(arg)
void *arg;
{
    USER_PARAMETERS *usrparam;
    usrparam = (USER_PARAMETERS *) arg;

    CpvInitialize(void*, CmiLocalQueue);

    Cmi_numpes =  usrparam->npe;
    Cmi_mynodesize  = Cmi_numpes;

    CthInit(usrparam->argv);
    ConverseCommonInit(usrparam->argv);
    CmiInitMc(usrparam->argv);
    usrparam->fn(usrparam->argc,usrparam->argv);
    if (usrparam->usched==0) CsdScheduler(-1);
    ConverseExit();
}


void CmiInitMc(argv)
char *argv[];
{
    neighbour_init(CmiMyPe());
    CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
    CmiSpanTreeInit();
    CmiTimerInit();
}



CmiExit()
{
}


CmiDeclareArgs()
{
}


void *CmiGetNonLocal()
{
     return McQueueRemoveFromFront(MsgQueue[CmiMyPe()]);
}


void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char *msg;
{
    /* Send the message of "size" bytes to the destPE.
       Return only after the message has been sent, i.e.,
       the buffer (msg) is free for re-use. */
        char *buf;

        buf=(void *)g_malloc(size+8);
        if(buf==(void *)0)
        {
                CmiError("Cannot allocate memory!\n");
                exit(1);
        }
        ((int *)buf)[0]=size;
        buf += 8;


        memcpy((double *)buf,(double *)msg,size);
        McQueueAddToBack(MsgQueue[destPE],buf); 
}


CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
int destPE;
int size;
char *msg;
{
    CmiSyncSendFn(destPE, size, msg); 
    return 0;
}


void CmiFreeSendFn(destPE, size, msg)
int destPE;
int size;
char  *msg;
{
        McQueueAddToBack(MsgQueue[destPE],msg); 
}

void CmiSyncBroadcastFn(size, msg)
int  size;
char *msg;
{
       int i;
       for(i=0; i<CmiNumPes(); i++)
         if (CmiMyPe() != i) CmiSyncSendFn(i,size,msg);
}

CmiCommHandle CmiAsyncBroadcastFn(size, msg)
int  size;
char *msg;
{
    CmiSyncBroadcastFn(size, msg);
    return 0;
}

void CmiFreeBroadcastFn(size, msg)
int  size;
char *msg;
{
    CmiSyncBroadcastFn(size,msg);
    CmiFree(msg);
}

void CmiSyncBroadcastAllFn(size, msg)
int  size;
char *msg;
{
    int i;
    for(i=0; i<CmiNumPes(); i++) CmiSyncSendFn(i,size,msg);
}


CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int  size;
char *msg;
{
    CmiSyncBroadcastAllFn(size, msg);
    return 0; 
}


void CmiFreeBroadcastAllFn(size, msg)
int  size;
char *msg;
{
    int i;
    for(i=0; i<CmiNumPes(); i++)
       if (CmiMyPe() != i) CmiSyncSendFn(i,size,msg);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
}



void CmiNodeBarrier()
{
   cps_barrier(barr,nthreads);
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



static void **
AllocBlock(len)
unsigned int len;
{
	void ** blk;

	blk=(void **)g_malloc(len*sizeof(void *));
	if(blk==(void **)0)
	{
		CmiError("Cannot Allocate Memory!\n");
		exit(1);
	}
	return blk;
}

static void
SpillBlock(srcblk, destblk, first, len)
void **srcblk, **destblk;
unsigned int first, len;
{
	memcpy(destblk, &srcblk[first], (len-first)*sizeof(void *));
	memcpy(&destblk[len-first],srcblk,first*sizeof(void *));
}

static
McQueue *
McQueueCreate()
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
		FreeFn(queue->blk);
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
