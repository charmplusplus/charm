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
 * Revision 2.16  1995-10-19 20:40:17  jyelon
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
 * Cmi_mype, Cmi_numpe and CmiLocalQueuea are accessed thru macros now
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
CpvDeclare(int, Cmi_mype);
CpvDeclare(int, Cmi_numpe);

static barrier_t barrier;
static barrier_t *barr;
static int *nthreads;
static int requested_npe;

static void mycpy();
static void threadInit();



double CmiTimer()
{
    return (double) 0.0;
}


static void CmiTimerInit()
{
}



void *CmiAlloc(size)
int size;
{
char *res;
res =(char *) memory_class_malloc(size+8,THREAD_PRIVATE_MEM);
if (res==0) printf("Memory allocation failed.");
((int *)res)[0]=size;
return (void *)(res+8);
}

int CmiSize(blk)
void *blk;
{
return ((int *)( ((char *) blk) - 8))[0];
}

void CmiFree(blk)
void *blk;
{
free( ((char *)blk) - 8);
}


void *CmiSvAlloc(size)
int size;
{
char *res;
res =(char *)memory_class_malloc(size+8,NODE_PRIVATE_MEM);
if (res==0) printf("Memory allocation failed.");
((int *)res)[0]=size;
return (void *)(res+8);
}

void CmiSvFree(blk)
char *blk;
{
free(blk-8);
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
} USER_PARAMETERS;



main(argc,argv)
int argc;
char *argv[];
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
       printf("Error: requested number of processors is invalid %d\n",requested_npe);
       exit();
    }


    usrparam.argc = argc;
    usrparam.argv = (void *) argv;
    usrparam.npe  = requested_npe;
    request.node = CPS_ANY_NODE;
    request.min  = requested_npe;
    request.max  = requested_npe;
    request.threadscope = CPS_THREAD_PARALLEL;

   
    nthreads = &requested_npe;
    barr     = &barrier; 

    cps_barrier_alloc(barr);

    MsgQueue=(McQueue **)g_malloc(requested_npe*sizeof(McQueue *));
    for(i=0; i<requested_npe; i++) MsgQueue[i] = McQueueCreate();

    cps_ppcall(&request, threadInit ,arg); 
    cps_barrier_free(barr);

}

static void threadInit(arg)
void *arg;
{
    USER_PARAMETERS *usrparam;
    usrparam = (USER_PARAMETERS *) arg;

    CpvInitialize(int, Cmi_mype);
    CpvInitialize(int, Cmi_numpe);
    CpvInitialize(void*, CmiLocalQueue);

    CpvAccess(Cmi_mype)  = my_thread();
    CpvAccess(Cmi_numpe) =  usrparam->npe;

    user_main(usrparam->argc,usrparam->argv);
}


void CmiInitMc(argv)
char *argv[];
{
    neighbour_init(CpvAccess(Cmi_mype));
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
        ((int *)buf)[0]=size;
        buf += 8;

        if(buf==(void *)0)
        {
                printf("Cannot allocate memory!\n");
                exit(1);
        }

        mycpy((double *)buf,(double *)msg,size);
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
    CmiSyncSendFn(destPE,size,msg);
    CmiFree(msg);
}

void CmiSyncBroadcastFn(size, msg)
int  size;
char *msg;
{
       int i;
       for(i=0; i<CmiNumPe(); i++)
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
    void *env;
    void *buf;

    int i;
    for(i=0; i<CmiNumPe(); i++)
       if (CmiMyPe() != i) CmiSyncSendFn(i,size,msg);

    buf=(void *)CmiAlloc(size);

    if(buf==(void *)0)
        {
                printf("Cannot allocate memory!\n");
                exit(1);
        }

    mycpy((double *)buf,(double *)msg,size);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),buf);
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
    for(i=0; i<CmiNumPe(); i++)
       if (CmiMyPe() != i) CmiSyncSendFn(i,size,msg);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
}


CmiMyRank()
{
   /* to be implemented */
   return CmiMyPe();
}



CmiNodeBarrier()
{
   cps_barrier(barr,nthreads);
}





static void mycpy(double *dst, double *src, int bytes)
{
        unsigned char *cdst, *csrc;

        while(bytes>8)
        {
                *dst++ = *src++;
                bytes -= 8;
        }
        cdst = (unsigned char *) dst;
        csrc = (unsigned char *) src;
        while(bytes)
        {
                *cdst++ = *csrc++;
                bytes--;
        }
}





/* ********************************************************************** */
/* For backward compatibility some of the previous functionality retained */
/* ********************************************************************** */

McTimerInit()
{
}

unsigned int McTimer()
{
}

int McHostPeNum()
{
}

int McMainPeNum()
{
        return 0;
}

McUTimerInit()
{
return 0;
}

unsigned int McUTimer()
{
return 0;
}

McHTimer()
{
return 0;
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

    a = (int) floor(sqrt((double)CmiNumPe()));
    b = (int) ceil( ((double)CmiNumPe() / (double)a) );

   
    _MC_numofneighbour = 0;
   
    /* east neighbour */
    if ( (p+1)%b == 0 )
           n = p-b+1;
    else {
           n = p+1;
           if (n>=CmiNumPe()) n = (a-1)*b; /* west-south corner */
    }
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* west neigbour */
    if ( (p%b) == 0) {
          n = p+b-1;
          if (n >= CmiNumPe()) n = CmiNumPe()-1;
       }
    else
          n = p-1;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* north neighbour */
    if ( (p/b) == 0) {
          n = (a-1)*b+p;
          if (n >= CmiNumPe()) n = n-b;
       }
    else
          n = p-b;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;
    
    /* south neighbour */
    if ( (p/b) == (a-1) )
           n = p%b;
    else {
           n = p+b;
           if (n >= CmiNumPe()) n = n%b;
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
		printf("Cannot Allocate Memory!\n");
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

McQueue *
McQueueCreate()
{
	McQueue *queue;
	int one = 1;

	queue = (McQueue *) g_malloc(sizeof(McQueue));
	if(queue==(McQueue *)0)
	{
		printf("Cannot Allocate Memory!\n");
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
