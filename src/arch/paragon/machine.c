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
 * Revision 1.11  1998-06-15 22:14:50  jyelon
 * Added support for ASCI RED
 *
 * Revision 1.10  1998/02/13 23:55:09  pramacha
 * Removed CmiAlloc, CmiFree and CmiSize
 * Added CmiAbort
 *
 * Revision 1.9  1997/12/10 21:01:22  jyelon
 * *** empty log message ***
 *
 * Revision 1.8  1997/10/03 19:51:51  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 1.7  1997/07/26 16:41:41  jyelon
 * *** empty log message ***
 *
 * Revision 1.6  1997/04/25 20:48:11  jyelon
 * Corrected CmiNotifyIdle
 *
 * Revision 1.5  1997/04/24 22:37:00  jyelon
 * Added CmiNotifyIdle
 *
 * Revision 1.4  1997/03/19 04:31:19  jyelon
 * Redesigned ConverseInit
 *
 * Revision 1.3  1997/02/13 09:31:35  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.2  1996/07/15 20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 1.1  1995/11/15 18:17:30  gursoy
 * Initial revision
 *
 * Revision 2.8  1995/11/08  23:29:00  gursoy
 * fixed varSize msg problem
 *
 * Revision 2.7  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.6  1995/10/24  23:08:45  sanjeev
 * fixed CmiTimer
 *
 * Revision 2.5  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 2.4  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.3  1995/09/20  16:03:27  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.2  1995/09/07  22:57:33  gursoy
 * Cmi_mype Cmi_numpes CmiLocalQueue accessed thru macros now
 *
 * Revision 2.1  1995/07/03  17:58:26  gursoy
 * changed charm_main to user_main
 *
 * Revision 2.0  1995/06/23  20:00:01  gursoy
 * Initial Revision
 *
 * Revision 1.1  1995/06/23  19:57:30  gursoy
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <nx.h>
#include <math.h>
#include "converse.h"

#define MSG_TYPE 1
#define PROCESS_PID 0
#define HOST_PID 1
#define ALL_NODES -1


int  Cmi_mype;
int  Cmi_numpes;
CpvDeclare(void*, CmiLocalQueue);


static int _MC_neighbour[4]; 
static int _MC_numofneighbour;
static esize_t hclockinitvalue;
static unsigned int clockinitvalue;
extern unsigned long mclock();

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



/**************************  TIMER FUNCTIONS **************************/
unsigned int utimerinit[2] ;

double CmiTimer()
{
   unsigned int tim[2]; double t;
 
   hwclock(tim);
   return (double)(tim[0]-utimerinit[0])/50000000.0;
}

double CmiWallTimer()
{
   unsigned int tim[2]; double t;
 
   hwclock(tim);
   return (double)(tim[0]-utimerinit[0])/50000000.0;
}

double CmiCpuTimer()
{
   unsigned int tim[2]; double t;
 
   hwclock(tim);
   return (double)(tim[0]-utimerinit[0])/50000000.0;
}

static void CmiTimerInit()
{
   hwclock(utimerinit) ;
}


/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal()
{
     void *env;
     int   msglength; 
     
     if  ( iprobe(MSG_TYPE)  )
	   {
          msglength = infocount();
          env = (void *) CmiAlloc(msglength);
          if (env == 0)
             CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
          else 
             crecv(MSG_TYPE, env, msglength);
          return env;
       }
     else
		return 0;
}

int CmiAsyncMsgSent(c)
CmiCommHandle c ;
{
    return (int) msgdone(c);
}


void CmiReleaseCommHandle(c)
CmiCommHandle c ;
{
}

void CmiNotifyIdle()
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#endif
}

/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
    char *temp;
    if (Cmi_mype == destPE)
       {
          temp = (char *)CmiAlloc(size) ;
          memcpy(temp, msg, size) ;
          FIFO_EnQueue(CpvAccess(CmiLocalQueue), temp);
       }
    else
          csend(MSG_TYPE, msg, size, destPE, PROCESS_PID);
}


CmiCommHandle CmiAsyncSendFn(destPE, size, msg)  
int destPE;
int size;
char * msg;
{
    long msgid;
    msgid = isend(MSG_TYPE, msg, size, destPE, PROCESS_PID);
    return msgid;
}





void CmiFreeSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
    if (Cmi_mype == destPE)
       {
          FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
       }
    else
       {  
          csend(MSG_TYPE, msg, size, destPE, PROCESS_PID);
          CmiFree(msg);
       }
}



void CmiSyncBroadcastFn(size, msg)        /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    if (Cmi_numpes > 1) 
       csend(MSG_TYPE, msg, size, ALL_NODES,PROCESS_PID);
}


CmiCommHandle CmiAsyncBroadcastFn(size, msg) /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
        long msgid;
        msgid = isend(MSG_TYPE, msg, size, ALL_NODES, PROCESS_PID);
        return msgid;
}

void CmiFreeBroadcastFn(size, msg)
    int size;
    char *msg;
{
    CmiSyncBroadcastFn(size,msg);
    CmiFree(msg);
}
 
void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
    char *temp;
    if (Cmi_numpes > 1) 
       csend(MSG_TYPE, msg, size, ALL_NODES,PROCESS_PID);
    temp = (char *)CmiAlloc(size) ;
    memcpy(temp, msg, size) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), temp); 
}


CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
        long msgid;
        char *temp;
        msgid = isend(MSG_TYPE, msg, size, ALL_NODES, PROCESS_PID);
        temp = (char *)CmiAlloc(size) ;
        memcpy(temp, msg, size) ;
        FIFO_EnQueue(CpvAccess(CmiLocalQueue), temp);
        return msgid;
}



void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
    if (Cmi_numpes > 1)
       csend(MSG_TYPE, msg, size, ALL_NODES,PROCESS_PID);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
}



/************************** SETUP ***********************************/

void ConverseExit()
{
  ConverseCommonExit();
  exit(0);
}

void ConverseInit(argc, argv, fn, usched, initret)
int argc;
char *argv[];
CmiStartFn fn;
int usched, initret;
{
  CpvInitialize(void*, CmiLocalQueue);
  Cmi_mype = mynode();
  Cmi_numpes = numnodes();
  neighbour_init(Cmi_mype);
  CpvAccess(CmiLocalQueue)= (void *) FIFO_Create();
  CmiSpanTreeInit();
  CmiTimerInit();
  CthInit(argv);
  ConverseCommonInit(argv);
  if (initret==0) {
    fn(argc, argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

/**********************  LOAD BALANCER NEEDS **********************/



long CmiNumNeighbours(node)
int node;
{
    if (node == Cmi_mype)
     return  _MC_numofneighbour;
    else
     return 0;
}


CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
    int i;

    if (node == Cmi_mype)
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



/* internal functions                                                 */
/* Following functions establishes a two dimensional torus connection */
/* among the procesors (for any number of precessors > 0              */
   
static neighbour_init(p)
int p;
{
    int a,b,n;

    a = (int) floor(sqrt((double) Cmi_numpes));
    b = (int) ceil( ((double)Cmi_numpes / (double)a) );

   
    _MC_numofneighbour = 0;
   
    /* east neighbour */
    if ( (p+1)%b == 0 )
           n = p-b+1;
    else {
           n = p+1;
           if (n>=Cmi_numpes) n = (a-1)*b; /* west-south corner */
    }
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* west neigbour */
    if ( (p%b) == 0) {
          n = p+b-1;
          if (n >= Cmi_numpes) n = Cmi_numpes-1;
       }
    else
          n = p-1;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* north neighbour */
    if ( (p/b) == 0) {
          n = (a-1)*b+p;
          if (n >= Cmi_numpes) n = n-b;
       }
    else
          n = p-b;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;
    
    /* south neighbour */
    if ( (p/b) == (a-1) )
           n = p%b;
    else {
           n = p+b;
           if (n >= Cmi_numpes) n = n%b;
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
