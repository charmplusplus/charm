#include <nx.h>
#include <math.h>
#include "converse.h"
#include "fifo.h"

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

void CmiAbort(const char *message)
{
  CmiError(message);
  exit(1);
}


#if CMK_TIMER_USE_DCLOCK
/**************************  TIMER FUNCTIONS **************************/
extern double dclock(void);
double initTime;

double CmiTimer()
{
   return dclock()-initTime;
}

double CmiWallTimer()
{
   return dclock()-initTime;
}

double CmiCpuTimer()
{
   return dclock()-initTime;
}

void CmiTimerInit()
{
  initTime = dclock();
}
#endif

#if CMK_TIMER_USE_SPECIAL
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

void CmiTimerInit()
{
   hwclock(utimerinit) ;
}
#endif

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
  /*  CmiTimerInit(); */
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
