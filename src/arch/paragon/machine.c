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
