/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include "converse.h"

#define MSG_TYPE 1

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))


CpvDeclare(int, Cmi_mype);
CpvDeclare(int,  Cmi_numpes);
CpvDeclare(void*, CmiLocalQueue);

static int Cmi_dim;

static int process, host, cflag, source, type;
static double uclockinitvalue;
extern double amicclk();

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

/**************************  TIMER FUNCTIONS **************************/

double CmiTimer()
{
  return ( (amicclk() - uclockinitvalue) / 1000000.0 );
}

double CmiWallTimer()
{
  return ( (amicclk() - uclockinitvalue) / 1000000.0 );
}

double CmiCpuTimer()
{
  return ( (amicclk() - uclockinitvalue) / 1000000.0 );
}

static void CmiTimerInit()
{
  uclockinitvalue = amicclk();
}

int CmiAsyncMsgSent(c)
CmiCommHandle c ;
{
    return 1;
}


void CmiReleaseCommHandle(c)
CmiCommHandle c ;
{
}




/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal()
{
        void *env;
        int  msglength;

        type = MSG_TYPE;
        source = -1;  /* dont care */

        if ( (msglength = ntest(&source, &type)) > 0)
        {
               env = (void *)  CmiAlloc(msglength); 
               if (env == 0) 
                  CmiPrintf("*** ERROR *** Memory Allocation Failed.\n");
               CmiSyncReceive(msglength, env);
               return env;
        }
        else
		return 0;
}

void CmiNotifyIdle()
{
#if CMK_WHEN_PROCESSOR_IDLE_USLEEP
  tv.tv_sec=0; tv.tv_usec=5000;
  select(0,0,0,0,&tv);
#endif
}
 
CmiSyncReceive(size, buffer)
int size;
char *buffer;
{
    nread(buffer, size, &source, &type, &cflag);
}


/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
    char *temp;
    if (CpvAccess(Cmi_mype) == destPE)
       {
          temp = (char *)CmiAlloc(size) ;
          memcpy(temp, msg, size) ;
          CdsFifo_Enqueue(CpvAccess(CmiLocalQueue), temp);
       }
    else
          nwrite(msg, size, destPE, MSG_TYPE, &cflag);
    CQdCreate(CpvAccess(cQdState), 1);
}


CmiCommHandle CmiAsyncSendFn(destPE, size, msg)   /* same as sync send for ncube */
int destPE;
int size;
char * msg;
{
    nwrite(msg, size, destPE, MSG_TYPE, &cflag);
    CQdCreate(CpvAccess(cQdState), 1);
    return 0 ;
}


void CmiFreeSendFn(destPE, size, msg)
     int destPE, size;
     char *msg;
{
    if (CpvAccess(Cmi_mype) == destPE)
        CdsFifo_Enqueue(CpvAccess(CmiLocalQueue), msg);
    else
      {
        nwrite(msg, size, destPE, MSG_TYPE, &cflag);
        CmiFree(msg);
      }
    CQdCreate(CpvAccess(cQdState), 1);
}

/*********************** BROADCAST FUNCTIONS **********************/


void CmiSyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
	int i;

	for (i=0; i<CpvAccess(Cmi_numpes); i++)
		if (i != CpvAccess(Cmi_mype))
			nwrite(msg, size, i, MSG_TYPE, &cflag);
        CQdCreate(CpvAccess(cQdState), CpvAccess(Cmi_numpes)-1);
}


CmiCommHandle CmiAsyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
/* Same as sync broadcast for now */
	int i;

	for (i=0; i<CpvAccess(Cmi_numpes); i++)
		if (i != CpvAccess(Cmi_mype))
			nwrite(msg, size, i, MSG_TYPE, &cflag);
        CQdCreate(CpvAccess(cQdState), CpvAccess(Cmi_numpes)-1);
	return 0 ;
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
    	int dest = 0xffff;
	nwrite(msg, size, dest, MSG_TYPE, &cflag); 
        CQdCreate(CpvAccess(cQdState), CpvAccess(Cmi_numpes));
}


CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
    	int dest = 0xffff;
	nwrite(msg, size, dest, MSG_TYPE, &cflag); 
        CQdCreate(CpvAccess(cQdState), CpvAccess(Cmi_numpes));
	return 0 ;
}

void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
	int dest = 0xffff;
	nwrite(msg, size, dest, MSG_TYPE, &cflag); 
        CQdCreate(CpvAccess(cQdState), CpvAccess(Cmi_numpes));
	CmiFree(msg) ; 
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
  CpvInitialize(int, Cmi_mype);
  CpvInitialize(int, Cmi_numpes);
  CpvInitialize(void*, CmiLocalQueue);
  whoami(&CpvAccess(Cmi_mype), &process, &host, &Cmi_dim);
  CpvAccess(Cmi_numpes) = (1 << Cmi_dim) ;
  CpvAccess(CmiLocalQueue)= CdsFifo_Create();
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



