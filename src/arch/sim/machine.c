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
 * Revision 1.12  1997-03-24 16:21:54  milind
 * removed an alignment bug caused by mycpy. Replaced mycpy with memcpy.
 *
 * Revision 1.11  1997/03/19 04:31:30  jyelon
 * Redesigned ConverseInit
 *
 * Revision 1.10  1996/07/24 21:42:32  gursoy
 * fixed CmiTimer superinstall problems
 *
 * Revision 1.9  1996/07/15  20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 1.8  1995/11/08 23:40:58  gursoy
 * fixed varsize msg related bug
 *
 * Revision 1.7  1995/11/08  00:42:13  jyelon
 * *** empty log message ***
 *
 * Revision 1.6  1995/11/08  00:40:20  jyelon
 * *** empty log message ***
 *
 * Revision 1.5  1995/11/07  23:22:56  jyelon
 * Fixed the neighbour functions.
 *
 * Revision 1.4  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 1.3  1995/10/13  22:05:59  gursoy
 * put CmiGrabBuffer init stuff
 *
 * Revision 1.2  1995/10/13  20:05:13  jyelon
 * *** empty log message ***
 *
 * Revision 1.1  1995/10/13  16:08:54  gursoy
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <math.h>
#include "machine.h"
#include "converse.h"

#ifdef CMK_TIMER_SIM_USE_TIMES
#include <sys/times.h>
#include <sys/unistd.h>
#endif
#ifdef CMK_TIMER_SIM_USE_GETRUSAGE
#include <sys/time.h>
#include <sys/resource.h>
#endif



static void **McQueue;

int Cmi_mype;
int Cmi_numpes;


CsvDeclare(int, CsdStopCount);
CpvDeclare(void*, CmiLocalQueue);
CpvExtern(int, CcdNumChecks);
CpvExtern(int, disable_sys_msgs);

double CmiTimer();

static void CsiTimerInit();
static double CsiTimer();


CpvStaticDeclare(int,CmiBufferGrabbed);


void CmiDeliversInit()
{
  CpvInitialize(int, CmiBufferGrabbed);
  CpvAccess(CmiBufferGrabbed) = 0;
}


void CmiGrabBuffer()
{
  CpvAccess(CmiBufferGrabbed) = 1;
}


int CmiDeliverMsgs(maxmsgs)
int maxmsgs;
{
     return maxmsgs;
}

void CmiDeliverSpecificMsg(handler)
int handler;
{
}

CmiUniContextSwitch(i)
int i;
{
     Cmi_mype = i; 
}


void *CmiAlloc(size)
int size;
{
char *res;
res =(char *)malloc(size+8);
if (res==0) printf("Memory allocation failed.");
((int *)res)[0]=size;
return (void *)(res+8);
}

int CmiSize(blk)
void *blk;
{
return ((int *)( ((char *)blk)-8))[0];
}

void CmiFree(blk)
void *blk;
{
free( ((char *)blk)-8);
}




/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
    char *buf;

    buf              =  (char *)malloc(size+8);
    ((int *)buf)[0]  =  size;
    buf += 8;
    memcpy(buf,msg,size);

    sim_send_message(Cmi_mype,buf,size,FALSE,destPE);
}




CmiCommHandle CmiAsyncSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
     CmiSyncSendFn(destPE, size, msg);
     return 0;
}



void CmiFreeSendFn(destPE, size, msg)
int destPE;
int size;
char * msg;
{
     sim_send_message(Cmi_mype,msg,size,FALSE,destPE);
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


void CmiNodeBarrier()
{
}


/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal(){return NULL;}




/*********************** BROADCAST FUNCTIONS **********************/


void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
    int i;
    for(i=0; i<Cmi_numpes; i++)
       if (i!= Cmi_mype) CmiSyncSendFn(i,size,msg);
         
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
}


void CmiSyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    int i;
    for(i=0; i<Cmi_numpes; i++)
       if (i!= Cmi_mype) CmiSyncSendFn(i,size,msg);
}


void CmiSyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
     int i;

     char *buf;

     for(i=0; i<Cmi_numpes; i++)
        if (i!= Cmi_mype) CmiSyncSendFn(i,size,msg);

     buf              =  (char *)malloc(size+8);
     ((int *)buf)[0]  =  size; 
     buf += 8;
     memcpy(buf,msg,size);
     FIFO_EnQueue(CpvAccess(CmiLocalQueue),buf);
}



void CmiFreeBroadcastFn(size, msg)      /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    CmiSyncBroadcastFn(size, msg);
    CmiFree(msg);
}





CmiCommHandle CmiAsyncBroadcastFn(size, msg)	/* ALL_EXCEPT_ME  */
int size;
char * msg;
{
        CmiSyncBroadcastFn(size, msg); 
	return 0 ;
}




CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
        CmiSyncBroadcastAll(size,msg);
	return 0 ;
}




/**********************  LOAD BALANCER NEEDS **********************/

long CmiNumNeighbours(node)
int node;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < Cmi_numpes) count++;
    bit<<1; if (bit > Cmi_numpes) break;
  }
  return count;
}

int CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < Cmi_numpes) neighbours[count++] = neighbour;
    bit<<1; if (bit > Cmi_numpes) break;
  }
  return count;
}
 
int CmiNeighboursIndex(node, nbr)
int node, nbr;
{
  int bit, count=0;
  bit = 1;
  while (1) {
    int neighbour = node ^ bit;
    if (neighbour < Cmi_numpes) { if (nbr==neighbour) return count; count++; }
    bit<<=1; if (bit > Cmi_numpes) break;
  }
  return(-1);
}


/************************** SETUP ***********************************/

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usc, int initret)
{
  char *argvec[1000];
  void simulate();
  int i, requested_npe;
  
  if ((usc)||(initret)) {
    fprintf(stderr,"ConverseInit in SIM version is limited:\n");
    fprintf(stderr," 1. User-Calls-Scheduler mode is not supported.\n");
    fprintf(stderr," 2. ConverseInit-Returns mode is not supported.\n");
    exit(1);
  }
  
  CthInit(argv);
  
  /* figure out number of processors required */
  
  i = 0; requested_npe = 0;
  while (argv[i] != NULL) {
    if (strcmp(argv[i], "+p") == 0) {
      requested_npe = atoi(argv[i+1]);
      break;
    } else if (strncmp(argv[i], "+p", 2)==0) {
      requested_npe = atoi(argv[i]+2);
      break;
    }
    i++;
  }

  if (requested_npe <= 0) {
    printf("Error: requested number of processors is invalid %d\n",
	   requested_npe);
    exit(1);
  }

  Cmi_numpes = requested_npe;
  Cmi_mype   = 0;

  McQueue = (void **) malloc(requested_npe * sizeof(void *)); 
  for(i=0; i<requested_npe; i++) McQueue[i] = (void *) FIFO_Create();
  sim_initialize("sim.param",requested_npe);
  
  CsiTimerInit();
  for(i=0; i<CmiNumPes(); i++) {
    CmiUniContextSwitch(i);
    CpvInitialize(void*, CmiLocalQueue);
    CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
    ConverseCommonInit(argv);
    memcpy(argvec, argv, argc*sizeof(char*));
    fn(argc, argvec);
    CpvAccess(CsdStopFlag) = 0;
  }
  
  CsvAccess(CsdStopCount) = CmiNumPes();
  CmiUniContextSwitch(0);

  while (CsvAccess(CsdStopCount)) simulate();

  exit(0);
}

void CsdExitScheduler()
{
  CpvAccess(CsdStopFlag) = 1;
  CsvAccess(CsdStopCount)--;
}



/* ********************************************************************* */
/*                      SIMULATOR                                        */
/* ********************************************************************* */



#if CMK_TIMER_SIM_USE_TIMES

static struct tms inittime;

static void CsiTimerInit()
{
  times(&inittime);
}

static double CsiTimer()
{
  double currenttime;
  int clk_tck;
    struct tms temp;

    times(&temp);
    clk_tck=sysconf(_SC_CLK_TCK);
    currenttime =
     (((temp.tms_utime - inittime.tms_utime)+
       (temp.tms_stime - inittime.tms_stime))*1.0)/clk_tck;
    return (currenttime);
}

#endif

#if CMK_TIMER_SIM_USE_GETRUSAGE

static struct rusage inittime;

static void CsiTimerInit()
{
  getrusage(0, &inittime);
}


static double CsiTimer() {
  double currenttime;

  struct rusage temp;
  getrusage(0, &temp);
  currenttime =
    (temp.ru_utime.tv_usec - inittime.ru_utime.tv_usec) * 0.000001+
      (temp.ru_utime.tv_sec - inittime.ru_utime.tv_sec) +
        (temp.ru_stime.tv_usec - inittime.ru_stime.tv_usec) * 0.000001+
          (temp.ru_stime.tv_sec - inittime.ru_stime.tv_sec) ;

  return (currenttime);
}

#endif



static double Csi_global_time;
static double Csi_start_time;



double CmiTimer()
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

double CmiWallTimer()
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

double CmiCpuTimer()
{
  return (CsiTimer() - Csi_start_time  + Csi_global_time);
}

#include "ext_func.h"
#include "sim.c"
#include "heap.c"
#include "net.c"
#include "simqmng.c"
#include "simrand.c"
