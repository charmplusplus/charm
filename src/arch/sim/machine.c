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
 * Revision 1.4  1995-10-27 21:45:35  jyelon
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

#ifdef CMK_TIMER_USE_TIMES
#include <sys/times.h>
#include <sys/unistd.h>
#endif
#ifdef CMK_TIMER_USE_GETRUSAGE
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

static void mycpy();
double CmiTimer();
static void CsiTimerInit();


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
    mycpy((double *)buf,(double *)msg,size);

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
    CmiSyncSendFn(destPE, size, msg);
    CmiFree(msg);
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
     mycpy((double *)buf,(double *)msg,size);
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

static int _MC_neighbour[4]; 
static int _MC_numofneighbour;
static neighbour_check();
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

    a = (int) floor(sqrt((double)CmiNumPes()));
    b = (int) ceil( ((double)CmiNumPes() / (double)a) );

   
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


/************************** SETUP ***********************************/

void CmiInitMc(argv)
char *argv[];
{
    neighbour_init(Cmi_mype);
    CpvAccess(CmiLocalQueue) = (void *) FIFO_Create();
    CmiSpanTreeInit();
    CsiTimerInit();
}



void CmiExit()
{}


void CmiDeclareArgs()
{}


main(argc,argv)
int argc;
char *argv[];
{
    int i, requested_npe;

    /* figure out number of processors required */
 
    i =  0;
    requested_npe = 1;
    while (argv[i] != NULL)
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

    Cmi_numpes = requested_npe;
    Cmi_mype  = 0;

    CpvInitialize(void*, CmiLocalQueue);
   
    McQueue = (void **) malloc(requested_npe * sizeof(void *)); 
    for(i=0; i<requested_npe; i++) McQueue[i] = (void *) FIFO_Create();
    sim_initialize("sim.param",requested_npe);
    user_main(argc,argv);
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



#define checkStopFlag() (CsvAccess(CsdStopCount) == 0)

void CsdExitScheduler()
{
    CpvAccess(CsdStopFlag) = 1;
    CsvAccess(CsdStopCount)--;
}






/* ********************************************************************* */
/*                      SIMULATOR                                        */
/* ********************************************************************* */



#ifdef CMK_TIMER_USE_TIMES

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

#ifdef CMK_TIMER_USE_GETRUSAGE

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




void CsdUniScheduler(count)
int count;
{
    void simulate();

    while (count--) {
       if (checkStopFlag()) return;
       simulate();
    }
}





#include "ext_func.h"
#include "sim.c"
#include "heap.c"
#include "net.c"
#include "simqmng.c"
#include "simrand.c"
