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
 * Revision 1.3  1998-02-13 23:56:16  pramacha
 * Removed CmiAlloc, CmiFree and CmiSize
 * Added CmiAbort
 *
 * Revision 1.2  1997/10/03 19:52:07  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 1.1  1997/07/30 22:22:21  rbrunner
 * Old t3d port, probably not working.
 *
 * Revision 1.8  1997/04/25 20:48:19  jyelon
 * Corrected CmiNotifyIdle
 *
 * Revision 1.7  1997/04/24 22:37:11  jyelon
 * Added CmiNotifyIdle
 *
 * Revision 1.6  1997/03/19 04:31:55  jyelon
 * Redesigned ConverseInit
 *
 * Revision 1.5  1997/02/13 09:31:57  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 1.4  1996/07/16 21:08:30  gursoy
 * added empty CmiDeliverSpecificMsg
 *
 * Revision 1.3  1996/07/15  20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 1.2  1996/06/28 20:30:51  gursoy
 * *** empty log message ***
 *
 * Revision 1.1  1996/05/16  15:59:43  gursoy
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <time.h>
#include <unistd.h>
#include <math.h>
#include "converse.h"
#include "fm.h"

CpvDeclare(int,  Cmi_mype);
CpvDeclare(int, Cmi_numpes);
CpvDeclare(void*, CmiLocalQueue);
CpvDeclare(int, CmiBufferGrabbed);

static int _MC_neighbour[4]; 
static int _MC_numofneighbour;
static long ticksPerSecond;
static long beginTicks;
static int  remainingMsgCount;
static int  lastMsgType;
static int  singlepktMsgLen;
static void *singlepktMsg;

static void singlepkt_msg_handler();
static void mulpkt_header_handler();
static void mulpkt_data_handler();
static void mulpkt_send();

/* default value used by FM is 1024 bytes */
#define MAX_PACKET_SIZE 1024

#define DATA_SIZE (MAX_PACKET_SIZE - sizeof(int))

#define RESET         0
#define SINGLEPKT_MSG 1
#define MULTIPKT_MSG  2

#define machine_send(dest, size, msg)  \
    { if (size <= MAX_PACKET_SIZE) \
        { FMs_send(dest, singlepkt_msg_handler, msg, size);\
          FMs_complete_send();}\
      else mulpkt_send(dest, size, msg);}\

typedef void (*FM_HANDLER)(void *, int) ;
typedef int (*FUNCTION_PTR)();   

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

double CmiWallTimer()
{
  double t;
  t = (double) (rtclock() - beginTicks); 
  return (double) (t/(double)ticksPerSecond);
}

double CmiCpuTimer()
{
  double t;
  t = (double) (rtclock() - beginTicks); 
  return (double) (t/(double)ticksPerSecond);
}

double CmiTimer()
{
  double t;
  t = (double) (rtclock() - beginTicks); 
  return (double) (t/(double)ticksPerSecond);
}

static void CmiTimerInit()
{
  ticksPerSecond = sysconf(_SC_CLK_TCK) ;
  beginTicks = rtclock() ;
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void CmiDeliversInit()
{
  CpvInitialize(int, CmiBufferGrabbed);
  CpvAccess(CmiBufferGrabbed) = 0;
}



void CmiGrabBuffer(void **msg)
{
  CpvAccess(CmiBufferGrabbed) = 1;

  if (lastMsgType == SINGLEPKT_MSG)
  {
       *msg = (void *) CmiAlloc(singlepktMsgLen);
       memcpy(*msg, singlepktMsg, singlepktMsgLen);
  }
}


int CmiDeliverMsgs(maxmsgs)
int maxmsgs;
{
    int n;
    void *msg;

    remainingMsgCount = maxmsgs;

    while (1)
    {
        n = FMs_extract_1();
        if (remainingMsgCount==0) break;

        FIFO_DeQueue(CpvAccess(CmiLocalQueue), &msg);
        if (msg) 
        {
            CpvAccess(CmiBufferGrabbed)=0;
            (CmiGetHandlerFunction(msg))(msg);
            if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msg);
            remainingMsgCount--; if (remainingMsgCount==0) break;
        }
 
        if(n==0 && msg==0) break;

    }

    return remainingMsgCount;
}




/*
 * CmiDeliverSpecificMsg(lang)
 *
 * - waits till a message with the specified handler is received,
 *   then delivers it.
 *
 */

void CmiDeliverSpecificMsg(handler)
int handler;
{
  /* not implemented yet */
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
    if (CpvAccess(Cmi_mype) == destPE)
       {
          temp = (char *)CmiAlloc(size) ;
          memcpy(temp, msg, size) ;
          FIFO_EnQueue(CpvAccess(CmiLocalQueue), temp);
       }
    else
          machine_send(destPE, size, msg);
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
     int destPE, size;
     char *msg;
{
    if (CpvAccess(Cmi_mype) == destPE)
       {
          FIFO_EnQueue(CpvAccess(CmiLocalQueue), msg);
       }
    else
       {  
          machine_send(destPE, size, msg);
          CmiFree(msg);
       }
}



void CmiSyncBroadcastFn(size, msg)        /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    int i;
    if (CpvAccess(Cmi_numpes) > 1) 
    {
        for(i=0; i<CpvAccess(Cmi_numpes); i++)
          if (i!= CpvAccess(Cmi_mype)) {machine_send(i, size, msg);}
    }
}



CmiCommHandle CmiAsyncBroadcastFn(size, msg) /* ALL_EXCEPT_ME  */
int size;
char * msg;
{
    CmiSyncBroadcastFn(size, msg);
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
    int i;
    char *temp;
    if (CpvAccess(Cmi_numpes) > 1) 
    {
       for(i=0; i<CpvAccess(Cmi_numpes); i++) 
          if (i!= CpvAccess(Cmi_mype)) {machine_send(i, size, msg);}
    } 
    temp = (char *)CmiAlloc(size) ;
    memcpy(temp, msg, size) ;
    FIFO_EnQueue(CpvAccess(CmiLocalQueue), temp); 
}


CmiCommHandle CmiAsyncBroadcastAllFn(size, msg)
int size;
char * msg;
{
    CmiSyncBroadcastAllFn(size, msg);
}



void CmiFreeBroadcastAllFn(size, msg)
int size;
char * msg;
{
    int i;
    if (CpvAccess(Cmi_numpes) > 1)
    {
        for(i=0; i<CpvAccess(Cmi_numpes); i++) 
          if (i!= CpvAccess(Cmi_mype)) {machine_send(i, size, msg);}

    }
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
{
  CpvInitialize(int, Cmi_mype);
  CpvInitialize(int, Cmi_numpes);
  CpvInitialize(void*, CmiLocalQueue);
  CpvAccess(Cmi_mype)  = _my_pe();
  CpvAccess(Cmi_numpes) = _num_pes();
  CpvAccess(CmiLocalQueue)= (void *) FIFO_Create();
  ConverseCommonSetup(argv);
  CthInit(argv);
  neighbour_init(CpvAccess(Cmi_mype));
  CmiSpanTreeInit();
  FM_set_parameter(MAX_MSG_SIZE_FINC, MAX_PACKET_SIZE) ;
  /* 512 is the default value used by FM */
  FM_set_parameter(MSG_BUFFER_SIZE_FINC, 512);
  FM_initialize() ;
  CmiTimerInit();
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
    if (node == CpvAccess(Cmi_mype) )
     return  _MC_numofneighbour;
    else
     return 0;
}


CmiGetNodeNeighbours(node, neighbours)
int node, *neighbours;
{
    int i;

    if (node == CpvAccess(Cmi_mype) )
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
/* Following functions establish a two dimensional torus connection */
/* among the procesors (for any number of processors > 0              */
   


static neighbour_init(p)
int p;
{
    int a,b,n;

    a = (int) floor(sqrt((double) CpvAccess(Cmi_numpes)));
    b = (int) ceil( ((double)CpvAccess(Cmi_numpes) / (double)a) );

   
    _MC_numofneighbour = 0;
   
    /* east neighbour */
    if ( (p+1)%b == 0 )
           n = p-b+1;
    else {
           n = p+1;
           if (n>=CpvAccess(Cmi_numpes)) n = (a-1)*b; /* west-south corner */
    }
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* west neigbour */
    if ( (p%b) == 0) {
          n = p+b-1;
          if (n >= CpvAccess(Cmi_numpes)) n = CpvAccess(Cmi_numpes)-1;
       }
    else
          n = p-1;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;

    /* north neighbour */
    if ( (p/b) == 0) {
          n = (a-1)*b+p;
          if (n >= CpvAccess(Cmi_numpes)) n = n-b;
       }
    else
          n = p-b;
    if (neighbour_check(p,n) ) _MC_neighbour[_MC_numofneighbour++] = n;
    
    /* south neighbour */
    if ( (p/b) == (a-1) )
           n = p%b;
    else {
           n = p+b;
           if (n >= CpvAccess(Cmi_numpes)) n = n%b;
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





/* ******************************************************************* */
/* Following functions performs packetizing and work with FM           */
/* ******************************************************************* */



typedef struct _MsgStruct {
	char *beginPtr ;
	char *currentPtr ;
	int  length;
        int  remainingPckts;
} MsgStruct ;


static MsgStruct MsgTable[512] ;




static void singlepkt_msg_handler(buf,len)
int *buf;
int len;
{
   lastMsgType     = SINGLEPKT_MSG;
   singlepktMsgLen = len;
   singlepktMsg    = buf;
   CpvAccess(CmiBufferGrabbed)=0;
   ( (FUNCTION_PTR) CmiGetHandlerFunction((void *)buf) ) ((void *)buf);
   remainingMsgCount--;
   lastMsgType = RESET;
}




static void mulpkt_header_handler(len,sourcePe,nPackets,firstWord)
int len ; /* in bytes */
int sourcePe;
int nPackets;
int firstWord;
{ 
	/* This is the control information for multi packet message */
        /* it has the following info 
           1-      message length in bytes 
           2-      source  processor number 
           3-      number of packets that will follow this 
           4-      the first word of the first packet in the sequence

           The fourth field is necessary because packey will have the 
           source pe in its first word. Therefore, The data of that word
           will be sent with previous word.
        */


	MsgTable[sourcePe].length         = len ;
        MsgTable[sourcePe].remainingPckts = nPackets;
	MsgTable[sourcePe].beginPtr       = (char *) CmiAlloc(len) ;
	MsgTable[sourcePe].currentPtr     = MsgTable[sourcePe].beginPtr;

        /* save the first of the first data packet */

        *((int *) MsgTable[sourcePe].beginPtr) = firstWord;
        MsgTable[sourcePe].currentPtr +=  sizeof(int);
}







static void mulpkt_data_handler(buf, len)
int *buf ;
int len ; /* in bytes */
{
	int  i,sourcePe;
        
	sourcePe = *buf;

        buf++; /* skip the control word */
        len -= sizeof(int);

        MsgTable[sourcePe].remainingPckts--;

        if (MsgTable[sourcePe].remainingPckts == 0)  /* last packet */
        {
            /* the last data packet is received */
            char *bufPtr = (char *)buf;
            char *msgPtr;     
            /* byte by byte copy */
            /* starting from the last poition we were left */
            msgPtr =  MsgTable[sourcePe].currentPtr;
            for(i=0; i<len; i++) *(msgPtr++) = *(bufPtr++);

            /* message is assembled, point to the beginning */ 
            msgPtr = MsgTable[sourcePe].beginPtr;



            lastMsgType = MULTIPKT_MSG;

            CpvAccess(CmiBufferGrabbed)=0;

            /* call the handler (converse) */
            ((FUNCTION_PTR)CmiGetHandlerFunction(msgPtr))(msgPtr) ;

            lastMsgType =  RESET;

            if (!CpvAccess(CmiBufferGrabbed)) CmiFree(msgPtr);

            remainingMsgCount--;

            MsgTable[sourcePe].length         = 0;
            MsgTable[sourcePe].remainingPckts = 0;
            MsgTable[sourcePe].beginPtr       = 0;
            MsgTable[sourcePe].currentPtr     = 0;
	}

        else
 
        {
           /* a data packet is received (not the last one) */
           /* do a word by word copy for a faster transfer */
           int *bufPtr = buf;
           int *msgPtr =  (int *) MsgTable[sourcePe].currentPtr;
           MsgTable[sourcePe].currentPtr += len;
           /* copy int by int for faster copying */
           len = len >> 3; /* divide by sizeof(int)==8 bytes*/
           for(i=0; i<len; i++) *msgPtr++ = *buf++;
        }
}








static void mulpkt_send(dest, size, msg)
int  dest, size;
int *msg;
{
/* msg size (in bytes) is bigger than MAX_PACKET_SIZE, so packetize */

     int i, numPackets,lastsize;
     char *nextPacketPtr;

     /* data packets are of length (DATA_SIZE + sizeof(int) (which is equal */
     /* to  MAX_PACKET_SIZE). The first integer is the control info and */
     /* the rest is the real data  */

     numPackets = size / DATA_SIZE;

     lastsize = size % DATA_SIZE;
     if ( lastsize ) 
        numPackets++; /* add the last short packet */
     else
        lastsize = DATA_SIZE; /* there is no short packet at the end */


     /* send header packet */
     FMs_send_4(dest,mulpkt_header_handler,size,CmiMyPe(),numPackets,*msg);
     FMs_complete_send();
     /* send the data packets */
     /* numPackets-1  packets are of size  MAX_PACKET_SIZE, for sure */

     *msg = CmiMyPe(); /* first word was sent already */
     nextPacketPtr = (char *) msg; 
     for(i=0; i<numPackets-1; i++)
     {
         FMs_send(dest, mulpkt_data_handler, nextPacketPtr, MAX_PACKET_SIZE);
         FMs_complete_send();
         nextPacketPtr += DATA_SIZE;
         *((int *)nextPacketPtr) = CmiMyPe();
     }

     /* last packet could be shorter */
     FMs_send(dest, mulpkt_data_handler, nextPacketPtr, lastsize);
     FMs_complete_send();
}
