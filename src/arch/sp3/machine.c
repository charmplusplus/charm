/***************************************************************************
 * RCS INFORMATION:
 *
 *  $RCSfile$
 *  $Author$  $Locker$    $State$
 *  $Revision$  $Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 1.8  1998-03-05 17:15:06  milind
 * Fixed conflicts.
 *
 * Revision 1.7  1998/02/13 23:56:06  pramacha
 * Removed CmiAlloc, CmiFree and CmiSize
 * Added CmiAbort
 *
 * Revision 1.6  1998/01/16 18:29:05  milind
 * Used high resolution timers in SP3 version.
 *
 * Revision 1.5  1998/01/15 22:25:52  milind
 * Fixed bugs in latencyBWtest and optimized SP3 communication.
 *
 * Revision 1.4  1997/12/10 21:01:41  jyelon
 * *** empty log message ***
 *
 * Revision 1.3  1997/10/03 19:52:03  milind
 * Made charmc to work again, after inserting trace calls in converse part,
 * i.e. threads and user events.
 *
 * Revision 1.2  1997/07/31 15:30:58  milind
 * ANSIfied.
 *
 * Revision 1.1  1997/07/08 22:10:56  milind
 * Added IBM SP3 version. Developed and Tested on ANL machine.
 *
 * Revision 2.15  1997/04/25 20:48:17  jyelon
 * Corrected CmiNotifyIdle
 *
 * Revision 2.14  1997/04/24 22:37:09  jyelon
 * Added CmiNotifyIdle
 *
 * Revision 2.13  1997/03/21 20:06:55  milind
 * Fixed a prototype mismatch.
 *
 * Revision 2.12  1997/03/19 04:31:52  jyelon
 * Redesigned ConverseInit
 *
 * Revision 2.11  1997/02/13 09:31:56  jyelon
 * Updated for new main/ConverseInit structure.
 *
 * Revision 2.10  1997/02/02 07:33:56  milind
 * Fixed Bugs in SP1 machine dependent code that made megacon to hang.
 * Consisted of almost 60 percent rewrite.
 *
 * Revision 2.9  1996/07/15 20:59:22  jyelon
 * Moved much timer, signal, etc code into common.
 *
 * Revision 2.8  1996/01/29 16:34:02  milind
 * Corrected a minor bug in CmiReleaseSetntMessages
 *
 * Revision 2.7  1995/11/09  18:23:11  milind
 * Fixed the CmiFreeSendFn bug for messages to self.
 *
 * Revision 2.6  1995/10/27  21:45:35  jyelon
 * Changed CmiNumPe --> CmiNumPes
 *
 * Revision 2.5  1995/10/10  06:10:58  jyelon
 * removed program_name
 *
 * Revision 2.4  1995/09/29  09:50:07  jyelon
 * CmiGet-->CmiDeliver, added protos, etc.
 *
 * Revision 2.3  1995/09/20  16:02:35  gursoy
 * made the arg of CmiFree and CmiSize void*
 *
 * Revision 2.2  1995/09/08  02:38:26  gursoy
 * Cmi_mype Cmi_numpes CmiLocalQueue accessed thru macros now
 *
 * Revision 2.1  1995/07/17  17:46:05  knauff
 * Fixed problem with machine.c
 *
 * Revision 2.0  1995/07/10  22:12:39  knauff
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <sys/time.h>
#include "converse.h"
#include <mpproto.h>
#include <sys/systemcfg.h>

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

int Cmi_mype;
int Cmi_numpes;
int Cmi_myrank;
CpvDeclare(void*, CmiLocalQueue);

#define BLK_LEN  512

static void **recdQueue_blk;
static unsigned int recdQueue_blk_len;
static unsigned int recdQueue_first;
static unsigned int recdQueue_len;
static void recdQueueInit(void);
static void recdQueueAddToBack(void *element);
static void *recdQueueRemoveFromFront(void);

typedef struct msg_list {
     int msgid;
     char *msg;
     struct msg_list *next;
} SMSG_LIST;

static int Cmi_dim;
static double itime;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

static int allmsg, dontcare, msgtype;

/**************************  TIMER FUNCTIONS **************************/

static void CmiTimerInit(void)
{
  timebasestruct_t time;
  read_real_time(&time, TIMEBASE_SZ);
  time_base_to_time(&time, TIMEBASE_SZ);
  itime=(double)time.tb_high + 1.0e-9*((double) time.tb_low);
}

double CmiTimer(void)
{
  double t;
  timebasestruct_t time;
  
  read_real_time(&time, TIMEBASE_SZ);
  time_base_to_time(&time, TIMEBASE_SZ);
  t=(double)time.tb_high + 1.0e-9*((double) time.tb_low);
  return (t-itime);
}

double CmiWallTimer(void)
{
  double t;
  timebasestruct_t time;
  
  read_real_time(&time, TIMEBASE_SZ);
  time_base_to_time(&time, TIMEBASE_SZ);
  t=(double)time.tb_high + 1.0e-9*((double) time.tb_low);
  return (t-itime);
}

double CmiCpuTimer(void)
{
  return CmiTimer();
}

static int CmiAllAsyncMsgsSent(void)
{
   SMSG_LIST *msg_tmp = sent_msgs;
     
   while(msg_tmp!=0) {
    if(mpc_status(msg_tmp->msgid)<0)
      return 0;
    msg_tmp = msg_tmp->next;
   }
   return 1;
}

int CmiAsyncMsgSent(CmiCommHandle c) {
     
  SMSG_LIST *msg_tmp = sent_msgs;

  while ((msg_tmp) && ((CmiCommHandle)msg_tmp->msgid != c))
    msg_tmp = msg_tmp->next;
     
  if ((msg_tmp) && (mpc_status(msg_tmp->msgid)<0))
    return 0;
  else
    return 1;
}

void CmiReleaseCommHandle(CmiCommHandle c)
{
  return;
}


static void CmiReleaseSentMessages(void)
{
  SMSG_LIST *msg_tmp=sent_msgs;
  SMSG_LIST *prev=0;
  SMSG_LIST *temp;
     
  while(msg_tmp!=0) {
    if(mpc_status(msg_tmp->msgid)>=0) {
      /* Release the message */
      temp = msg_tmp->next;
      if(prev==0)  /* first message */
        sent_msgs = temp;
      else
        prev->next = temp;
      CmiFree(msg_tmp->msg);
      CmiFree(msg_tmp);
      msg_tmp = temp;
    } else {
      prev = msg_tmp;
      msg_tmp = msg_tmp->next;
    }
  }
  end_sent = prev;
}

static void PumpMsgs(void)
{
  int src, type, mstat;
  size_t nbytes;
  char *msg;

  while(1) {
    src = dontcare; type = msgtype;
    mpc_probe(&src, &type, &mstat);
    if(mstat<0)
      return;
    msg = (char *) CmiAlloc((size_t) mstat);
    mpc_brecv(msg, (size_t)mstat, &src, &type, &nbytes);
    recdQueueAddToBack(msg);
  }
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal(void)
{
  return recdQueueRemoveFromFront();
}

void CmiNotifyIdle(void)
{
  CmiReleaseSentMessages();
  PumpMsgs();
}
 
/********************* MESSAGE SEND FUNCTIONS ******************/

void CmiSyncSendFn(int destPE, int size, char *msg)
{
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);
  if (Cmi_mype==destPE)
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),dupmsg);
  else
    CmiAsyncSendFn(destPE, size, dupmsg);
}


CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  SMSG_LIST *msg_tmp;
  int msgid;
     
  mpc_send(msg, size, destPE, msgtype, &msgid);
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msgid = msgid;
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  if (Cmi_mype==destPE) {
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  } else {
    CmiAsyncSendFn(destPE, size, msg);
  }
}


/*********************** BROADCAST FUNCTIONS **********************/

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  int i ;
     
  for ( i=Cmi_mype+1; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
  for ( i=0; i<Cmi_mype; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
}


CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)  
{
  int i ;

  for ( i=Cmi_mype+1; i<Cmi_numpes; i++ ) 
    CmiAsyncSendFn(i,size,msg) ;
  for ( i=0; i<Cmi_mype; i++ ) 
    CmiAsyncSendFn(i,size,msg) ;
  return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastFn(int size, char *msg)
{
   CmiSyncBroadcastFn(size,msg);
   CmiFree(msg);
}
 
void CmiSyncBroadcastAllFn(int size, char *msg)        /* All including me */
{
  int i ;
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)  
{
  int i ;

  for ( i=1; i<Cmi_numpes; i++ ) 
    CmiAsyncSendFn(i,size,msg) ;
  return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastAllFn(int size, char *msg)  /* All including me */
{
  int i ;
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
  CmiFree(msg) ;
}

/* Neighbour functions used mainly in LDB : pretend the SP1 is a hypercube */

int CmiNumNeighbours(int node)
{
  return Cmi_dim;
}

void CmiGetNodeNeighbours(int node, int *neighbours)
{
  int i;
     
  for (i = 0; i < Cmi_dim; i++)
    neighbours[i] = FLIPBIT(node,i);
}

int CmiNeighboursIndex(int node, int neighbour)
{
  int index = 0;
  int linenum = node ^ neighbour;

  while (linenum > 1) {
    linenum = linenum >> 1;
    index++;
  }
  return index;
}

/************************** MAIN ***********************************/

void ConverseExit(void)
{
  int msgid = allmsg; 
  size_t nbytes;
  ConverseCommonExit();
  mpc_wait(&msgid, &nbytes);
  exit(0);
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n ;
  int nbuf[4];
  
  Cmi_myrank = 0;
  mpc_environ(&Cmi_numpes, &Cmi_mype);
  mpc_task_query(nbuf, 4, 3);
  dontcare = nbuf[0];
  allmsg = nbuf[1];
  mpc_task_query(nbuf, 2, 2);
  msgtype = nbuf[0];
  
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;
  CmiSpanTreeInit();
  CmiTimerInit();
  CpvInitialize(void *, CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = (void *)FIFO_Create();
  recdQueueInit();
  CthInit(argv);
  ConverseCommonInit(argv);
  if (initret==0) {
    fn(argc, argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

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
 
/* ****************************************************************** */
/*    The following internal functions implement recd msg queue       */
/* ****************************************************************** */

static void ** AllocBlock(unsigned int len)
{
  void ** blk;

  blk=(void **)CmiAlloc(len*sizeof(void *));
  if(blk==(void **)0) {
    CmiError("Cannot Allocate Memory!\n");
    mpc_stopall(1);
  }
  return blk;
}

static void 
SpillBlock(void **srcblk, void **destblk, unsigned int first, unsigned int len)
{
  memcpy(destblk, &srcblk[first], (len-first)*sizeof(void *));
  memcpy(&destblk[len-first],srcblk,first*sizeof(void *));
}

void recdQueueInit(void)
{
  recdQueue_blk = AllocBlock(BLK_LEN);
  recdQueue_blk_len = BLK_LEN;
  recdQueue_first = 0;
  recdQueue_len = 0;
}

void recdQueueAddToBack(void *element)
{
  if(recdQueue_len==recdQueue_blk_len) {
    void **blk;
    recdQueue_blk_len *= 3;
    blk = AllocBlock(recdQueue_blk_len);
    SpillBlock(recdQueue_blk, blk, recdQueue_first, recdQueue_len);
    CmiFree(recdQueue_blk);
    recdQueue_blk = blk;
    recdQueue_first = 0;
  }
  recdQueue_blk[(recdQueue_first+recdQueue_len++)%recdQueue_blk_len] = element;
}


void * recdQueueRemoveFromFront(void)
{
  if(recdQueue_len) {
    void *element;
    element = recdQueue_blk[recdQueue_first++];
    recdQueue_first %= recdQueue_blk_len;
    recdQueue_len--;
    return element;
  }
  return 0;
}

