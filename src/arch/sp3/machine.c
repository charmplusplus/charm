/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** @file
 * SP3 machine layer
 * @ingroup Machine
 * @{
 */

#include <stdio.h>
#include <sys/time.h>
#include "converse.h"
#include <mpproto.h>
#include <sys/systemcfg.h>

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))

int _Cmi_mype;
int _Cmi_numpes;
int _Cmi_myrank;
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

void CmiTimerInit(char **argv)
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
  CmiReleaseSentMessages();
  PumpMsgs();
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
  if (_Cmi_mype==destPE) {
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
    CQdCreate(CpvAccess(cQdState), 1);
  }
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
  CQdCreate(CpvAccess(cQdState), 1);
  return (CmiCommHandle) msgid;
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  if (_Cmi_mype==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  } else {
    CmiAsyncSendFn(destPE, size, msg);
  }
}


/*********************** BROADCAST FUNCTIONS **********************/

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
  int i ;
     
  for ( i=_Cmi_mype+1; i<_Cmi_numpes; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
  for ( i=0; i<_Cmi_mype; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
}


CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)  
{
  int i ;

  for ( i=_Cmi_mype+1; i<_Cmi_numpes; i++ ) 
    CmiAsyncSendFn(i,size,msg) ;
  for ( i=0; i<_Cmi_mype; i++ ) 
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
     
  for ( i=0; i<_Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)  
{
  int i ;

  for ( i=1; i<_Cmi_numpes; i++ ) 
    CmiAsyncSendFn(i,size,msg) ;
  return (CmiCommHandle) (CmiAllAsyncMsgsSent());
}

void CmiFreeBroadcastAllFn(int size, char *msg)  /* All including me */
{
  int i ;
     
  for ( i=0; i<_Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
  CmiFree(msg) ;
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
  
  _Cmi_myrank = 0;
  mpc_environ(&_Cmi_numpes, &_Cmi_mype);
  mpc_task_query(nbuf, 4, 3);
  dontcare = nbuf[0];
  allmsg = nbuf[1];
  mpc_task_query(nbuf, 2, 2);
  msgtype = nbuf[0];
  
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=_Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;
  /* CmiSpanTreeInit(); */
  CmiTimerInit(argv);
  CpvInitialize(void *, CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = CdsFifo_Create();
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

void CmiAbort(const char *message)
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

/*@}*/
