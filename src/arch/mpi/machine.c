/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include "converse.h"
#include <mpi.h>

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))
#define MAX_QLEN 200

/*
    To reduce the buffer used in broadcast and distribute the load from 
  broadcasting node, define CMK_BROADCAST_SPANNING_TREE enforce the use of 
  spanning tree broadcast algorithm.
    This will use the fourth short in message as an indicator of spanning tree
  root.
*/
#define CMK_BROADCAST_SPANNING_TREE    1

#define BROADCAST_SPANNING_FACTOR      20

#define CMI_BROADCAST_ROOT(msg)          ((CmiUInt2 *)(msg))[3]

#if CMK_BROADCAST_SPANNING_TREE
#  define CMI_SET_BROADCAST_ROOT(msg, root)  CMI_BROADCAST_ROOT(msg) = (root);
#else
#  define CMI_SET_BROADCAST_ROOT(msg, root)
#endif


#define TAG     1375

int Cmi_mype;
int Cmi_numpes;
int Cmi_myrank;
CpvDeclare(void*, CmiLocalQueue);

#define BLK_LEN  512

static int MsgQueueLen=0;
static int request_max;
static void **recdQueue_blk;
static unsigned int recdQueue_blk_len;
static unsigned int recdQueue_first;
static unsigned int recdQueue_len;
static void recdQueueInit(void);
static void recdQueueAddToBack(void *element);
static void *recdQueueRemoveFromFront(void);

typedef struct msg_list {
     MPI_Request req;
     char *msg;
     struct msg_list *next;
} SMSG_LIST;

static int Cmi_dim;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

double starttimer;

void CmiAbort(const char *message);
void CmiMemLock(void) {}
void CmiMemUnlock(void) {}

void SendSpanningChildren(int size, char *msg);
/**************************  TIMER FUNCTIONS **************************/

void CmiTimerInit(void)
{
  starttimer = MPI_Wtime();
}

double CmiTimer(void)
{
  return MPI_Wtime() - starttimer;
}

double CmiWallTimer(void)
{
  return MPI_Wtime() - starttimer;
}

double CmiCpuTimer(void)
{
  return MPI_Wtime() - starttimer;
}

static int CmiAllAsyncMsgsSent(void)
{
   SMSG_LIST *msg_tmp = sent_msgs;
   MPI_Status sts;
   int done;
     
   while(msg_tmp!=0) {
    done = 0;
    if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts)) 
      CmiAbort("CmiAllAsyncMsgsSent: MPI_Test failed!\n");
    if(!done)
      return 0;
    msg_tmp = msg_tmp->next;
    MsgQueueLen--;
   }
   return 1;
}

int CmiAsyncMsgSent(CmiCommHandle c) {
     
  SMSG_LIST *msg_tmp = sent_msgs;
  int done;
  MPI_Status sts;

  while ((msg_tmp) && ((CmiCommHandle)&(msg_tmp->req) != c))
    msg_tmp = msg_tmp->next;
  if(msg_tmp) {
    done = 0;
    if (MPI_SUCCESS != MPI_Test(&(msg_tmp->req), &done, &sts)) 
      CmiAbort("CmiAsyncMsgSent: MPI_Test failed!\n");
    return ((done)?1:0);
  } else {
    return 1;
  }
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
  int done;
  MPI_Status sts;
     
  while(msg_tmp!=0) {
    done =0;
    if(MPI_Test(&(msg_tmp->req), &done, &sts) != MPI_SUCCESS)
      CmiAbort("CmiReleaseSentMessages: MPI_Test failed!\n");
    if(done) {
      MsgQueueLen--;
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

static int PumpMsgs(void)
{
  int nbytes, flg, res;
  char *msg;
  MPI_Status sts;
  int recd=0;

  while(1) {
    flg = 0;
    res = MPI_Iprobe(MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &flg, &sts);
    if(res != MPI_SUCCESS)
      CmiAbort("MPI_Iprobe failed\n");
    if(!flg)
      return recd;
    recd = 1;
    MPI_Get_count(&sts, MPI_BYTE, &nbytes);
    msg = (char *) CmiAlloc(nbytes);
    if (MPI_SUCCESS != MPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,TAG, MPI_COMM_WORLD,&sts)) 
      CmiAbort("PumpMsgs: MPI_Recv failed!\n");
    recdQueueAddToBack(msg);
#if CMK_BROADCAST_SPANNING_TREE
    if (CMI_BROADCAST_ROOT(msg))
      SendSpanningChildren(nbytes, msg);
#endif
  }
}

/********************* MESSAGE RECEIVE FUNCTIONS ******************/

void *CmiGetNonLocal(void)
{
  void *msg = recdQueueRemoveFromFront();
  if(!msg) {
    CmiReleaseSentMessages();
    if (PumpMsgs())
      return recdQueueRemoveFromFront();
    else
      return 0;
  }
  return msg;
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

  CMI_SET_BROADCAST_ROOT(dupmsg, 0);

  if (Cmi_mype==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiAsyncSendFn(destPE, size, dupmsg);
}


CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  SMSG_LIST *msg_tmp;
     
  CQdCreate(CpvAccess(cQdState), 1);
  if(destPE == CmiMyPe()) {
    char *dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
    return 0;
  }
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
	/*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
	CmiReleaseSentMessages();
  }
  if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,destPE,TAG,MPI_COMM_WORLD,&(msg_tmp->req))) 
    CmiAbort("CmiAsyncSendFn: MPI_Isend failed!\n");
  MsgQueueLen++;
  if(sent_msgs==0)
    sent_msgs = msg_tmp;
  else
    end_sent->next = msg_tmp;
  end_sent = msg_tmp;
  return (CmiCommHandle) &(msg_tmp->req);
}

void CmiFreeSendFn(int destPE, int size, char *msg)
{
  CMI_SET_BROADCAST_ROOT(msg, 0);

  if (Cmi_mype==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  } else {
    CmiAsyncSendFn(destPE, size, msg);
  }
}


/*********************** BROADCAST FUNCTIONS **********************/

/* same as CmiSyncSendFn, but don't set broadcast root in msg header */
void CmiSyncSendFn1(int destPE, int size, char *msg)
{
  char *dupmsg = (char *) CmiAlloc(size);
  memcpy(dupmsg, msg, size);
  if (Cmi_mype==destPE) {
    CQdCreate(CpvAccess(cQdState), 1);
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dupmsg);
  }
  else
    CmiAsyncSendFn(destPE, size, dupmsg);
}

/* send msg to its spanning children in broadcast. G. Zheng */
void SendSpanningChildren(int size, char *msg)
{
  int startpe = CMI_BROADCAST_ROOT(msg)-1;
  int i;

  assert(startpe>=0 && startpe<Cmi_numpes);

  for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
    int p = Cmi_mype-startpe;
    if (p<0) p+=Cmi_numpes;
    p = BROADCAST_SPANNING_FACTOR*p + i;
    if (p > Cmi_numpes - 1) break;
    p += startpe;
    p = p%Cmi_numpes;
//CmiPrintf("send: %d %d %d %d\n", startpe, Cmi_mype, p , p%Cmi_numpes);
    assert(p>=0 && p<Cmi_numpes && p!=Cmi_mype);
    CmiSyncSendFn1(p, size, msg);
  }
}

void CmiSyncBroadcastFn(int size, char *msg)     /* ALL_EXCEPT_ME  */
{
#if CMK_BROADCAST_SPANNING_TREE
  CMI_SET_BROADCAST_ROOT(msg, Cmi_mype+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=Cmi_mype+1; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
  for ( i=0; i<Cmi_mype; i++ ) 
    CmiSyncSendFn(i, size,msg) ;
#endif
}


/*  FIXME: luckily async is never used  G. Zheng */
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
#if CMK_BROADCAST_SPANNING_TREE
  CmiSyncSendFn(Cmi_mype, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, Cmi_mype+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif
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
#if CMK_BROADCAST_SPANNING_TREE
  CmiSyncSendFn(Cmi_mype, size,msg) ;
  CMI_SET_BROADCAST_ROOT(msg, Cmi_mype+1);
  SendSpanningChildren(size, msg);
#else
  int i ;
     
  for ( i=0; i<Cmi_numpes; i++ ) 
    CmiSyncSendFn(i,size,msg) ;
#endif
  CmiFree(msg) ;
}

/************************** MAIN ***********************************/
#define MPI_REQUEST_MAX 1024*10 

void ConverseExit(void)
{
  while(!CmiAllAsyncMsgsSent()) {
    PumpMsgs();
    CmiReleaseSentMessages();
  }
  if (MPI_SUCCESS != MPI_Barrier(MPI_COMM_WORLD)) 
    CmiAbort("ConverseExit: MPI_Barrier failed!\n");
  ConverseCommonExit();
  MPI_Finalize();
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
  if (CmiMyPe() == 0){
    CmiPrintf("End of program\n");
  }
#endif
  exit(0);
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n,i ;
#if CMK_USE_HP_MAIN_FIX
#if FOR_CPLUS
  _main(argc,argv);
#endif
#endif
  
  Cmi_myrank = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &Cmi_numpes);
  MPI_Comm_rank(MPI_COMM_WORLD, &Cmi_mype);
  /* find dim = log2(numpes), to pretend we are a hypercube */
  for ( Cmi_dim=0,n=Cmi_numpes; n>1; n/=2 )
    Cmi_dim++ ;
 /* CmiSpanTreeInit();*/
  i=0;
  request_max=MAX_QLEN;
  CmiGetArgInt(argv,"+requestmax",&request_max);
  /*printf("request max=%d\n", request_max);*/
  CmiTimerInit();
  CpvInitialize(void *, CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = CdsFifo_Create();
  recdQueueInit();
  CthInit(argv);
  ConverseCommonInit(argv);
  if (initret==0) {
    fn(CmiGetArgc(argv), argv);
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
  MPI_Abort(MPI_COMM_WORLD, 1);
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
    MPI_Abort(MPI_COMM_WORLD, 1);
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
#if NODE_0_IS_CONVHOST
  inside_comm = 1;
#endif
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
#if NODE_0_IS_CONVHOST
  inside_comm = 0;
#endif
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

