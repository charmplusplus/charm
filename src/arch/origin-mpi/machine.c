#include <stdio.h>
#include <sys/time.h>
#include "converse.h"
#include <mpi.h>

#define FLIPBIT(node,bitnumber) (node ^ (1 << bitnumber))
#define MAX_QLEN 200

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
static double itime;

static SMSG_LIST *sent_msgs=0;
static SMSG_LIST *end_sent=0;

#if NODE_0_IS_CONVHOST
int inside_comm = 0;
#endif

double starttimer;

void CmiAbort(char *message);

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
    MPI_Test(&(msg_tmp->req), &done, &sts);
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
    MPI_Test(&(msg_tmp->req), &done, &sts);
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
      CmiAbort("MPI_Test failed\n");
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
    res = MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flg, &sts);
    if(res != MPI_SUCCESS)
      CmiAbort("MPI_Iprobe failed\n");
    if(!flg)
      return recd;
    recd = 1;
    MPI_Get_count(&sts, MPI_BYTE, &nbytes);
    msg = (char *) CmiAlloc(nbytes);
    MPI_Recv(msg,nbytes,MPI_BYTE,sts.MPI_SOURCE,1,MPI_COMM_WORLD,&sts);
    recdQueueAddToBack(msg);
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
  if (Cmi_mype==destPE)
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),dupmsg);
  else
    CmiAsyncSendFn(destPE, size, dupmsg);
}


CmiCommHandle CmiAsyncSendFn(int destPE, int size, char *msg)
{
  SMSG_LIST *msg_tmp;
  int res;
     
  if(destPE == CmiMyPe()) {
    char *dupmsg = (char *) CmiAlloc(size);
    memcpy(dupmsg, msg, size);
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),dupmsg);
    return;
  }
  msg_tmp = (SMSG_LIST *) CmiAlloc(sizeof(SMSG_LIST));
  msg_tmp->msg = msg;
  msg_tmp->next = 0;
  while (MsgQueueLen > request_max) {
	/*printf("Waiting for %d messages to be sent\n", MsgQueueLen);*/
	CmiReleaseSentMessages();
  }
  res = MPI_Isend((void *)msg,size,MPI_BYTE,destPE,1,MPI_COMM_WORLD,&(msg_tmp->req));
  MsgQueueLen++;
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
#define MPI_REQUEST_MAX=1024*10 

void ConverseExit(void)
{
  ConverseCommonExit();
  MPI_Finalize();
  if (CmiMyPe() == 0) CmiPrintf("End of program\n");
  exit(0);
}

void ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  int n,i ;
  int nbuf[4];
  
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
  while (argv[i] != 0) {
    if (strncmp(argv[i], "+requestmax",11) == 0) {
      if (strlen(argv[i]) > 11)
	sscanf(argv[i], "+p%d", &request_max);
      else
	sscanf(argv[i+1], "%d", &request_max);
    } 
    i++;
  }
  /*printf("request max=%d\n", request_max);*/
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
  inside_comm = 1;
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
  inside_comm = 0;
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

