/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*
 * This is a complete port, but could be made considerably more efficient
 * by handling asynchronous messages correctly, ie. without doing
 * an extra copy and synchronous send
 */

#include <stdlib.h>
#include <unistd.h>
#include <malloc.h>
#include "converse.h"
#include CMK_SHMEM_H

#if !CMK_SHMEM_T3E
#define globalexit exit
#endif 

/*
 *  We require statically allocated variables for locks.  This defines
 *  the max number of processors available.
 */
#define MAX_PES 2048

/*
 * Some constants
 */
enum boolean {false = 0, true = 1};
enum {list_empty = -1 };


/*
 * Local declarations for Cmi, used by common code
 */
CpvDeclare(void*, CmiLocalQueue);
int Cmi_mype;
int Cmi_numpes;
int Cmi_myrank;

/*
 * Local queue functions, used by common code to store messages 
 * to my own node efficiently.  These are used when 
 * CMK_CMIDELIVERS_USE_COMMON_CODE is true.
 */
/*
 * Distributed list declarations.  This linked list goes across machines,
 * storing all the messages for this node until this processor copies them
 * into local memory.
 */
typedef struct McDistListS
{
  int nxt_node;
  struct McMsgHdrS *nxt_addr;
  int msg_sz;
} McDistList;

typedef struct McMsgHdrS
{
  McDistList list_node;
  enum {Unknown, Message, BcastMessage } msg_type;
  enum boolean received_f;
  union
  {
    struct McMsgHdrS *ptr;
    int count;
  } bcast;
  int bcast_msg_size;
  int handler;
} McMsgHdr;


/*
 * Mc functions, used in machine.c only.
 */
static void McInit();
static void McInitList();
static void McEnqueueRemote(void *msg, int msg_sz, int dst_pe);
static void McRetrieveRemote(void);
static void McCleanUpInTransit(void);

/*
 * These declarations are for a local linked list to hold messages which
 * have been copied into local memory.  It is a modified version of the
 * Origin2000 code with the locks removed.
 */
/* Allocation block size, to reduce num of mallocs */
#define BLK_LEN  512  

typedef struct McQueueS
{
  void     **blk;
  unsigned int blk_len;
  unsigned int first;
  unsigned int len;
} McQueue;

static McQueue *McQueueCreate(void);
static void McQueueAddToBack(McQueue *queue, void *element);
static void *McQueueRemoveFromFront(McQueue *queue);
static void *McQueueRemoveFromBack(McQueue *queue);

/*************************************************************
 * static variable declarations
 */
/*
 *  Local queues used for mem management.
 *
 * These queues hold outgoing messages which will be picked up by
 * receiver PEs.  Garbage collection works by scanning the 
 * in_transit_queue for messages, freeing delivered ones, and moving
 * others to in_transit_tmp_queue.  Then the pointers are swapped,
 * so in_transit_queue contains all the undelivered messages, and
 * in_transit_tmp_queue is empty.
 */
static McQueue *in_transit_queue;
static McQueue *in_transit_tmp_queue;

/* tmp_queue is used to invert the order of incoming messages */
static McQueue *tmp_queue;  

/* received_queue holds all the messages which have been moved
 * into local memory.  Messages are dequede from here.
 */
static McQueue *received_queue;

/* received_token_queue saves incoming broadcast-message tokens,
 * until McRetrieveRemote is done with them.
 */
static McQueue *received_token_queue;

/* outgoing broadcast message queue, holds messages until all receivers have
 * picked it up
 */
static McQueue *broadcast_queue;
static McQueue *broadcast_tmp_queue;

/*
 * head is the pointer to my next incoming message.
 */
static McDistList head;

/* Static variables are necessary for locks. */
static long *my_lock;
static long head_lock[MAX_PES];
static long bcast_lock[MAX_PES];

#define ALIGN8(x)  (8*(((x)+7)/8))

int McChecksum(char *msg, int size)
{
  int chksm;
  int i;

  chksm=0xff;
  for(i=0; i < size; i++)
    chksm ^= *(msg+i);
  return chksm;
}

/**********************************************************************
 *  CMI Functions START HERE
 */


/**********************************************************************
 * Cmi Message calls.  This implementation uses sync-type sends for
 * everything.  An async interface would be efficient, and not difficult
 * to add
 */
void CmiSyncSendFn(int dest_pe, int size, char *msg)
{
  McMsgHdr *dup_msg;

  dup_msg = (McMsgHdr *)CmiAlloc(ALIGN8(size));
  memcpy(dup_msg,msg,size);
  dup_msg->msg_type = Message;

  McRetrieveRemote();

  if (dest_pe == Cmi_mype)
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),dup_msg);
  else
  {
    McEnqueueRemote(dup_msg,ALIGN8(size),dest_pe); 
  }
  CQdCreate(CpvAccess(cQdState), 1);
}

CmiCommHandle CmiAsyncSendFn(int dest_pe, int size, char *msg)
{
  CmiSyncSendFn(dest_pe, size, msg);
  return 0;
}

void CmiFreeSendFn(int dest_pe, int size, char *msg)
{
  /* No need to copy message, since we will immediately free it */
  McRetrieveRemote();
  ((McMsgHdr *)msg)->msg_type = Message;

  if (dest_pe == Cmi_mype)
    CdsFifo_Enqueue(CpvAccess(CmiLocalQueue),msg);
  else
  {
    McEnqueueRemote(msg,size,dest_pe); 
  }
  CQdCreate(CpvAccess(cQdState), 1);
}

void CmiSyncBroadcastFn(int size, char *msg)
{
  int i;
  McMsgHdr *dup_msg;
  McMsgHdr bcast_msg_tok;
  McMsgHdr *dup_tok;
  int hdr_size;

  /*
   * Copy user's message, and set count to the correct number of recients
   */
  dup_msg = (McMsgHdr *)CmiAlloc(ALIGN8(size));
  memcpy(dup_msg,msg,size);
  dup_msg->bcast.count = Cmi_numpes - 1;
  /*
  CmiPrintf("PE %d broadcast handler=%d\n",Cmi_mype,dup_msg->handler);
  */
  /*
   * Make the broadcast token point to the copied message
   */
  bcast_msg_tok.msg_type = BcastMessage;
  bcast_msg_tok.bcast.ptr = dup_msg;
  bcast_msg_tok.bcast_msg_size = size;

  hdr_size = ALIGN8(sizeof(McMsgHdr));

  /*
   * Enqueue copies of the token message on other nodes.  This code should
   * be similar to CmiSyncSend
   */
  for(i=0; i<Cmi_numpes; i++)
    if (i != Cmi_mype)
    {
      dup_tok = (McMsgHdr *)CmiAlloc(ALIGN8(hdr_size));
      memcpy(dup_tok,&bcast_msg_tok,hdr_size);
      McEnqueueRemote(dup_tok,hdr_size,i); 
    }
  /*
   * The token message will be deleted as a normal message,
   * but the message being broadcast needs to be saved for future
   * garbage collection.
   */
  McQueueAddToBack(broadcast_queue,dup_msg);
  CQdCreate(CpvAccess(cQdState), Cmi_numpes-1);
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size,msg);
  return 0;
}

void CmiFreeBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size,msg);
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg)
{
  int i;
  CmiSyncBroadcastFn(size,msg);
  CmiSyncSendFn(Cmi_mype, size, msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)
{
  CmiSyncBroadcastAllFn(size,msg);
  return 0;
}

void CmiFreeBroadcastAllFn(int size, char *msg)
{
  CmiSyncBroadcastAllFn(size,msg);
  CmiFree(msg);
}


void CmiSyncListSendFn(int npes, int *pes, int size, char *msg)
{
  int i;
  McMsgHdr *dup_msg;
  McMsgHdr bcast_msg_tok;
  McMsgHdr *dup_tok;
  int hdr_size;
  int n_remote_pes;

  /*
   * Count how many remote PEs, and send to the local PE if it is in the list
   */
  /*
  CmiPrintf("CmiSyncListSendFn: size %d handler %d\n",
	    size,((McMsgHdr *)msg)->handler);
  CmiPrintf("CmiSyncListSendFn: size %d npes %d\n",size,npes);
  */
  n_remote_pes = 0;
  for (i=0; i < npes; i++)
  {
    if (pes[i] == Cmi_mype)
      CmiSyncSendFn(Cmi_mype, size, msg);
    else
      n_remote_pes++;
  }
  if (n_remote_pes == 0)  // Nothing to do
    return;
  
  /*
   * Copy user's message, and set count to the correct number of recients
   */
  dup_msg = (McMsgHdr *)CmiAlloc(ALIGN8(size));
  memcpy(dup_msg,msg,size);
  dup_msg->bcast.count = n_remote_pes;
  /*
   * Make the broadcast token point to the copied message
   */
  bcast_msg_tok.msg_type = BcastMessage;
  bcast_msg_tok.bcast.ptr = dup_msg;
  bcast_msg_tok.bcast_msg_size = size;

  hdr_size = ALIGN8(sizeof(McMsgHdr));

  /*
   * Enqueue copies of the token message on other nodes.  This code should
   * be similar to CmiSyncSend
   */
  for(i=0; i<npes; i++)
    if (pes[i] != Cmi_mype)
    {
      dup_tok = (McMsgHdr *)CmiAlloc(ALIGN8(hdr_size));
      memcpy(dup_tok,&bcast_msg_tok,hdr_size);
      McEnqueueRemote(dup_tok,hdr_size,pes[i]); 
      CQdCreate(CpvAccess(cQdState), 1);
    }
  /*
   * The token message will be deleted as a normal message,
   * but the message being broadcast needs to be saved for future
   * garbage collection.
   */
  McQueueAddToBack(broadcast_queue,dup_msg);
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int size, char *msg)
{
  CmiSyncListSendFn(npes, pes, size, msg);
  return 0;
}

void CmiFreeListSendFn(int npes, int *pes, int size, char *msg)
{
  CmiSyncListSendFn(npes,pes,size,msg);
  CmiFree(msg);
}

typedef struct {
  char header[CmiMsgHeaderSizeBytes];
  CmiGroup grp;
  int size;
  char *user_msg;
} McMultiMsg;

CpvDeclare(int,McMulticastWaitHandler);

void CmiSyncMulticastFn(CmiGroup grp, int size, char *msg)
{
  int npes;
  int *pes;
  McMultiMsg multi_msg;
  
  /*
  CmiPrintf("CmiSyncMulticastFn: size %d handler %d\n",
	    size,((McMsgHdr *)msg)->handler);
   */
  /*
   *  Check for group, and busy-wait, if necessary, for group info
   */
  CmiLookupGroup(grp, &npes, &pes);
  if (pes != 0)
    CmiSyncListSendFn( npes, pes, size, msg);
  else
  {
    multi_msg.grp = grp;
    multi_msg.size = size;
    multi_msg.user_msg = (char *) CmiAlloc(ALIGN8(size));
    memcpy(multi_msg.user_msg,msg,size);

    CmiSetHandler(&multi_msg,CpvAccess(McMulticastWaitHandler));
    CmiSyncSendFn(CmiMyPe(),sizeof(McMultiMsg),(char *)&multi_msg);
  }
}

void McMulticastWaitFn(McMultiMsg *msg)
{
  CmiFreeMulticastFn(msg->grp,msg->size,msg->user_msg);
}

void CmiMulticastInit(void)
{
  CpvInitialize(int,McMulticastWaitHandler);
  CpvAccess(McMulticastWaitHandler) = CmiRegisterHandler(McMulticastWaitFn);
}

CmiCommHandle CmiAsyncMulticastFn(CmiGroup grp, int size, char *msg)
{
  CmiSyncMulticastFn(grp, size, msg);
  return 0;
}

void CmiFreeMulticastFn(CmiGroup grp, int size, char *msg)
{
  CmiSyncMulticastFn(grp, size, msg);
  CmiFree(msg);
}


/***********************************************************************
 *
 * Abort function:
 *
 ************************************************************************/

void CmiAbort(const char *message)
{
  CmiError(message);
  globalexit(1);
}

/**********************************************************************
 * CMI utility functions for startup, shutdown, and other miscellaneous
 * activities.
 */

/*
 * This port uses the common CmiDeliver code, so we only provide
 * CmiGetNonLocal()
 */
void *CmiGetNonLocal()
{
  McRetrieveRemote();

  return (void *)McQueueRemoveFromFront(received_queue);
}

void 
ConverseInit(int argc, char **argv, CmiStartFn fn, int usched, int initret)
{
  McInit();
  CthInit(argv);
  ConverseCommonInit(argv);
  for(argc=0;argv[argc];argc++);
  if (initret==0)
  {
    fn(argc,argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

void ConverseExit()
{
#if (CMK_DEBUG_MODE || CMK_WEB_MODE || NODE_0_IS_CONVHOST)
  if (CmiMyPe() == 0){
    CmiPrintf("End of program\n");
  }
#endif
  ConverseCommonExit();
  exit(0);
}

void CmiNotifyIdle(void)
{
  /* Use this opportunity to clean up the in_transit_queue */
  McCleanUpInTransit();
}

/* lock
*/

#if CMK_SHMEM_LOCK

void set_lock(long *lock, int pe)
{
  while (shmem_long_swap(lock, 1L, pe)) {
    //shmem_long_wait(&tmp, 1);
  }
//printf("set lock end: %ld. \n", *lock);
}

void clear_lock(long *lock, int pe)
{
//printf("[%d] clear lock on %d %ld. \n", Cmi_mype, pe, *lock);
  shmem_long_swap(lock, 0L, pe);
//  shmem_long_p(lock, 0L, pe);
//printf("clear lock end lock:%ld\n", *lock);
}

#else
#define set_lock(lock, pe)  shmem_set_lock(lock)
#define clear_lock(lock, pe)  shmem_clear_lock(lock)
#endif

/**********************************************************************
 * Mc Functions:
 * Mc functions are used internally in machine.c only
 */

static void McInit(void)
{
  CMK_SHMEM_INIT;

  CpvInitialize(void *, CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = CdsFifo_Create();
  Cmi_mype = _my_pe();
  Cmi_numpes = _num_pes();
  Cmi_myrank = 0;

  McInitList();
}

static void McInitList(void)
{
  int i;

  received_queue = McQueueCreate();
  tmp_queue = McQueueCreate();
  received_token_queue = McQueueCreate();
  broadcast_queue = McQueueCreate();
  broadcast_tmp_queue = McQueueCreate();
  in_transit_tmp_queue = McQueueCreate();
  in_transit_queue = McQueueCreate();

  head.nxt_node = list_empty;
  head.nxt_addr = NULL;
  head.msg_sz = 0;
  if (Cmi_numpes > MAX_PES)
  {
    CmiPrintf("Not enough processors allocated in machine.c.\n");
    CmiPrintf("Change MAX_PES in t3e/machine.c to at least %d and recompile Converse\n",
    Cmi_numpes);
  }
  for(i=0; i < Cmi_numpes; i++)
  {
    head_lock[i] = 0;
    bcast_lock[i] = 0;
  }
  my_lock = &(head_lock[Cmi_mype]);
  shmem_barrier_all();
/*
  clear_lock(my_lock, Cmi_mype);
  clear_lock(&bcast_lock[Cmi_mype], Cmi_mype);
*/
}

static void McEnqueueRemote(void *msg, int msg_sz, int dst_pe)
{
 /*
  * To enqueue on a remote node, we should:
  * 0. Free any delivered messages from the message_in_transit list.
  * 1. Add message in the "message_in_transit" list
  * 2. Fill in the fields in the message header
  * 3. Lock the head pointer on the remote node.
  * 4. Swap the list pointer with that on the other node.
  * 5. Release lock
  */

  McDistList tmp_link;
  McDistList *msg_link;

  /*  CmiPrintf("PE %d outgoing msg = %d msg_type = %d size = %d dst_pe=%d\n",
	    Cmi_mype,msg,((McMsgHdr *)msg)->msg_type,msg_sz, dst_pe); */
  /* 0. Free any delivered messages from the in_transit_queue list. */
  McCleanUpInTransit();

  /* 1. Add message in the "in_transit_queue" list */
  McQueueAddToBack(in_transit_queue,msg);

  /* 2. Fill in the fields in the message header */
  msg_link = &(((McMsgHdr *)msg)->list_node);
  ((McMsgHdr *)msg)->received_f = false;

  /* Set list fields to point back to this processor, this message.  */
  tmp_link.nxt_node = Cmi_mype;
  tmp_link.nxt_addr = msg;
  tmp_link.msg_sz = msg_sz;

  /* 3. Lock the head pointer on the remote node.
     Acquire lock on the destination queue.  If locks turn oout to
     be inefficient, use fetch and increment to imp. lock
   */

  set_lock(&head_lock[dst_pe], dst_pe);


  /* 4. Swap the list pointer with that on the other node.
   */
  /* First, get current head pointer, and stick it in this 
   * message data area.
   */
  shmem_int_get((int*)msg_link, (int*)&head, sizeof(McDistList)/sizeof(int), dst_pe);
  /* Next, write the new message into the top of the list */
  shmem_int_put((int*)&head, (int*)&tmp_link, sizeof(McDistList)/sizeof(int),dst_pe);
  shmem_quiet();

#ifdef DEBUG
  printf("[%d] Adding Message to pe %d\n",Cmi_mype,dst_pe);
  printf("[%d]   nxt_node = %d\n",Cmi_mype,tmp_link.nxt_node);
  printf("[%d]   nxt_addr = %x\n",Cmi_mype,tmp_link.nxt_addr);
  printf("[%d]   msg_sz = %x\n",Cmi_mype,tmp_link.msg_sz);
  printf("[%d] Old Message is now at %x\n",Cmi_mype,msg_link);
  printf("[%d]   nxt_node = %d\n",Cmi_mype,msg_link->nxt_node);
  printf("[%d]   nxt_addr = %x\n",Cmi_mype,msg_link->nxt_addr);
  printf("[%d]   msg_sz = %x\n",Cmi_mype,msg_link->msg_sz);
#endif

  /* 5. Release lock */
  clear_lock(&head_lock[dst_pe], dst_pe);
}

static void McRetrieveRemote(void)
{
  /*
   * The local host should retrieve messages from the distributed list
   * and put them in local memory, in a messages queue.
   * Steps:
   * 0) Lock list pointer.
   * 1) Replace list pointer with NULL and unlock list
   * 2) Get each message into local memory
   * 3) Enqueue list into local message queue, in reverse order
   */

  McDistList list_head;
  McDistList *cur_node;
  McMsgHdr *cur_msg; 
  int received_f;
  enum boolean bcast_msg;
  McMsgHdr *bcast_ptr;

  /* Get the head of the list */

  if (head.nxt_node == list_empty)  /* apparently there are no messages */
    return;

  /* 0) Lock list pointer. */
  set_lock(my_lock, Cmi_mype);

  /* 1) Replace list pointer with NULL and unlock list */
  list_head = head;
  head.nxt_node = list_empty;
  head.nxt_addr = NULL;
  head.msg_sz = 0;
  clear_lock(my_lock, Cmi_mype);

  /* 2) Get each message into local memory
   * Start copying the messages into local memory, putting messages into
   * a local list for future reversing.
   */
  cur_node = &list_head;
  received_f = true;

  while (cur_node->nxt_node != list_empty)
  {
    cur_msg = (McMsgHdr *)CmiAlloc(ALIGN8(cur_node->msg_sz));
    if (cur_msg ==NULL)
    {
      CmiError("%s:%d Cannot Allocate Memory\n",__FILE__,__LINE__);
      globalexit(1);
    }

    shmem_get64((long*)cur_msg, (long*)cur_node->nxt_addr,
              ALIGN8(cur_node->msg_sz)/8, cur_node->nxt_node);

    /*    CmiPrintf("PE %d incoming msg = %d msg_type = %d, size = %d\n",
	      Cmi_mype,cur_msg,cur_msg->msg_type,cur_node->msg_sz);*/

    /* If it is a broadcast message, retrieve the actual message */
    if (cur_msg->msg_type == BcastMessage)
    {

      bcast_msg = true;
      bcast_ptr = (McMsgHdr *)CmiAlloc(ALIGN8(cur_msg->bcast_msg_size));
      set_lock(&bcast_lock[cur_node->nxt_node], cur_node->nxt_node);

      /*
      CmiPrintf(
	"PE %d getting message from node %d at addr %d to %d, size=%d\n",
	Cmi_mype,cur_node->nxt_node,cur_msg->bcast.ptr,bcast_ptr,
	cur_msg->bcast_msg_size
	);
	*/
      /* Get the message */
      shmem_get64((long*)bcast_ptr,(long*)cur_msg->bcast.ptr,
		ALIGN8(cur_msg->bcast_msg_size)/8,cur_node->nxt_node);
      /* Decrement the count, and write it back to the original node. */
      /*
      CmiPrintf(
      "PE %d received broadcast message count=%d size=%d handler=%d\n",
      Cmi_mype,bcast_ptr->bcast.count,
      cur_msg->bcast_msg_size,bcast_ptr->handler
      );
      */

      bcast_ptr->bcast.count--;

      shmem_int_put(&(cur_msg->bcast.ptr->bcast.count),
		&bcast_ptr->bcast.count,1,cur_node->nxt_node);
      shmem_quiet();
      clear_lock(&bcast_lock[cur_node->nxt_node], cur_node->nxt_node);
    }
    else bcast_msg = false;

    /* Mark the remote message for future deletion */
    shmem_int_put(&(cur_node->nxt_addr->received_f),&received_f,
              1, cur_node->nxt_node);
    shmem_quiet();

    /* Add to list for reversing */
    if (bcast_msg)
    {
      McQueueAddToBack(received_token_queue,cur_msg);
      McQueueAddToBack(tmp_queue,bcast_ptr);
    }
    else 
      McQueueAddToBack(tmp_queue,cur_msg);

    /* Move pointer to next message */
    cur_node = &(cur_msg->list_node);
  }

  /* 3) Enqueue list into local message queue, in reverse order */
  while ((cur_msg = McQueueRemoveFromBack(tmp_queue)) != NULL)  {
    McQueueAddToBack(received_queue,cur_msg);
  }

  /* 4) Delete broadcast-message tokens */
  while ((cur_msg = McQueueRemoveFromBack(received_token_queue)) != NULL)  {
    CmiFree(cur_msg);
  }
  return;
}

static void McCleanUpInTransit(void)
{
  McMsgHdr *msg;
  McQueue *swap_ptr;

  /* Check broadcast message queue, to see if messages have been retrieved
   */
  while ((msg = (McMsgHdr *)McQueueRemoveFromFront(broadcast_queue)) 
	 != NULL)
  {
    if (msg->bcast.count == 0)
    {
      /* 
	 CmiPrintf("PE %d freeing broadcast message at %d\n",Cmi_mype,msg);
       */
      CmiFree(msg);
    }
    else
    {
      McQueueAddToBack(broadcast_tmp_queue,msg);
    }
  }
  /*
   * swap queues, so tmp_queue is now empty, and in_transit_queue has
   * only non-received messages.
   */
  swap_ptr = broadcast_tmp_queue;
  broadcast_tmp_queue = broadcast_queue;
  broadcast_queue = swap_ptr;

  /* 
   * Free received messages, and move others to tmp_queue.  Similar to
   * above
   */
  while ((msg = (McMsgHdr *)McQueueRemoveFromFront(in_transit_queue)) 
	 != NULL)
  {
    if (msg->received_f)
    {
      CmiFree(msg);
    }
    else
    {
      McQueueAddToBack(in_transit_tmp_queue,msg);
    }
  }
  /*
   * swap queues, so tmp_queue is now empty, and in_transit_queue has
   * only non-received messages.
   */
  swap_ptr = in_transit_tmp_queue;
  in_transit_tmp_queue = in_transit_queue;
  in_transit_queue = swap_ptr;
#ifdef DEBUG
  CmiPrintf("[%d] done in_transit_queue = %d, tmp_queue = %d\n",
	Cmi_mype,in_transit_queue->len,in_transit_tmp_queue->len);
#endif
}

/*******************************************************************
 * The following internal functions implements FIFO queues for
 * messages in the local address space.  This is used for the
 * received_queue, the in_transit_queue, and tmp_queue.  Code
 * originally comes from the Origin2000 port, with modifications.
 */
static void **McQueueAllocBlock(unsigned int len)
{
  void ** blk;

  blk=(void **)malloc(len*sizeof(void *));
  if(blk==(void **)0) {
    CmiError("Cannot Allocate Memory!\n");
    abort();
  }
  return blk;
}

static void 
McQueueSpillBlock(void **srcblk, void **destblk, 
	     unsigned int first, unsigned int len)
{
  memcpy(destblk, &srcblk[first], (len-first)*sizeof(void *));
  memcpy(&destblk[len-first],srcblk,first*sizeof(void *));
}

static McQueue * McQueueCreate(void)
{
  McQueue *queue;

  queue = (McQueue *) malloc(sizeof(McQueue));
  if(queue==(McQueue *)0) {
    CmiError("Cannot Allocate Memory!\n");
    abort();
  }
  queue->blk = McQueueAllocBlock(BLK_LEN);
  queue->blk_len = BLK_LEN;
  queue->first = 0;
  queue->len = 0;
  return queue;
}

int inside_comm = 0;

static void McQueueAddToBack(McQueue *queue, void *element)
{
  inside_comm = 1;
  if(queue->len==queue->blk_len) {
    void **blk;

    queue->blk_len *= 3;
    blk = McQueueAllocBlock(queue->blk_len);
    McQueueSpillBlock(queue->blk, blk, queue->first, queue->len);
    free(queue->blk);
    queue->blk = blk;
    queue->first = 0;
  }
#ifdef DEBUG
  CmiPrintf("[%d] Adding %x\n",Cmi_mype,element);
#endif
  queue->blk[(queue->first+queue->len++)%queue->blk_len] = element;
  inside_comm = 0;
}

static void * McQueueRemoveFromBack(McQueue *queue)
{
  void *element;
  element = (void *) 0;
  if(queue->len) {
    element = queue->blk[(queue->first+queue->len-1)%queue->blk_len];
    queue->len--;
  }
  return element;
}

static void * McQueueRemoveFromFront(McQueue *queue)
{
  void *element;
  element = (void *) 0;
  if(queue->len) {
    element = queue->blk[queue->first++];
    queue->first = (queue->first+queue->blk_len)%queue->blk_len;
    queue->len--;
  }
  return element;
}


/* Timer Functions */
#if 0

static double clocktick;
static long inittime_wallclock;
static long inittime_virtual;

void CmiTimerInit()
{
  inittime_wallclock = _rtc();
  inittime_virtual = cpused();
  clocktick = 1.0 / (sysconf(_SC_CLK_TCK));
}

double CmiWallTimer()
{
  long now;

  now = _rtc();
  return (clocktick * (now - inittime_wallclock));
}

double CmiCpuTimer()
{
  long now;

  now = cpused();
  return (clocktick * (now - inittime_virtual));
}

double CmiTimer()
{
  long now;

  now = _rtc();
  return (clocktick * (now - inittime_wallclock));
}

#elif CMK_TIMER_USE_CLOCK_GETTIME

CpvStaticDeclare(double,inittime_wallclock);
CpvStaticDeclare(double,inittime_virtual);

void CmiTimerInit(void)
{
  struct timespec temp;
  CpvInitialize(double, inittime_wallclock);
  CpvInitialize(double, inittime_virtual);
  clock_gettime(CLOCK_SGI_CYCLE, &temp);
  CpvAccess(inittime_wallclock) = (double) temp.tv_sec +
                                  1e-9 * temp.tv_nsec;
  CpvAccess(inittime_virtual) = CpvAccess(inittime_wallclock);
}

double CmiWallTimer(void)
{
  struct timespec temp;
  double currenttime;

  clock_gettime(CLOCK_SGI_CYCLE, &temp); 
  currenttime = (double) temp.tv_sec +
                1e-9 * temp.tv_nsec;
  return (currenttime - CpvAccess(inittime_wallclock));
}

double CmiCpuTimer(void)
{
  struct timespec temp;
  double currenttime;

  clock_gettime(CLOCK_SGI_CYCLE, &temp);
  currenttime = (double) temp.tv_sec +
                1e-9 * temp.tv_nsec;
  return (currenttime - CpvAccess(inittime_virtual));
}

double CmiTimer(void)
{
  return CmiCpuTimer();
}              
#endif

/*   Memory lock and unlock functions */
/*      --- on T3E, no shared memory and quickthreads are used, so memory */
/*          calls reentrant problem. these are only dummy functions */

static volatile int memflag;
void CmiMemLock() { memflag=1; }
void CmiMemUnlock() { memflag=0; }
