/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile$
 *      $Author$        $Locker$                $State$
 *      $Revision$      $Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

/*
 * This is a complete port, but could be made considerably more efficient
 * by handling asynchronous messages correctly, ie. without doing
 * an extra copy and synchronous send
 */

#include <stdlib.h>
#include <malloc.h>
#include <mpp/shmem.h>
#include "converse.h"

/*
 *  We require statically allocated variables for locks.  This defines
 *  the max number of processors available.
 */
#define MAX_PES 8192

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
extern void *FIFO_Create(void);
extern void FIFO_EnQueue(void *, void *);

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
  enum boolean received_f;
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

/*
 * head is the pointer to my next incoming message.
 */
static McDistList head;

/* Static variables are necessary for locks. */
static long *my_lock;
static long head_lock[MAX_PES];

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

  dup_msg = (McMsgHdr *)CmiAlloc(size);
  memcpy(dup_msg,msg,size);

  McRetrieveRemote();

  if (dest_pe == Cmi_mype)
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),dup_msg);
  else
  {
    McEnqueueRemote(dup_msg,size,dest_pe); 
  }
}

CmiCommHandle CmiAsyncSendFn(int dest_pe, int size, char *msg)
{
  CmiSyncSendFn(dest_pe, size, msg);
  return 1;
}

void CmiFreeSendFn(int dest_pe, int size, char *msg)
{
  /* No need to copy message, since we will immediately free it */
  McRetrieveRemote();

  if (dest_pe == Cmi_mype)
    FIFO_EnQueue(CpvAccess(CmiLocalQueue),msg);
  else
  {
    McEnqueueRemote(msg,size,dest_pe); 
  }
}

void CmiSyncBroadcastFn(int size, char *msg)
{
  int i;
  for(i=0; i<Cmi_numpes; i++)
    if (i != Cmi_mype)
      CmiSyncSendFn(i, size, msg);
}

CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size,msg);
  return 1;
}

void CmiFreeBroadcastFn(int size, char *msg)
{
  CmiSyncBroadcastFn(size,msg);
  CmiFree(msg);
}

void CmiSyncBroadcastAllFn(int size, char *msg)
{
  int i;
  for(i=0; i<Cmi_numpes; i++)
      CmiSyncSendFn(i, size, msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg)
{
  CmiSyncBroadcastAllFn(size,msg);
  return 1;
}

void CmiFreeBroadcastAllFn(int size, char *msg)
{
  CmiSyncBroadcastAllFn(size,msg);
  CmiFree(msg);
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
  CmiSpanTreeInit();
  CmiTimerInit();
  McInit();
  CthInit(argv);
  ConverseCommonInit(argv);
  if (initret==0)
  {
    fn(argc,argv);
    if (usched==0) CsdScheduler(-1);
    ConverseExit();
  }
}

void ConverseExit()
{
  ConverseCommonExit();
  exit(0);
}

void CmiNotifyIdle(void)
{
  /* Use this opportunity to clean up the in_transit_queue */
  McCleanUpInTransit();
}

/**********************************************************************
 * Mc Functions:
 * Mc functions are used internally in machine.c only
 */
static void McInit(void)
{
  CpvInitialize(void *, CmiLocalQueue);
  CpvAccess(CmiLocalQueue) = FIFO_Create();
  Cmi_mype = _my_pe();
  Cmi_numpes = _num_pes();
  Cmi_myrank = 0;
  shmem_set_cache_inv();

  McInitList();
}

static void McInitList(void)
{
  int i;

  received_queue = McQueueCreate();
  tmp_queue = McQueueCreate();
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
    head_lock[i] = 0;
  my_lock = &(head_lock[Cmi_mype]);
  barrier();
  shmem_clear_lock(my_lock);
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
#ifdef DEBUG
  printf("[%d] Locking on %d:%lx\n",Cmi_mype,
         dst_pe,&(head_lock[dst_pe]));
#endif

  shmem_set_lock(&(head_lock[dst_pe]));

#ifdef DEBUG
  printf("[%d] Locked on %d:%lx\n",Cmi_mype,
         dst_pe,&(head_lock[dst_pe]));
#endif

  /* 4. Swap the list pointer with that on the other node.
   */
  /* First, get current head pointer, and stick it in this 
   * message data area.
   */
  shmem_get(msg_link, &head, sizeof(McDistList)/sizeof(int), dst_pe);
  /* Next, write the new message into the top of the list */
  shmem_put(&head, &tmp_link, sizeof(McDistList)/sizeof(int),dst_pe);

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

#ifdef DEBUG
  printf("[%d] Releasing lock %d:%lx\n",Cmi_mype,
         dst_pe,&(head_lock[dst_pe]));
#endif

  /* 5. Release lock */
  shmem_clear_lock(&(head_lock[dst_pe]));

#ifdef DEBUG
  printf("[%d] Released lock %d:%lx\n",Cmi_mype,
         dst_pe,&(head_lock[dst_pe]));
#endif
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

#ifdef DEBUG
  printf("[%d] Start front = %lx, back = %lx\n",Cmi_mype,received_queue_front,received_queue_back);
#endif
  /* Get the head of the list */

  if (head.nxt_node == list_empty)  /* apparently there are no messages */
    return;
#ifdef DEBUG
  printf("[%d] Locking on %lx\n",Cmi_mype,
         my_lock);
#endif
  /* 0) Lock list pointer. */
  shmem_set_lock(my_lock);
#ifdef DEBUG
  printf("[%d] Locked on %lx\n",Cmi_mype,
         my_lock);
#endif
  /* 1) Replace list pointer with NULL and unlock list */
  list_head = head;
  head.nxt_node = list_empty;
  head.nxt_addr = NULL;
  head.msg_sz = 0;
  shmem_clear_lock(my_lock);
#ifdef DEBUG
  printf("[%d] Released lock on %lx\n",Cmi_mype,
         my_lock);
#endif

  /* 2) Get each message into local memory
   * Start copying the messages into local memory, putting messages into
   * a local list for future reversing.
   */
  cur_node = &list_head;
  received_f = true;

  while (cur_node->nxt_node != list_empty)
  {
#ifdef DEBUG
    printf("%d allocating %d bytes\n",Cmi_mype,cur_node->msg_sz);
#endif
    cur_msg = (McMsgHdr *)CmiAlloc(cur_node->msg_sz);
    if (cur_msg ==NULL)
    {
      CmiError("%s:%d Cannot Allocate Memory\n",__FILE__,__LINE__);
      exit(1);
    }

#ifdef DEBUG
    printf("[%d] Retrieving message from [%d]:%x size %d at %x to %x\n",
           Cmi_mype,cur_node->nxt_node,cur_node->nxt_addr,
           cur_node->msg_sz,cur_node,cur_msg);
#endif

    shmem_get(cur_msg, cur_node->nxt_addr,
              cur_node->msg_sz/8, cur_node->nxt_node);

#ifdef DEBUG
    printf("[%d]   nxt_node = %d\n",
	   Cmi_mype,cur_msg->list_node.nxt_node);
    printf("[%d]   nxt_addr = %x\n",
	   Cmi_mype,cur_msg->list_node.nxt_addr);
    printf("[%d]   msg_sz = %x\n",
#endif

    /* Mark the remote message for future deletion */
    shmem_put(&(cur_node->nxt_addr->received_f),&received_f,
              1, cur_node->nxt_node);

    /* Add to list for reversing */
    McQueueAddToBack(tmp_queue,cur_msg);

    /* Move pointer to next message */
    cur_node = &(cur_msg->list_node);
  }

  /* 3) Enqueue list into local message queue, in reverse order */
  while ((cur_msg = McQueueRemoveFromBack(tmp_queue)) != NULL)  {
    McQueueAddToBack(received_queue,cur_msg);
  }
  return;
}

static void McCleanUpInTransit(void)
{
  McMsgHdr *msg;
  McQueue *swap_ptr;

#ifdef DEBUG
  CmiPrintf("[%d] in_transit_queue = %d, tmp_queue = %d\n",
	Cmi_mype,in_transit_queue->len,in_transit_tmp_queue->len);
#endif
  /* 
   * Free received messages, and move others to tmp_queue
   */
  while ((msg = (McMsgHdr *)McQueueRemoveFromFront(in_transit_queue)) 
	 != NULL)
  {
    if (msg->received_f)
    {
#ifdef DEBUG
      CmiPrintf("[%d] Freeing message at %x\n",Cmi_mype,msg);
#endif
      CmiFree(msg);
    }
    else
    {
#ifdef DEBUG
      CmiPrintf("[%d] Not freeing message at %x\n",Cmi_mype,msg);
#endif
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

static void McQueueAddToBack(McQueue *queue, void *element)
{
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

