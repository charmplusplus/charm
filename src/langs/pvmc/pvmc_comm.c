#include <stddef.h>
#include "converse.h"
#include "pvmc.h"

CpvStaticDeclare(CmmTable,seq_table);

CpvStaticDeclare(int,pvmc_control_handler);
CpvStaticDeclare(int,pvmc_msg_handler);
CpvStaticDeclare(int*,send_seq_num);
CpvStaticDeclare(int*,recv_seq_num);

CpvExtern(int,pvmc_barrier_num);
CpvExtern(int,pvmc_at_barrier_num);

typedef struct msg_hdr_struct {
  char handler[CmiMsgHeaderSizeBytes];
  int sender;
  unsigned int seq_num;
} msg_hdr;

typedef struct control_msg_struct {
  char handler[CmiMsgHeaderSizeBytes];
  int type;
} control_msg;

static void pvmc_control_handler_func();
static void pvmc_msg_handler_func();

void pvmc_init_comm(void)
{
  int i;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_init_comm()\n",MYPE(),pvm_mytid());
#endif

  CpvInitialize(CmmTable,seq_table);
  CpvAccess(seq_table) = CmmNew();

  CpvInitialize(int,pvmc_control_handler);
  CpvAccess(pvmc_control_handler)=
    CmiRegisterHandler(pvmc_control_handler_func);
  
  CpvInitialize(int,pvmc_msg_handler);
  CpvAccess(pvmc_msg_handler)=CmiRegisterHandler(pvmc_msg_handler_func);

  CpvInitialize(int*,recv_seq_num);
  CpvAccess(recv_seq_num)=MALLOC(CmiNumPes()*sizeof(int));

  if (CpvAccess(recv_seq_num)==NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_init_comm() can't allocate seq buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    exit(1);
  }
  for(i=0; i<CmiNumPes(); i++)
    CpvAccess(recv_seq_num)=0;

  CpvInitialize(int*,send_seq_num);
  CpvAccess(send_seq_num)=MALLOC(CmiNumPes()*sizeof(int));

  if (CpvAccess(send_seq_num)==NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_init_comm() can't allocate seq buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    exit(1);
  }
  for(i=0; i<CmiNumPes(); i++)
    CpvAccess(send_seq_num)[i]=0;

}

void pvmc_send_control_msg(int type, int pe)
{
  control_msg *msg;

  msg=CmiAlloc(sizeof(control_msg));
  msg->type=type;
  CmiSetHandler(msg,CpvAccess(pvmc_control_handler));
  CmiSyncSendAndFree(pe,sizeof(control_msg),msg);
}

static void pvmc_control_handler_func(control_msg *msg)
{
  switch (msg->type)  {
  case PVMC_CTRL_AT_BARRIER:
    CpvAccess(pvmc_at_barrier_num)++;
    break;
  case PVMC_CTRL_THROUGH_BARRIER:
    CpvAccess(pvmc_barrier_num)++;
    break;
  case PVMC_CTRL_KILL:
    ConverseExit();
    exit(0);
    break;
  default:
    PRINTF("WARNING: %s:%d, Illegal control message\n",__FILE__,__LINE__);
  }
}

static void pvmc_msg_handler_func(void *msg)
{
  int seq_num;
  int sender;
  int pvm_tag;
  int tags[2];
  int rtags[2];

  sender=((msg_hdr *)msg)->sender;
  seq_num=((msg_hdr *)msg)->seq_num;
  
  tags[0]=sender;
  tags[1]=pvmc_gettag((char *)msg+sizeof(msg_hdr));
  CmmPut(CpvAccess(seq_table),2,tags,msg);
}

int pvm_kill(int tid)
{
  control_msg *exit_msg;
  
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_kill(%d)\n",
	MYPE(),pvm_mytid(),tid);
#endif
  pvmc_send_control_msg(PVMC_CTRL_KILL,TID2PE(tid));
}

int pvm_send(int pvm_tid, int tag)
{
  void *msg;
  int msg_sz, conv_tid, conv_tag;
  
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_send(%d,%d)\n",
	MYPE(),pvm_mytid(),pvm_tid,tag);
#endif
  pvmc_settidtag(pvm_mytid(),tag);

  if ((pvm_tid<1) || ( pvm_tid > CmiNumPes())) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_send() illegal tid %d\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__,pvm_tid);
    return -1;
  } else conv_tid = pvm_tid-1;

  if (tag<0) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_send() illegal tag\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  } else conv_tag = tag;

  msg_sz = sizeof(msg_hdr)+pvmc_sendmsgsz();
  msg = CmiAlloc(msg_sz);

  if (msg==NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_send() can't alloc msg buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }

  CmiSetHandler(msg,CpvAccess(pvmc_msg_handler));
  ((msg_hdr *)msg)->sender=MYPE();
  ((msg_hdr *)msg)->seq_num=CpvAccess(send_seq_num)[conv_tid];
  CpvAccess(send_seq_num)[conv_tid]++;
  
  pvmc_packmsg((char *)msg + (int)sizeof(msg_hdr));
  CmiSyncSendAndFree(conv_tid,msg_sz,msg);
  return 0;
}

int pvm_mcast(int *tids, int ntask, int msgtag)
{
  int i;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_mcast(%x,%d,%d)\n",
	MYPE(),pvm_mytid(),tids,ntask,msgtag);
#endif
  for(i=0;i<ntask;i++)
    pvm_send(tids[i],msgtag);
}

int pvm_nrecv(int tid, int tag)
{
  int conv_tid, conv_tag;
  void *msg;
  int tags[2];
  int rtags[2];
  int sender, seq_num;
  int rbuf;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_nrecv(%d,%d)\n",
	MYPE(),pvm_mytid(),tid,tag);
#endif

  if (tid==-1)
    conv_tid=CmmWildCard;
  else conv_tid=tid-1;

  if (tag==-1)
    conv_tag=CmmWildCard;
  else conv_tag=tag;

  /*
   * Empty messages from machine layer.
   */

  while(CmiDeliverMsgs(1)==0)
    ;

  /*
   *  See if the message is already in the tag table and extract it.
   */
  
  tags[0]=conv_tid;
  tags[1]=conv_tag;
  msg=CmmGet(CpvAccess(seq_table),2,tags,rtags);
  if (msg!=NULL) {
    sender = rtags[0];
    /*
    seq_num = CpvAccess(recv_seq_num)[sender];

    if ((((msg_hdr *)msg)->seq_num) != seq_num)
      PRINTF("tid=%d:%s:%d pvm_recv() seq number mismatch, I'm confused\n",
	     tid,__FILE__,__LINE__);
    else CpvAccess(recv_seq_num)[sender]++;
    */


    rbuf=pvm_setrbuf(pvm_mkbuf(PvmDataRaw));
    if (rbuf > 0)
      {
#ifdef PVM_DEBUG
      PRINTF("Pe(%d) tid=%d:%s:%d pvm_nrecv() says pvm_setrbuf=%d\n",
	MYPE(),tid,__FILE__,__LINE__,rbuf);
#endif
      pvm_freebuf(rbuf);
      }
    pvmc_unpackmsg(msg,(char *)msg+sizeof(msg_hdr));

#ifdef PVM_DEBUG
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_nrecv() returning pvm_getrbuf()=%d\n",
	MYPE(),tid,__FILE__,__LINE__,pvm_getrbuf());
#endif
    return pvm_getrbuf();
  }
  else return 0;  /* Non blocking receive returns immediately. */
}

int pvm_recv(int tid, int tag)
{
  int bufid=0;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_recv(%d,%d)\n",
	MYPE(),pvm_mytid(),tid,tag);
#endif
  while (bufid==0)
    bufid=pvm_nrecv(tid,tag);

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_recv(%d,%d) returning %d\n",
	MYPE(),pvm_mytid(),tid,tag,bufid);
#endif

  return bufid;
}

int pvm_probe(int tid, int tag)
{
  int conv_tid, conv_tag;
  void *msg;
  int tags[2];
  int rtags[2];
  int sender, seq_num;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_probe(%d,%d)\n",
	MYPE(),pvm_mytid(),tid,tag);
#endif
  if (tid==-1)
    conv_tid=CmmWildCard;
  else conv_tid=tid;

  if (tag==-1)
    conv_tag=CmmWildCard;
  else conv_tag=tag;

  /*
   * Empty messages from machine layer.
   */

  while(CmiDeliverMsgs(1)==0)
    ;

  /*
   *  See if the message is already in the tag table
   */
  
  tags[0]=conv_tid;
  tags[1]=conv_tag;
  msg=CmmProbe(CpvAccess(seq_table),2,tags,rtags);
  if (msg!=NULL) {
    /*
    sender = rtag[0];
    seq_num = CpvAccess(recv_seq_num)[sender];

    if ((((msg_hdr *)msg)->seq_num) != seq_num)
      PRINTF("Pe(%d) tid=%d:%s:%d pvm_recv() seq num mismatch, I'm confused\n",
             MYPE(),pvm_mytid(),__FILE__,__LINE__);
    else CpvAccess(recv_seq_num)[sender]++;
    */

  /*
   * We will just unpack the message, so bufinfo works, but this
   * should really just set up what bufinfo needs and unpack the
   * rest later
   */
    pvmc_unpackmsg(msg,(char *)msg+sizeof(msg_hdr));


#ifdef PVM_DEBUG
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_probe() returning pvm_getrbuf()=%d\n",
	MYPE(),tid,__FILE__,__LINE__,pvm_getrbuf());
#endif
    return pvm_getrbuf();
  }
  else return 0;  /* Probe returns immediately. */
}
