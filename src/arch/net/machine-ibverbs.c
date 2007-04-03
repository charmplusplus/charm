/** @file
 * Ibverbs (infiniband)  implementation of Converse NET version
 * @ingroup NET
 * contains only Ibverbs specific code for:
 * - CmiMachineInit()
 * - CmiNotifyStillIdle()
 * - DeliverViaNetwork()
 * - CommunicationServer()
 * - CmiMachineExit()

  created by 
	Sayantan Chakravorty, sayantan@gmail.com ,21st March 2007
*/

/**
 * @addtogroup NET
 * @{
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <malloc.h>
#include <getopt.h>
#include <time.h>
#include <assert.h>

#include <infiniband/verbs.h>

enum ibv_mtu mtu = IBV_MTU_2048;
static int page_size;
static int mtu_size;
static int packetSize;


typedef struct {
char none;  
} CmiIdleState;


/********
** The notify idle methods
***/

static CmiIdleState *CmiNotifyGetState(void) { return NULL; }

static void CmiNotifyStillIdle(CmiIdleState *s);

static void CmiNotifyBeginIdle(CmiIdleState *s)
{
  CmiNotifyStillIdle(s);
}

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
}

/***************
Data Structures 
***********************/

struct infiBufferPool;

/***
	This structure represents the data needed by the infiniband
	communication routines of a node
	TODO: add locking for the smp version
*/
struct infiContext {
	struct ibv_context	*context;
	int ibPort;
//	struct ibv_comp_channel *channel;
	struct ibv_pd		*pd;
	struct ibv_cq		*cq;
	struct ibv_qp		**qp; //Array of qps (numNodes long) to temporarily store the queue pairs
												//It is used between CmiMachineInit and the call to node_addresses_store
												//when the qps are stored in the corresponding OtherNodes

	struct infiAddr *localAddr; //store the lid,qpn,msn address of ur qpair until they are sent
};


/** Represents a buffer that can be used for ibverbs operation
*/
struct infiBuffer{
	char *buf;
	int size;
	int fromNode;
	struct ibv_mr *key;
};



/** At the moment it is a simple pool with just a list of buffers
	* TODO; extend it to make it an element in a linklist of pools
*/
struct infiBufferPool{
	int numBuffers;
	struct infiBuffer *buffers;
};

/** Represents a qp used to send messages to another node
 There is one for each remote node
*/
struct infiAddr {
	int lid,qpn,psn;
};

/**
 Stored in the OtherNode structure in machine-dgram.c 
 Store the per node data for ibverbs layer
*/
enum { INFI_HEADER_DATA=21,INFI_DATA};

struct infiOtherNodeData{
	struct ibv_qp *qp ;
	int state;// does it expect a packet with a header (first packet) or one without
	struct infiBufferPool *recvBufferPool;
};




struct infiContext *context;
/******************CmiMachineInit and its helper functions*/

void createLocalQps(struct ibv_device *dev,int ibPort, int myNode,int numNodes,struct infiAddr *localAddr);
static uint16_t getLocalLid(struct ibv_context *context, int port);

static void CmiMachineInit(char **argv){
	struct ibv_device **devList;
	struct ibv_device *dev;
	int ibPort;

	MACHSTATE(3,"CmiMachineInit {");
	MACHSTATE2(3,"_Cmi_numnodes %d CmiNumNodes() %d",_Cmi_numnodes,CmiNumNodes());
	
	//TODO: make the device and ibport configureable by commandline parameter
	//Check example for how to do that
	devList =  ibv_get_device_list(NULL);
	assert(devList != NULL);

	dev = *devList;
	assert(dev != NULL);

	ibPort=1;

	MACHSTATE1(3,"device name %s",ibv_get_device_name(dev));

	context = (struct infiContext *)malloc(sizeof(struct infiContext));
	
	//localAddr will store the local addresses of all the qps
	context->localAddr = (struct infiAddr *)malloc(sizeof(struct infiAddr)*_Cmi_numnodes);
	
	context->ibPort = ibPort;
	//the context for this infiniband device 
	context->context = ibv_open_device(dev);
	assert(context->context != NULL);

	//protection domain
	context->pd = ibv_alloc_pd(context->context);
	assert(context->pd != NULL);
	MACHSTATE2(3,"pd %p pd->handle %d",context->pd,context->pd->handle);
	

	mtu_size=2048;
	packetSize = mtu_size-48;//infiniband rc header size -estimate
	createLocalQps(dev,ibPort,_Cmi_mynode,_Cmi_numnodes,context->localAddr);

	
	
	MACHSTATE(3,"} CmiMachineInit");
}

/*********
	Open a qp for every processor
*****/
void createLocalQps(struct ibv_device *dev,int ibPort, int myNode,int numNodes,struct infiAddr *localAddr){
	int myLid;
	int i;
	int rx_depth=500;
	int send_wr=500;
	
	
	//find my lid
	myLid = getLocalLid(context->context,ibPort);
	
	MACHSTATE1(3,"myLid %d",myLid);

	//create a completion queue to be used with all the queue pairs
	context->cq = ibv_create_cq(context->context,rx_depth+1,NULL,NULL,0);

	assert(context->cq != NULL);
	
	MACHSTATE(3,"cq created");
	
	//array of queue pairs

	context->qp = (struct ibv_qp **)malloc(sizeof(struct ibv_qp *)*numNodes);

	{
		struct ibv_qp_init_attr initAttr = {
			.qp_type = IBV_QPT_RC,
			.send_cq = context->cq,
			.recv_cq = context->cq,
			.sq_sig_all = 0,
			.srq = NULL,
			.qp_context = NULL,
			.cap     = {
				.max_send_wr  = send_wr,
				.max_recv_wr  = rx_depth,
				.max_send_sge = 1,
				.max_recv_sge = 1
			},
		};
		struct ibv_qp_attr attr;

		attr.qp_state        = IBV_QPS_INIT;
		attr.pkey_index      = 0;
		attr.port_num        = ibPort;
		attr.qp_access_flags = 0;

/*		MACHSTATE1(3,"context->pd %p",context->pd);
		struct ibv_qp *qp = ibv_create_qp(context->pd,&initAttr);
		MACHSTATE1(3,"TEST QP %p",qp);*/

		for( i=0;i<numNodes;i++){
			if(i == myNode){
			}else{
				localAddr[i].lid = myLid;
				context->qp[i] = ibv_create_qp(context->pd,&initAttr);
			
				MACHSTATE2(3,"qp[%d] created %p",i,context->qp[i]);
			
				assert(context->qp[i] != NULL);
			
			
				ibv_modify_qp(context->qp[i], &attr,
					  IBV_QP_STATE              |
					  IBV_QP_PKEY_INDEX         |
				  	IBV_QP_PORT               |
				  	IBV_QP_ACCESS_FLAGS);		

				localAddr[i].qpn = context->qp[i]->qp_num;
				localAddr[i].psn = lrand48() & 0xffffff;
				MACHSTATE4(3,"i %d lid Ox%x qpn 0x%x psn 0x%x",i,localAddr[i].lid,localAddr[i].qpn,localAddr[i].psn);
			}
		}
	}
	MACHSTATE(3,"qps created");
};

void copyInfiAddr(ChInfiAddr *qpList){
	int qpListIdx=0;
	int i;
	MACHSTATE1(3,"copyInfiAddr _Cmi_mynode %d",_Cmi_mynode);
	for(i=0;i<_Cmi_numnodes;i++){
		if(i == _Cmi_mynode){
		}else{
			qpList[qpListIdx].lid = ChMessageInt_new(context->localAddr[i].lid);
			qpList[qpListIdx].qpn = ChMessageInt_new(context->localAddr[i].qpn);
			qpList[qpListIdx].psn = ChMessageInt_new(context->localAddr[i].psn);			
			qpListIdx++;
		}
	}
}


static uint16_t getLocalLid(struct ibv_context *dev_context, int port){
	struct ibv_port_attr attr;

	if (ibv_query_port(dev_context, port, &attr))
		return 0;

	return attr.lid;
}

/**************** END OF CmiMachineInit and its helper functions*/

struct infiBufferPool * allocateInfiBufferPool(int numRecvsPerNode,int sizePerBuffer);
void postInitialRecvs(struct infiBufferPool *recvBufferPool,struct ibv_qp *qp ,int numRecvsPerNode,int sizePerBuffer);

/* Initial the infiniband specific data for a remote node
	1. connect the qp and store it in and return it
**/
struct infiOtherNodeData *initInfiOtherNodeData(int node,int addr[3]){
	struct infiOtherNodeData * ret = malloc(sizeof(struct infiOtherNodeData));
	int err;
	ret->state = INFI_HEADER_DATA;
	ret->qp = context->qp[node];
	
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= mtu,
		.dest_qp_num		= addr[1],
		.rq_psn 		= addr[2],
		.max_dest_rd_atomic	= 1,
		.min_rnr_timer		= 12,
		.ah_attr		= {
			.is_global	= 0,
			.dlid		= addr[0],
			.sl		= 0,
			.src_path_bits	= 0,
			.port_num	= context->ibPort
		}
	};
	
	MACHSTATE2(3,"initInfiOtherNodeData %d{ qp %p",node,ret->qp);
	MACHSTATE3(3,"dlid 0x%x qp 0x%x psn 0x%x",attr.ah_attr.dlid,attr.dest_qp_num,attr.rq_psn);
	
	if (err = ibv_modify_qp(ret->qp, &attr,
	  IBV_QP_STATE              |
	  IBV_QP_AV                 |
	  IBV_QP_PATH_MTU           |
	  IBV_QP_DEST_QPN           |
	  IBV_QP_RQ_PSN             |
	  IBV_QP_MAX_DEST_RD_ATOMIC |
	  IBV_QP_MIN_RNR_TIMER)) {
			MACHSTATE1(3,"ERROR %d",err);
			CmiAbort("failed to change qp state to RTR");
	}

	MACHSTATE(3,"qp state changed to RTR");
	
	attr.qp_state 	    = IBV_QPS_RTS;
	attr.timeout 	    = 14;
	attr.retry_cnt 	    = 7;
	attr.rnr_retry 	    = 7;
	attr.sq_psn 	    = context->localAddr[node].psn;
	attr.max_rd_atomic  = 1;

	
	if (ibv_modify_qp(ret->qp, &attr,
	  IBV_QP_STATE              |
	  IBV_QP_TIMEOUT            |
	  IBV_QP_RETRY_CNT          |
	  IBV_QP_RNR_RETRY          |
	  IBV_QP_SQ_PSN             |
	  IBV_QP_MAX_QP_RD_ATOMIC)) {
			fprintf(stderr, "Failed to modify QP to RTS\n");
			exit(1);
	}
	MACHSTATE(3,"qp state changed to RTS");

	//TODO: create the pool and post the receives
	ret->recvBufferPool = allocateInfiBufferPool(100,packetSize);
	postInitialRecvs(ret->recvBufferPool,ret->qp ,100,packetSize);
	MACHSTATE(3,"} initInfiOtherNodeData");
	return ret;
}


void 	cleanUpInfiContext(){
	free(context->qp);
	context->qp = NULL;
	free(context->localAddr);
	context->localAddr= NULL;
}

struct infiBufferPool * allocateInfiBufferPool(int numRecvsPerNode,int sizePerBuffer){
	int numBuffers;
	int i;
	struct infiBufferPool *ret;

	MACHSTATE(3,"allocateInfiBufferPool");

	page_size = sysconf(_SC_PAGESIZE);
	ret = malloc(sizeof(struct infiBufferPool));
	numBuffers=ret->numBuffers = numRecvsPerNode*(_Cmi_numnodes -1 );
	
	ret->buffers = malloc(sizeof(struct infiBuffer)*numBuffers);
	
	for(i=0;i<numBuffers;i++){
		struct infiBuffer *buffer =  &(ret->buffers[i]);
		buffer->size = sizePerBuffer;
		buffer->buf = memalign(page_size,sizePerBuffer);
		buffer->key = ibv_reg_mr(context->pd,buffer->buf,buffer->size,IBV_ACCESS_LOCAL_WRITE);
	}
	return ret;
};



/**
	 Post the buffers as recv work requests
*/
void postInitialRecvs(struct infiBufferPool *recvBufferPool,struct ibv_qp *qp ,int numRecvsPerNode,int sizePerBuffer){
	int i,j;
	struct ibv_recv_wr *workRequests = malloc(sizeof(struct ibv_recv_wr)*numRecvsPerNode);
	struct ibv_sge *sgElements = malloc(sizeof(struct ibv_sge)*numRecvsPerNode);
	struct ibv_recv_wr *bad_wr;
	
	for(i=0;i<_Cmi_numnodes ;i++){
		if(i == _Cmi_mynode){
		}else{
			int startBufferIdx;
			MACHSTATE3(3,"posting %d receives for node %d of size %d",numRecvsPerNode,i,sizePerBuffer);
			if(i < _Cmi_mynode){
				startBufferIdx = i*numRecvsPerNode;
			}else{
				startBufferIdx = (i-1)*numRecvsPerNode;
			}
			for(j=0;j<numRecvsPerNode;j++){
				
				recvBufferPool->buffers[startBufferIdx+j].fromNode = i;
				
				sgElements[j].addr = (uint64_t) recvBufferPool->buffers[startBufferIdx+j].buf;
				sgElements[j].length = sizePerBuffer;
				sgElements[j].lkey = recvBufferPool->buffers[startBufferIdx+j].key->lkey;
				
				workRequests[j].wr_id = (uint64_t)&(recvBufferPool->buffers[startBufferIdx+j]);
				workRequests[j].sg_list = &sgElements[j];
				workRequests[j].num_sge = 1;
				if(j != numRecvsPerNode-1){
					workRequests[j].next = &workRequests[j+1];
				}
				
			}
			if(ibv_post_recv(qp,workRequests,&bad_wr)){
				assert(0);
			}
		}
	}

	free(workRequests);
	free(sgElements);
}





static void CmiMachineExit()
{
}

static void CmiNotifyStillIdle(CmiIdleState *s) {
}

/**
	Packetize this data and send it
**/

static void EnqueuePacket(OutgoingMsg ogm, OtherNode node, int rank,char *data,int size,int broot,int copy){
	ogm->refcount++;
	
	struct ibv_sge sendElement;
	sendElement.addr = (uintptr_t)data;
	sendElement.length = size;
	sendElement.lkey = ogm->key->lkey;

	struct ibv_send_wr wr = {
		.wr_id 	    = (uint64_t)ogm,
		.sg_list    = &sendElement,
		.num_sge    = 1,
		.opcode     = IBV_WR_SEND,
		.send_flags = IBV_SEND_SIGNALED,
		.next       = NULL 
	};
	struct ibv_send_wr *bad_wr;
	
	if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
		assert(0);
	}

};


static void CommunicationServer_nolock(int withDelayMs);

void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy){
	int size; char *data;
	
  size = ogm->size;
  data = ogm->data;

	MACHSTATE2(3,"Sending ogm %p of size %d",ogm,size);
	//First packet has dgram header, other packets dont
	
  DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, 1, broot);
	
	CmiMsgHeaderSetLength(ogm->data,ogm->size);
	ogm->refcount=0;
	ogm->key = ibv_reg_mr(context->pd,ogm->data,ogm->size,IBV_ACCESS_LOCAL_WRITE);
	CmiAssert(ogm->key->lkey != 0);
	
	while(size > packetSize){
		EnqueuePacket(ogm,node,rank,data,packetSize,broot,copy);
		size -= packetSize;
		data += packetSize;
	}
	if(size > 0){
		EnqueuePacket(ogm,node,rank,data,size,broot,copy);
	}



	CommunicationServer_nolock(0);
}


static void processRecvWC(struct ibv_wc *recvWC);
static void processSendWC(struct ibv_wc *sendWC);


static void CommunicationServer_nolock(int withDelayMs) {
	int i;
	struct ibv_wc wc[100];
	int ne;
	
	ne = ibv_poll_cq(context->cq,100,&wc[0]);
	assert(ne >=0);
	for(i=0;i<ne;i++){
		if(wc[i].status != IBV_WC_SUCCESS){
			assert(0);
		}
		switch(wc[i].opcode){
			case IBV_WC_RECV:
				//message received
				processRecvWC(&wc[i]);
				break;
			case IBV_WC_SEND:
				processSendWC(&wc[i]);
				break;
		}
			
	}

}


static void CommunicationServer(int sleepTime, int where){
	CommunicationServer_nolock(sleepTime);
}

static void processMessage(int nodeNo,int len,struct infiBuffer *buffer){
	char *msg = buffer->buf;
	
	char *newmsg;
	
	MACHSTATE2(3,"Processing packet from node %d len %d",nodeNo,len);
	
	OtherNode node = &nodes[nodeNo];
	newmsg = node->asm_msg;		
	
	/// This simple state machine determines if this packet marks the beginning of a new message
	// from another node, or if this is another in a sequence of packets
	switch(node->infiData->state){
		case INFI_HEADER_DATA:
		{
			int size;
			int rank, srcpe, seqno, magic, i;
			unsigned int broot;
			DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);
			
			CmiAssert(nodes_by_pe[srcpe] == node);
			
			CmiAssert(newmsg == NULL);
			size = CmiMsgHeaderGetLength(msg);
			if(len > size){
				CmiPrintf("size: %d, len:%d.\n", size, len);
				CmiAbort("\n\n\t\tLength mismatch!!\n\n");
			}
			newmsg = (char *)CmiAlloc(size);
      _MEMCHECK(newmsg);
      memcpy(newmsg, msg, len);
      node->asm_rank = rank;
      node->asm_total = size;
      node->asm_fill = len;
      node->asm_msg = newmsg;
			if(len == size){
				//this is the only packet for this message 
				node->infiData->state = INFI_HEADER_DATA;
			}else{
				//there are more packets following
				node->infiData->state = INFI_DATA;
			}
			break;
		}
		case INFI_DATA:
		{
			if(node->asm_fill+len > node->asm_total){
				CmiPrintf("asm_total: %d, asm_fill: %d, len:%d.\n", node->asm_total, node->asm_fill, len);
				CmiAbort("\n\n\t\tLength mismatch!!\n\n");
			}
			memcpy(newmsg + node->asm_fill,msg,len);
			node->asm_fill += len;
			if(node->asm_fill == node->asm_total){
				node->infiData->state = INFI_HEADER_DATA;
			}else{
				node->infiData->state = INFI_DATA;
			}
			break;
		}
	}
	/// if this packet was the last packet in a message ie state was 
	/// reset to infi_header_data
	
	if(node->infiData->state == INFI_HEADER_DATA){
		int total_size = node->asm_total;
		node->asm_msg = NULL;
		MACHSTATE2(3,"Message from node %d of length %d completely received",nodeNo,total_size);
		switch (node->asm_rank) {
    	case DGRAM_BROADCAST: {
				int i;
				for (i=1; i<_Cmi_mynodesize; i++){
					CmiPushPE(i, CopyMsg(newmsg, total_size));
				}
          CmiPushPE(0, newmsg);
          break;
      }
#if CMK_NODE_QUEUE_AVAILABLE
        case DGRAM_NODEBROADCAST: 
        case DGRAM_NODEMESSAGE: {
          CmiPushNode(newmsg);
          break;
        }
#endif
        default:
          CmiPushPE(node->asm_rank, newmsg);
  	}    /* end of switch */
	}

};


static void processRecvWC(struct ibv_wc *recvWC){
	struct infiBuffer *buffer = (struct infiBuffer *) recvWC->wr_id;	
	int len = recvWC->byte_len;
	int nodeNo = buffer->fromNode;
	
	processMessage(nodeNo,len,buffer);
	
	{
		struct ibv_sge list = {
			.addr 	= (uintptr_t) buffer->buf,
			.length = buffer->size,
			.lkey 	= buffer->key->lkey
		};
	
		struct ibv_recv_wr wr = {
			.wr_id = (uint64_t)buffer,
			.sg_list = &list,
			.num_sge = 1,
			.next = NULL
		};
		struct ibv_recv_wr *bad_wr;
	
		if(ibv_post_recv(nodes[nodeNo].infiData->qp,&wr,&bad_wr)){
			assert(0);
		}
	}

};


static void processSendWC(struct ibv_wc *sendWC){
	OutgoingMsg ogm = (OutgoingMsg )sendWC->wr_id;
	ogm->refcount--;
	if(ogm->refcount == 0){
		ibv_dereg_mr(ogm->key);
		GarbageCollectMsg(ogm);
	}
};

