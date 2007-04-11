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
static int tokensPerProcessor; /*number of outstanding sends and receives between any two nodes*/
static int sendPacketPoolSize; /*total number of send buffers created*/

static double _startTime=0;
static int regCount;

static int pktCount;
static int msgCount;


static double regTime;

#define CMK_IBVERBS_STATS 0

#define WC_LIST_SIZE 100
#define WC_BUFFER_SIZE 100

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

/** Represents a buffer that is used to receive messages
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


typedef struct infiPacketStruct {	
	char *buf;
	int size;
	struct ibv_mr *key;
	struct OtherNodeStruct *destNode;
	struct infiPacketStruct *next;
}* infiPacket;

/*
typedef struct infiBufferedWCStruct{
	struct ibv_wc wcList[WC_BUFFER_SIZE];
	int count;
	struct infiBufferedWCStruct *next,*prev;
} * infiBufferedWC;
*/

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
	struct ibv_cq		*sendCq;
	struct ibv_cq   *recvCq;
	struct ibv_qp		**qp; //Array of qps (numNodes long) to temporarily store the queue pairs
												//It is used between CmiMachineInit and the call to node_addresses_store
												//when the qps are stored in the corresponding OtherNodes

	struct infiAddr *localAddr; //store the lid,qpn,msn address of ur qpair until they are sent

	infiPacket infiPacketFreeList; 
	
/*	infiBufferedWC infiBufferedRecvList;*/
};

static struct infiContext *context;

static inline infiPacket newPacket(int size){
	infiPacket pkt = malloc(sizeof(struct infiPacketStruct));
	pkt->size = size;
	pkt->buf = malloc(sizeof(char)*size);
	pkt->next = NULL;
	pkt->destNode = NULL;
	pkt->key = ibv_reg_mr(context->pd,pkt->buf,pkt->size,IBV_ACCESS_LOCAL_WRITE);
	
	return pkt;
};

#define FreeInfiPacket(pkt){ \
	pkt->next = context->infiPacketFreeList; \
	context->infiPacketFreeList = pkt; \
}

#define MallocInfiPacket(pkt) { \
	infiPacket p = context->infiPacketFreeList; \
	if(p == NULL){ p = newPacket(packetSize);} \
	         else{context->infiPacketFreeList = p->next; } \
	pkt = p;\
}


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
	int tokensLeft;
	int nodeNo;
	
	int broot;//needed to store the root of a multi-packet broadcast sent along a spanning tree or hypercube
};





/******************CmiMachineInit and its helper functions*/

void createLocalQps(struct ibv_device *dev,int ibPort, int myNode,int numNodes,struct infiAddr *localAddr);
static uint16_t getLocalLid(struct ibv_context *context, int port);

static void CmiMachineInit(char **argv){
	struct ibv_device **devList;
	struct ibv_device *dev;
	int ibPort;
	int i;
	infiPacket *pktPtrs;

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
	

	mtu_size=1048;
	packetSize = mtu_size-48;//infiniband rc header size -estimate
	tokensPerProcessor=350;
	createLocalQps(dev,ibPort,_Cmi_mynode,_Cmi_numnodes,context->localAddr);
		
	//create the pool of arrays
	sendPacketPoolSize = (_Cmi_numnodes-1)*(tokensPerProcessor);
	context->infiPacketFreeList=NULL;
	pktPtrs = malloc(sizeof(infiPacket)*sendPacketPoolSize);
	//Silly way of allocating the memory buffers (slow as well) but simplifies the code
	for(i=0;i<sendPacketPoolSize;i++){
		MallocInfiPacket(pktPtrs[i]);	
	}
	for(i=0;i<sendPacketPoolSize;i++){
		FreeInfiPacket(pktPtrs[i]);	
	}
	free(pktPtrs);
	
/*	context->infiBufferedRecvList = NULL;*/
#if CMK_IBVERBS_STATS	
	regCount =0;
	regTime  = 0;

	pktCount=0;
	msgCount=0;
#endif	
	MACHSTATE(3,"} CmiMachineInit");
}

/*********
	Open a qp for every processor
*****/
void createLocalQps(struct ibv_device *dev,int ibPort, int myNode,int numNodes,struct infiAddr *localAddr){
	int myLid;
	int i;
	
	
	//find my lid
	myLid = getLocalLid(context->context,ibPort);
	
	MACHSTATE1(3,"myLid %d",myLid);

	//create a completion queue to be used with all the queue pairs
	context->sendCq = ibv_create_cq(context->context,(tokensPerProcessor*(numNodes-1))+1,NULL,NULL,0);
	assert(context->sendCq != NULL);
	

	context->recvCq = ibv_create_cq(context->context,(tokensPerProcessor*(numNodes-1))+1,NULL,NULL,0);
	assert(context->recvCq != NULL);
	
	MACHSTATE(3,"cq created");
	
	//array of queue pairs

	context->qp = (struct ibv_qp **)malloc(sizeof(struct ibv_qp *)*numNodes);

	{
		struct ibv_qp_init_attr initAttr = {
			.qp_type = IBV_QPT_RC,
			.send_cq = context->sendCq,
			.recv_cq = context->recvCq,
			.sq_sig_all = 0,
			.srq = NULL,
			.qp_context = NULL,
			.cap     = {
				.max_send_wr  = tokensPerProcessor,
				.max_recv_wr  = tokensPerProcessor,
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
void postInitialRecvs(int node, struct infiBufferPool *recvBufferPool,struct ibv_qp *qp ,int numRecvsPerNode,int sizePerBuffer);

/* Initial the infiniband specific data for a remote node
	1. connect the qp and store it in and return it
**/
struct infiOtherNodeData *initInfiOtherNodeData(int node,int addr[3]){
	struct infiOtherNodeData * ret = malloc(sizeof(struct infiOtherNodeData));
	int err;
	ret->state = INFI_HEADER_DATA;
	ret->qp = context->qp[node];
	ret->tokensLeft = tokensPerProcessor;
	ret->nodeNo = node;

	
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
	ret->recvBufferPool = allocateInfiBufferPool(tokensPerProcessor,packetSize);
	postInitialRecvs(node,ret->recvBufferPool,ret->qp ,tokensPerProcessor,packetSize);
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
void postInitialRecvs(int node,struct infiBufferPool *recvBufferPool,struct ibv_qp *qp ,int numRecvsPerNode,int sizePerBuffer){
	int j;
	struct ibv_recv_wr *workRequests = malloc(sizeof(struct ibv_recv_wr)*numRecvsPerNode);
	struct ibv_sge *sgElements = malloc(sizeof(struct ibv_sge)*numRecvsPerNode);
	struct ibv_recv_wr *bad_wr;
	
			int startBufferIdx;
			MACHSTATE3(3,"posting %d receives for node %d of size %d",numRecvsPerNode,node,sizePerBuffer);
			if(node < _Cmi_mynode){
				startBufferIdx = node*numRecvsPerNode;
			}else{
				startBufferIdx = (node-1)*numRecvsPerNode;
			}
			for(j=0;j<numRecvsPerNode;j++){
				
				recvBufferPool->buffers[startBufferIdx+j].fromNode = node;
				
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

	free(workRequests);
	free(sgElements);
}




static inline void CommunicationServer_nolock(int toBuffer); //if buffer ==1 recvd messages are buffered but not processed

static void CmiMachineExit()
{
//	printf("[%d] totalTime %.6lf total RegTime %.6lf total RegCount %d avg regTime %.6lf \n",_Cmi_mynode,CmiWallTimer()-_startTime,regTime,regCount,regTime/(double )regCount);
//	printf("[%d] calling CmiMachineExit \n",_Cmi_mynode);
#if CMK_IBVERBS_STATS	
	printf("[%d] msgCount %d pktCount %d packetSize %d # 0 tokens %d time loss due to 0 tokens %.6lf \n",_Cmi_mynode,msgCount,pktCount,packetSize,regCount,regTime);
#endif
}

static void CmiNotifyStillIdle(CmiIdleState *s) {
	CommunicationServer_nolock(0);
}

/**
	Packetize this data and send it
**/

static inline void pollRecvCq(int toBuffer);
static inline void pollSendCq(void);

static void EnqueuePacket(OutgoingMsg ogm, OtherNode node, int rank,char *data,int size,int broot,int copy){
	int full=0;
	infiPacket packet;
	double _regStartTime;
	MallocInfiPacket(packet);
	packet->destNode = node;
	//copy the data
	memcpy(packet->buf,data,size);
	
#if CMK_IBVERBS_STATS	
	pktCount++;
#endif

	struct ibv_sge sendElement = {
		.addr = (uintptr_t)packet->buf,
		.length = size,
		.lkey = packet->key->lkey
	};

	struct ibv_send_wr wr = {
		.wr_id 	    = (uint64_t)packet,
		.sg_list    = &sendElement,
		.num_sge    = 1,
		.opcode     = IBV_WR_SEND,
		.send_flags = IBV_SEND_SIGNALED,
		.next       = NULL 
	};
	struct ibv_send_wr *bad_wr;
	
#if CMK_IBVERBS_STATS	
	if(node->infiData->tokensLeft == 0){
/*		CmiPrintf("[%d] Number of tokens to node %d is 0 \n",_Cmi_mynode,node->infiData->nodeNo); */
		full = 1;
		regCount++;
		_regStartTime = CmiWallTimer();
	}
#endif	
	while(node->infiData->tokensLeft == 0){
		CommunicationServer_nolock(1); //buffer any messages received now. do not process them as processing a broadcast request can result in messages being sent to a processor in turn. It would result in a broadcast message getting sent in between packets of another message
//		pollSendCq();
	}
	
#if CMK_IBVERBS_STATS	
	if(full){
/*		CmiPrintf("[%d] Number of tokens to node %d is no longer 0 but %d \n",_Cmi_mynode,node->infiData->nodeNo,node->infiData->tokensLeft);*/
		regTime += (CmiWallTimer()-_regStartTime);
	}
#endif	
	
	if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
		assert(0);
	}
	node->infiData->tokensLeft--;

	MACHSTATE4(3,"Packet send ogm %p size %d packet %p tokensLeft %d",ogm,size,packet,packet->destNode->infiData->tokensLeft);

};



void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy){
	int size; char *data;
	
  size = ogm->size;
  data = ogm->data;

#if CMK_IBVERBS_STATS	
	msgCount++;
#endif

	MACHSTATE3(3,"Sending ogm %p of size %d to %d",ogm,size,node->infiData->nodeNo);
	//First packet has dgram header, other packets dont
	
  DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, 1, broot);
	
	CmiMsgHeaderSetLength(ogm->data,ogm->size);
	
	while(size > packetSize){
		EnqueuePacket(ogm,node,rank,data,packetSize,broot,copy);
		size -= packetSize;
		data += packetSize;
	}
	if(size > 0){
		EnqueuePacket(ogm,node,rank,data,size,broot,copy);
	}
	MACHSTATE3(3,"DONE Sending ogm %p of size %d to %d",ogm,size,node->infiData->nodeNo);
}


static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer);
static inline void processSendWC(struct ibv_wc *sendWC);
static int _count=0;

static inline  void CommunicationServer_nolock(int toBuffer) {
	_count++;
	MACHSTATE(2,"CommServer_nolock{");
	

	pollRecvCq(toBuffer);
	

	pollSendCq();
	
	MACHSTATE(2,"} CommServer_nolock ne");
	
}
/*
static inline infiBufferedWC createInfiBufferedWC(){
	infiBufferedWC ret = malloc(sizeof(struct infiBufferedWCStruct));
	ret->count = 0;
	ret->next = ret->prev =NULL;
	return ret;
}*/

/****
	The buffered recvWC are stored in a doubly linked list of 
	arrays or blocks of wcs.
	To keep the average insert cost low, a new block is added 
	to the top of the list. (resulting in a reverse seq of blocks)
	Within a block however wc are stored in a sequence
*****/
/*static  void insertBufferedRecv(struct ibv_wc *wc){
	infiBufferedWC block;
	MACHSTATE(3,"Insert Buffered Recv called");
	if( context->infiBufferedRecvList ==NULL){
		context->infiBufferedRecvList = createInfiBufferedWC();
		block = context->infiBufferedRecvList;
	}else{
		if(context->infiBufferedRecvList->count == WC_BUFFER_SIZE){
			block = createInfiBufferedWC();
			context->infiBufferedRecvList->prev = block;
			block->next = context->infiBufferedRecvList;
			context->infiBufferedRecvList = block;
		}else{
			block = context->infiBufferedRecvList;
		}
	}
	
	block->wcList[block->count] = *wc;
	block->count++;
};
*/


/********
go through the blocks of bufferedWC. Process the last block first.
Then the next one and so on. (Processing within a block should happen
in sequence).
Leave the last block in place to avoid having to allocate again
******/
/*static inline void processBufferedRecvList(){
	infiBufferedWC start;
	start = context->infiBufferedRecvList;
	while(start->next != NULL){
		start = start->next;
	}
	while(start != NULL){
		int i=0;
		infiBufferedWC tmp;
		for(i=0;i<start->count;i++){
			processRecvWC(&start->wcList[i]);
		}
		if(start != context->infiBufferedRecvList){
			//not the first one
			tmp = start;
			start = start->prev;
			free(tmp);
			start->next = NULL;
		}else{
			start = start->prev;
		}
	}
	context->infiBufferedRecvList->next = NULL;
	context->infiBufferedRecvList->prev = NULL;
	context->infiBufferedRecvList->count = 0;
}
*/

static inline void pollRecvCq(const int toBuffer){
	int i;
	int ne;
	struct ibv_wc wc[WC_LIST_SIZE];
	

	ne = ibv_poll_cq(context->recvCq,WC_LIST_SIZE,&wc[0]);
//	assert(ne >=0);
	
	
	for(i=0;i<ne;i++){
		if(wc[i].status != IBV_WC_SUCCESS){
			assert(0);
		}
		switch(wc[i].opcode){
			case IBV_WC_RECV:
					processRecvWC(&wc[i],toBuffer);
				break;
			default:
				CmiAbort("Wrong type of work completion object in recvq");
				break;
		}
			
	}

}

static inline void pollSendCq(){
	int i;
	int ne;
	struct ibv_wc wc[WC_LIST_SIZE];

	ne = ibv_poll_cq(context->sendCq,WC_LIST_SIZE,&wc[0]);
//	assert(ne >=0);
	
	
	for(i=0;i<ne;i++){
		if(wc[i].status != IBV_WC_SUCCESS){
			assert(0);
		}
		switch(wc[i].opcode){
			case IBV_WC_SEND:{
				//message received
				processSendWC(&wc[i]);
				
				break;
				}
			default:
				CmiAbort("Wrong type of work completion object in recvq");
				break;
		}
			
	}
}

static void CommunicationServer(int sleepTime, int where){
	CommunicationServer_nolock(0);
}

static inline void processMessage(int nodeNo,int len,struct infiBuffer *buffer,const int toBuffer){
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
			
//			CmiAssert(nodes_by_pe[srcpe] == node);
			
//			CmiAssert(newmsg == NULL);
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
			node->infiData->broot = broot;
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
			if(node->asm_fill + len < node->asm_total && len != packetSize){
				CmiPrintf("from node %d asm_total: %d, asm_fill: %d, len:%d.\n",node->infiData->nodeNo, node->asm_total, node->asm_fill, len);
				CmiAbort("packet in the middle does not have expected length");
			}
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
		MACHSTATE3(3,"Message from node %d of length %d completely received msg %p",nodeNo,total_size,newmsg);

#if CMK_BROADCAST_SPANNING_TREE
        if (node->asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || node->asm_rank == DGRAM_NODEBROADCAST
#endif
           )
          SendSpanningChildren(NULL, 0, total_size, newmsg, node->infiData->broot, node->asm_rank);
#elif CMK_BROADCAST_HYPERCUBE
        if (node->asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || node->asm_rank == DGRAM_NODEBROADCAST
#endif
           )
          SendHypercube(NULL, 0, total_size, newmsg, node->infiData->broot, node->asm_rank);
#endif


		
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
				{
					
          CmiPushPE(node->asm_rank, newmsg);
				}
  	}    /* end of switch */
	}

};


static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer){
	struct infiBuffer *buffer = (struct infiBuffer *) recvWC->wr_id;	
	int len = recvWC->byte_len;
	int nodeNo = buffer->fromNode;
	
	processMessage(nodeNo,len,buffer,toBuffer);
	
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


static inline  void processSendWC(struct ibv_wc *sendWC){
	infiPacket packet = (infiPacket )sendWC->wr_id;
	
	packet->destNode->infiData->tokensLeft++;
	
	MACHSTATE2(3,"Packet send complete packet %p tokensLeft %d",packet,packet->destNode->infiData->tokensLeft);

	FreeInfiPacket(packet);
};

