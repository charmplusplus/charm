/** @file
			size = CmiMsgHeaderGetLength(msg);
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
static int dataSize;
static int rdma;
static int rdmaThreshold;
static int firstBinSize;
static int blockAllocRatio;
static int blockThreshold;


static int maxRecvBuffers;
static int maxTokens;
//static int tokensPerProcessor; /*number of outstanding sends and receives between any two nodes*/
static int sendPacketPoolSize; /*total number of send buffers created*/

static double _startTime=0;
static int regCount;

static int pktCount;
static int msgCount;


static double regTime;

static double processBufferedTime;
static int processBufferedCount;

#define CMK_IBVERBS_STATS 0
#define CMK_IBVERBS_TOKENS_FLOW 1
#define CMK_IBVERBS_INCTOKENS 0 //never turn this on 
#define CMK_IBVERBS_DEBUG 1

#define WC_LIST_SIZE 100
/*#define WC_BUFFER_SIZE 100*/

#define INCTOKENS_FRACTION 0.04
#define INCTOKENS_INCREASE .50

struct infiIncTokenAckPacket{
	int a;
};




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

/******
	This is a header attached to the beginning of every infiniband packet
*******/
#define INFIPACKETCODE_DATA 1
#define INFIPACKETCODE_INCTOKENS 2
#define INFIRDMA_START 4
#define INFIRDMA_ACK 8
#define INFIDIRECT_REQUEST 16
#define INFIPACKETCODE_INCTOKENSACK 32
#define INFIDUMMYPACKET 64

struct infiPacketHeader{
	char code;
	int nodeNo;
#if	CMK_IBVERBS_DEBUG
	int psn;
#endif	
};

/*
	Types of rdma packets
*/
#define INFI_MESG 1 
#define INFI_DIRECT 2

struct infiRdmaPacket{
	int fromNodeNo;
	int type;
	struct ibv_mr key;
	struct ibv_mr *keyPtr;
	int remoteSize;
	char *remoteBuf;
	void *localBuffer;
	OutgoingMsg ogm;
	struct infiRdmaPacket *next,*prev;
};


/** Represents a buffer that is used to receive messages
*/
#define BUFFER_RECV 1
#define BUFFER_RDMA 2
struct infiBuffer{
	int type;
	char *buf;
	int size;
	struct ibv_mr *key;
};



/** At the moment it is a simple pool with just a list of buffers
	* TODO; extend it to make it an element in a linklist of pools
*/
struct infiBufferPool{
	int numBuffers;
	struct infiBuffer *buffers;
	struct infiBufferPool *next;
};

/*****
	It is the structure for the send buffers that are used
	to send messages to other nodes
********/


typedef struct infiPacketStruct {	
	char *buf;
	int size;
	struct infiPacketHeader header;
	struct ibv_mr *keyHeader;
	struct OtherNodeStruct *destNode;
	struct infiPacketStruct *next;
	OutgoingMsg ogm;
	struct ibv_sge elemList[2];
	struct ibv_send_wr wr;
}* infiPacket;

/*
typedef struct infiBufferedWCStruct{
	struct ibv_wc wcList[WC_BUFFER_SIZE];
	int count;
	struct infiBufferedWCStruct *next,*prev;
} * infiBufferedWC;
*/

#define BCASTLIST_SIZE 50

struct infiBufferedBcastStruct{
	char *msg;
	int size;
	int broot;
	int asm_rank;
	int valid;
};

typedef struct infiBufferedBcastPoolStruct{
	struct infiBufferedBcastPoolStruct *next,*prev;
	struct infiBufferedBcastStruct bcastList[BCASTLIST_SIZE];
	int count;

} *infiBufferedBcastPool;




/***
	This structure represents the data needed by the infiniband
	communication routines of a node
	TODO: add locking for the smp version
*/
struct infiContext {
	struct ibv_context	*context;
	
	fd_set  asyncFds;
	struct timeval tmo;
	
	int ibPort;
//	struct ibv_comp_channel *channel;
	struct ibv_pd		*pd;
	struct ibv_cq		*sendCq;
	struct ibv_cq   *recvCq;
	struct ibv_srq  *srq;
	
	struct ibv_qp		**qp; //Array of qps (numNodes long) to temporarily store the queue pairs
												//It is used between CmiMachineInit and the call to node_addresses_store
												//when the qps are stored in the corresponding OtherNodes

	struct infiAddr *localAddr; //store the lid,qpn,msn address of ur qpair until they are sent

	infiPacket infiPacketFreeList; 
	
	struct infiBufferPool *recvBufferPool;

	struct infiPacketHeader header;

	int srqSize;
	int sendCqSize,recvCqSize;
	int tokensLeft;

	infiBufferedBcastPool bufferedBcastList;
	
	struct infiRdmaPacket *bufferedRdmaAcks;
	
	struct infiRdmaPacket *bufferedRdmaRequests;
/*	infiBufferedWC infiBufferedRecvList;*/
	
	int insideProcessBufferedBcasts;
};

static struct infiContext *context;

static inline infiPacket newPacket(){
	infiPacket pkt = malloc(sizeof(struct infiPacketStruct));
	pkt->size = -1;
	pkt->header = context->header;
	pkt->next = NULL;
	pkt->destNode = NULL;
	pkt->keyHeader = ibv_reg_mr(context->pd,&pkt->header,sizeof(struct infiPacketHeader),IBV_ACCESS_LOCAL_WRITE);
	pkt->ogm=NULL;
	assert(pkt->keyHeader!=NULL);
	
	pkt->elemList[0].addr = (uintptr_t)&(pkt->header);
	pkt->elemList[0].length = sizeof(struct infiPacketHeader);
	pkt->elemList[0].lkey = pkt->keyHeader->lkey;
	
	pkt->wr.wr_id = (uint64_t)pkt;
	pkt->wr.sg_list = &(pkt->elemList[0]);
	pkt->wr.num_sge = 2;
	pkt->wr.opcode = IBV_WR_SEND;
	pkt->wr.send_flags = IBV_SEND_SIGNALED;
	pkt->wr.next = NULL;
	
	return pkt;
};

#define FreeInfiPacket(pkt){ \
	pkt->size = -1;\
	pkt->ogm=NULL;\
	pkt->next = context->infiPacketFreeList; \
	context->infiPacketFreeList = pkt; \
}

#define MallocInfiPacket(pkt) { \
	infiPacket p = context->infiPacketFreeList; \
	if(p == NULL){ p = newPacket();} \
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
	int totalTokens;
	int tokensLeft;
	int nodeNo;
	
	int postedRecvs;
	int broot;//needed to store the root of a multi-packet broadcast sent along a spanning tree or hypercube
#if	CMK_IBVERBS_DEBUG	
	int psn;
#endif	
};


/********************************
Memory management structures and types
*****************/

struct infiCmiChunkHeaderStruct;

typedef struct infiCmiChunkMetaDataStruct {
	struct ibv_mr *key;
	int poolIdx;
	void *nextBuf;
	struct infiCmiChunkHeaderStruct *owner;
	int count;
} infiCmiChunkMetaData;


typedef struct infiCmiChunkHeaderStruct{
	infiCmiChunkMetaData *metaData;
	CmiChunkHeader chunkHeader;
} infiCmiChunkHeader;


#define METADATAFIELD(m) (((infiCmiChunkHeader *)m)[-1].metaData)

typedef struct {
	int size;//without infiCmiChunkHeader
	void *startBuf;
	int count;
} infiCmiChunkPool;

#define INFINUMPOOLS 20
#define INFIMAXPERPOOL 100
infiCmiChunkPool infiCmiChunkPools[INFINUMPOOLS];

static void initInfiCmiChunkPools();






/******************CmiMachineInit and its helper functions*/
static inline int pollSendCq(const int toBuffer);

void createLocalQps(struct ibv_device *dev,int ibPort, int myNode,int numNodes,struct infiAddr *localAddr);
static uint16_t getLocalLid(struct ibv_context *context, int port);
static int  checkQp(struct ibv_qp *qp){
	struct ibv_qp_attr attr;
	struct ibv_qp_init_attr init_attr;
		 
	ibv_query_qp(qp, &attr, IBV_QP_STATE | IBV_QP_CUR_STATE|IBV_QP_CAP  ,&init_attr);
	if(attr.cur_qp_state != IBV_QPS_RTS){
		MACHSTATE2(3,"CHECKQP failed cap wr %d sge %d",attr.cap.max_send_wr,attr.cap.max_send_sge);
		return 0;
	}
	return 1;
}
static void checkAllQps(){
	int i;
	for(i=0;i<_Cmi_numnodes;i++){
		if(i != _Cmi_mynode){
			if(!checkQp(nodes[i].infiData->qp)){
				pollSendCq(0);
				assert(0);
			}
		}
	}
}


static void CmiMachineInit(char **argv){
	struct ibv_device **devList;
	struct ibv_device *dev;
	int ibPort;
	int i;
	int calcMaxSize;
	infiPacket *pktPtrs;
	struct infiRdmaPacket **rdmaPktPtrs;

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
	
	MACHSTATE1(3,"context allocated %p",context);
	
	//localAddr will store the local addresses of all the qps
	context->localAddr = (struct infiAddr *)malloc(sizeof(struct infiAddr)*_Cmi_numnodes);
	
	MACHSTATE1(3,"context->localAddr allocated %p",context->localAddr);
	
	context->ibPort = ibPort;
	//the context for this infiniband device 
	context->context = ibv_open_device(dev);
	assert(context->context != NULL);
	
	MACHSTATE1(3,"device opened %p",context->context);

/*	FD_ZERO(&context->asyncFds);
	FD_SET(context->context->async_fd,&context->asyncFds);
	context->tmo.tv_sec=0;
	context->tmo.tv_usec=0;
	
	MACHSTATE(3,"asyncFds zeroed and set");*/

	//protection domain
	context->pd = ibv_alloc_pd(context->context);
	assert(context->pd != NULL);
	MACHSTATE2(3,"pd %p pd->handle %d",context->pd,context->pd->handle);
	
	context->header.nodeNo = _Cmi_mynode;

	mtu_size=1200;
	packetSize = mtu_size;
	dataSize = packetSize-sizeof(struct infiPacketHeader);
	
	calcMaxSize=5000;
	if(_Cmi_numnodes*50 > calcMaxSize){
		calcMaxSize = _Cmi_numnodes*50;
		if(calcMaxSize > 100000){
			calcMaxSize = 100000;
		}
	}
//	maxRecvBuffers=80;
	maxRecvBuffers=calcMaxSize;
	maxTokens = maxRecvBuffers;
//	maxTokens = 80;
	context->tokensLeft=maxTokens;
	//tokensPerProcessor=4;
	if(_Cmi_numnodes > 1){
		createLocalQps(dev,ibPort,_Cmi_mynode,_Cmi_numnodes,context->localAddr);
	}
		
	/*create the pool of send packets*/
	sendPacketPoolSize = maxTokens/2;	
	
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
	
	context->bufferedBcastList=NULL;
	context->bufferedRdmaAcks = NULL;
	context->bufferedRdmaRequests = NULL;
	context->insideProcessBufferedBcasts=0;
	
	//TURN ON RDMA
	rdma=1;
//	rdmaThreshold=32768;
	rdmaThreshold=22000;
	firstBinSize = 120;
	CmiAssert(rdmaThreshold > firstBinSize);
	blockAllocRatio=16;
	blockThreshold=8;
	
	initInfiCmiChunkPools();
	
	if(rdma){
		int numPkts;
		int k;
		if( _Cmi_numnodes*4 < maxRecvBuffers/4){
			numPkts = _Cmi_numnodes*4;
		}else{
			numPkts = maxRecvBuffers/4;
		}
		
		rdmaPktPtrs = (struct infiRdmaPacket **)malloc(numPkts*sizeof(struct infiRdmaPacket));
		for(k=0;k<numPkts;k++){
			rdmaPktPtrs[k] = CmiAlloc(sizeof(struct infiRdmaPacket));
		}
		
		for(k=0;k<numPkts;k++){
			CmiFree(rdmaPktPtrs[k]);
		}
		free(rdmaPktPtrs);
	}
	
/*	context->infiBufferedRecvList = NULL;*/
#if CMK_IBVERBS_STATS	
	regCount =0;
	regTime  = 0;

	pktCount=0;
	msgCount=0;

	processBufferedCount=0;
	processBufferedTime=0;
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
	
	MACHSTATE2(3,"myLid %d numNodes %d",myLid,numNodes);

	context->sendCqSize = maxTokens+2;
	context->sendCq = ibv_create_cq(context->context,context->sendCqSize,NULL,NULL,0);
	assert(context->sendCq != NULL);
	
	MACHSTATE1(3,"sendCq created %p",context->sendCq);
	
	
	context->recvCqSize = maxRecvBuffers;
	context->recvCq = ibv_create_cq(context->context,context->recvCqSize,NULL,NULL,0);
	
	MACHSTATE2(3,"recvCq created %p %d",context->recvCq,context->recvCqSize);
	assert(context->recvCq != NULL);
	
	//array of queue pairs

	context->qp = (struct ibv_qp **)malloc(sizeof(struct ibv_qp *)*numNodes);

	if(numNodes > 1)
	{
		context->srqSize = (maxRecvBuffers+2);
		struct ibv_srq_init_attr srqAttr = {
			.attr = {
			.max_wr  = context->srqSize,
			.max_sge = 1
			}
		};
		context->srq = ibv_create_srq(context->pd,&srqAttr);
		assert(context->srq != NULL);
	
		struct ibv_qp_init_attr initAttr = {
			.qp_type = IBV_QPT_RC,
			.send_cq = context->sendCq,
			.recv_cq = context->recvCq,
			.srq		 = context->srq,
			.sq_sig_all = 0,
			.qp_context = NULL,
			.cap     = {
				.max_send_wr  = maxTokens,
				.max_send_sge = 2,
			},
		};
		struct ibv_qp_attr attr;

		attr.qp_state        = IBV_QPS_INIT;
		attr.pkey_index      = 0;
		attr.port_num        = ibPort;
		attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

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

struct infiBufferPool * allocateInfiBufferPool(int numRecvs,int sizePerBuffer);
void postInitialRecvs(struct infiBufferPool *recvBufferPool,int numRecvs,int sizePerBuffer);

/* Initial the infiniband specific data for a remote node
	1. connect the qp and store it in and return it
**/
struct infiOtherNodeData *initInfiOtherNodeData(int node,int addr[3]){
	struct infiOtherNodeData * ret = malloc(sizeof(struct infiOtherNodeData));
	int err;
	ret->state = INFI_HEADER_DATA;
	ret->qp = context->qp[node];
//	ret->totalTokens = tokensPerProcessor;
//	ret->tokensLeft = tokensPerProcessor;
	ret->nodeNo = node;
//	ret->postedRecvs = tokensPerProcessor;
#if	CMK_IBVERBS_DEBUG	
	ret->psn = 0;
#endif
	
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

	MACHSTATE(3,"} initInfiOtherNodeData");
	return ret;
}


void 	infiPostInitialRecvs(){
	//create the pool and post the receives
	int numPosts;
/*	if(tokensPerProcessor*(_Cmi_numnodes-1) <= maxRecvBuffers){
		numPosts = tokensPerProcessor*(_Cmi_numnodes-1);
	}else{
		numPosts = maxRecvBuffers;
	}*/

	if(_Cmi_numnodes > 1){
		numPosts = maxRecvBuffers;
	}else{
		numPosts = 0;
	}
	if(numPosts > 0){
		context->recvBufferPool = allocateInfiBufferPool(numPosts,packetSize);
		postInitialRecvs(context->recvBufferPool,numPosts,packetSize);
	}


	free(context->qp);
	context->qp = NULL;
	free(context->localAddr);
	context->localAddr= NULL;
}

struct infiBufferPool * allocateInfiBufferPool(int numRecvs,int sizePerBuffer){
	int numBuffers;
	int i;
	int bigSize;
	char *bigBuf;
	struct infiBufferPool *ret;
	struct ibv_mr *bigKey;

	MACHSTATE2(3,"allocateInfiBufferPool numRecvs %d sizePerBuffer%d ",numRecvs,sizePerBuffer);

	page_size = sysconf(_SC_PAGESIZE);
	ret = malloc(sizeof(struct infiBufferPool));
	ret->next = NULL;
	numBuffers=ret->numBuffers = numRecvs;
	
	ret->buffers = malloc(sizeof(struct infiBuffer)*numBuffers);
	
	bigSize = numBuffers*sizePerBuffer;
	bigBuf=malloc(bigSize);
	bigKey = ibv_reg_mr(context->pd,bigBuf,bigSize,IBV_ACCESS_LOCAL_WRITE);
	assert(bigKey != NULL);
	
	for(i=0;i<numBuffers;i++){
		struct infiBuffer *buffer =  &(ret->buffers[i]);
		buffer->type = BUFFER_RECV;
		buffer->size = sizePerBuffer;
		
		/*buffer->buf = malloc(sizePerBuffer);
		buffer->key = ibv_reg_mr(context->pd,buffer->buf,buffer->size,IBV_ACCESS_LOCAL_WRITE);*/
		buffer->buf = &bigBuf[i*sizePerBuffer];
		buffer->key = bigKey;

		if(buffer->key == NULL){
			MACHSTATE2(3,"i %d buffer->buf %p",i,buffer->buf);
			CmiAssert(buffer->key != NULL);
		}
	}
	return ret;
};



/**
	 Post the buffers as recv work requests
*/
void postInitialRecvs(struct infiBufferPool *recvBufferPool,int numRecvs,int sizePerBuffer){
	int j,err;
	struct ibv_recv_wr *workRequests = malloc(sizeof(struct ibv_recv_wr)*numRecvs);
	struct ibv_sge *sgElements = malloc(sizeof(struct ibv_sge)*numRecvs);
	struct ibv_recv_wr *bad_wr;
	
	int startBufferIdx=0;
	MACHSTATE2(3,"posting %d receives of size %d",numRecvs,sizePerBuffer);
	for(j=0;j<numRecvs;j++){
		
		
		sgElements[j].addr = (uint64_t) recvBufferPool->buffers[startBufferIdx+j].buf;
		sgElements[j].length = sizePerBuffer;
		sgElements[j].lkey = recvBufferPool->buffers[startBufferIdx+j].key->lkey;
		
		workRequests[j].wr_id = (uint64_t)&(recvBufferPool->buffers[startBufferIdx+j]);
		workRequests[j].sg_list = &sgElements[j];
		workRequests[j].num_sge = 1;
		if(j != numRecvs-1){
			workRequests[j].next = &workRequests[j+1];
		}
		
	}
	workRequests[numRecvs-1].next = NULL;
	MACHSTATE(3,"About to call ibv_post_srq_recv");
	if(ibv_post_srq_recv(context->srq,workRequests,&bad_wr)){
		assert(0);
	}

	free(workRequests);
	free(sgElements);
}




static inline void CommunicationServer_nolock(int toBuffer); //if buffer ==1 recvd messages are buffered but not processed

static void CmiMachineExit()
{
#if CMK_IBVERBS_STATS	
	printf("[%d] msgCount %d pktCount %d packetSize %d total Time %.6lf s processBufferedCount %d processBufferedTime %.6lf s \n",_Cmi_mynode,msgCount,pktCount,packetSize,CmiTimer(),processBufferedCount,processBufferedTime);
#endif
}

static void CmiNotifyStillIdle(CmiIdleState *s) {
	CommunicationServer_nolock(0);
}

static inline void increaseTokens(OtherNode node);

static inline int pollRecvCq(const int toBuffer);
static inline int pollSendCq(const int toBuffer);


static inline void getFreeTokens(struct infiOtherNodeData *infiData){
#if !CMK_IBVERBS_TOKENS_FLOW
	return;
#else
	//if(infiData->tokensLeft == 0){
	if(context->tokensLeft == 0){
		MACHSTATE(3,"GET FREE TOKENS {{{");
	}else{
		return;
	}
	while(context->tokensLeft == 0){
		CommunicationServer_nolock(1); 
	}
	MACHSTATE1(3,"}}} GET FREE TOKENS %d",context->tokensLeft);
#endif
}


/**
	Packetize this data and send it
**/



static void inline EnqueuePacket(OtherNode node,infiPacket packet,int size,struct ibv_mr *dataKey){
	int incTokens=0;
	int retval;
#if CMK_IBVERBS_STATS
	double _startRegTime =CmiWallTimer();
#endif	

#if CMK_IBVERBS_STATS
		_startRegTime = CmiWallTimer();
#endif
#if	CMK_IBVERBS_DEBUG
	packet->header.psn = (++node->infiData->psn);
#endif	



	packet->elemList[1].addr = (uintptr_t)packet->buf;
	packet->elemList[1].length = size;
	packet->elemList[1].lkey = dataKey->lkey;
	
	
	
#if CMK_IBVERBS_STATS
			regCount++;
			regTime += CmiWallTimer()-_startRegTime;
#endif
	packet->destNode = node;
	
#if CMK_IBVERBS_STATS	
	pktCount++;
#endif	
	
	getFreeTokens(node->infiData);

#if CMK_IBVERBS_INCTOKENS	
	if((node->infiData->tokensLeft < INCTOKENS_FRACTION*node->infiData->totalTokens || node->infiData->tokensLeft < 2) && node->infiData->totalTokens < maxTokens){
		packet->header.code |= INFIPACKETCODE_INCTOKENS;
		incTokens=1;
	}
#endif
/*
	if(!checkQp(node->infiData->qp)){
		pollSendCq(1);
		assert(0);
	}*/

	struct ibv_send_wr *bad_wr=NULL;
	if(retval = ibv_post_send(node->infiData->qp,&(packet->wr),&bad_wr)){
		CmiPrintf("[%d] Sending to node %d failed with return value %d\n",_Cmi_mynode,node->infiData->nodeNo,retval);
		assert(0);
	}
#if	CMK_IBVERBS_TOKENS_FLOW
	context->tokensLeft--;
#endif

/*	if(!checkQp(node->infiData->qp)){
		pollSendCq(1);
		assert(0);
	}*/

#if CMK_IBVERBS_INCTOKENS	
	if(incTokens){
		increaseTokens(node);
	}
#endif


#if	CMK_IBVERBS_DEBUG
	MACHSTATE4(3,"Packet send size %d node %d tokensLeft %d psn %d",size,packet->destNode->infiData->nodeNo,context->tokensLeft,packet->header.psn);
#else
	MACHSTATE4(3,"Packet send size %d node %d tokensLeft %d packet->buf %p",size,packet->destNode->infiData->nodeNo,context->tokensLeft,packet->buf);
#endif

};


static void inline EnqueueDummyPacket(OtherNode node,int size){
	infiPacket packet;
	MallocInfiPacket(packet);
	packet->size = size;
	packet->buf = CmiAlloc(size);
	
	packet->header.code = INFIDUMMYPACKET;

	struct ibv_mr *key = METADATAFIELD(packet->buf)->key;
	
	MACHSTATE2(3,"Dummy packet to %d size %d",node->infiData->nodeNo,size);
	EnqueuePacket(node,packet,size,key);
}






static void inline EnqueueDataPacket(OutgoingMsg ogm, OtherNode node, int rank,char *data,int size,int broot,int copy){
	infiPacket packet;
	MallocInfiPacket(packet);
	packet->size = size;
	packet->buf=data;
	
	//the nodeNo is added at time of packet allocation
	packet->header.code = INFIPACKETCODE_DATA;
	
	ogm->refcount++;
	packet->ogm = ogm;
	
	struct ibv_mr *key = METADATAFIELD(ogm->data)->key;
	
	EnqueuePacket(node,packet,size,key);
};

static inline void EnqueueRdmaPacket(OutgoingMsg ogm, OtherNode node);
static inline void processAllBufferedMsgs();

void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy){
	int size; char *data;
//	processAllBufferedMsgs();

	
	ogm->refcount++;
  size = ogm->size;
  data = ogm->data;

#if CMK_IBVERBS_STATS	
	msgCount++;
#endif

	MACHSTATE3(3,"Sending ogm %p of size %d to %d",ogm,size,node->infiData->nodeNo);
	//First packet has dgram header, other packets dont
	
  DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, 1, broot);
	
	CmiMsgHeaderSetLength(ogm->data,ogm->size);

	if(rdma && size > rdmaThreshold){
			EnqueueRdmaPacket(ogm,node);
	}else{
	
		while(size > dataSize){
			EnqueueDataPacket(ogm,node,rank,data,dataSize,broot,copy);
			size -= dataSize;
			data += dataSize;
		}
		if(size > 0){
			EnqueueDataPacket(ogm,node,rank,data,size,broot,copy);
		}
	}
	processAllBufferedMsgs();
	ogm->refcount--;
	MACHSTATE3(3,"DONE Sending ogm %p of size %d to %d",ogm,ogm->size,node->infiData->nodeNo);
}


static inline void EnqueueRdmaPacket(OutgoingMsg ogm, OtherNode node){
	infiPacket packet;

	ogm->refcount++;
	
	MallocInfiPacket(packet);
 
 {
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)CmiAlloc(sizeof(struct infiRdmaPacket));

		
		packet->size = sizeof(struct infiRdmaPacket);
		packet->buf = (char *)rdmaPacket;
		
/*		struct ibv_mr *key = ibv_reg_mr(context->pd,ogm->data,ogm->size,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);*/
		struct ibv_mr *key = METADATAFIELD(ogm->data)->key;
		MACHSTATE3(3,"ogm->data %p metadata %p key %p",ogm->data,METADATAFIELD(ogm->data),key);
		
		packet->header.code = INFIRDMA_START;
		packet->header.nodeNo = _Cmi_mynode;
		packet->ogm = NULL;

		rdmaPacket->type = INFI_MESG;
		rdmaPacket->ogm = ogm;
		rdmaPacket->key = *key;
		rdmaPacket->keyPtr = key;
		rdmaPacket->remoteBuf = ogm->data;
		rdmaPacket->remoteSize = ogm->size;
		
		
		struct ibv_mr *packetKey = METADATAFIELD((void *)rdmaPacket)->key;
		
		MACHSTATE3(3,"rdmaRequest being sent to node %d buf %p size %d",node->infiData->nodeNo,ogm->data,ogm->size);
		EnqueuePacket(node,packet,sizeof(struct infiRdmaPacket),packetKey);
	}
}



static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer);
static inline void processSendWC(struct ibv_wc *sendWC);
static unsigned int _count=0;
extern int errno;
static int _countAsync=0;
static inline void processAsyncEvents(){
	struct ibv_async_event event;
	int ready;
	_countAsync++;
	if(_countAsync < 1){
		return;
	}
	_countAsync=0;
	FD_SET(context->context->async_fd,&context->asyncFds);
	assert(FD_ISSET(context->context->async_fd,&context->asyncFds));
	ready = select(1, &context->asyncFds,NULL,NULL,&context->tmo);
	
	if(ready==0){
		return;
	}
	if(ready == -1){
//		printf("[%d] strerror %s \n",_Cmi_mynode,strerror(errno));
		return;
	}
	
	if (ibv_get_async_event(context->context, &event)){
		return;
		CmiAbort("get async event failed");
	}
	printf("[%d] async event %d \n",_Cmi_mynode, event.event_type);
	ibv_ack_async_event(&event);

	
}

static inline  void CommunicationServer_nolock(int toBuffer) {
	int processed;
	if(_Cmi_numnodes <= 1){
		return;
	}
	MACHSTATE(2,"CommServer_nolock{");
	
//	processAsyncEvents();
	
//	checkAllQps();

	processed = pollRecvCq(toBuffer);
	

	processed += pollSendCq(toBuffer);
	
	if(toBuffer == 0){
		if(processed != 0)
			processAllBufferedMsgs();
	}
	
//	checkAllQps();
//	_count--;
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

static inline int pollRecvCq(const int toBuffer){
	int i;
	int ne;
	struct ibv_wc wc[WC_LIST_SIZE];
	
	MACHSTATE1(2,"pollRecvCq %d (((",toBuffer);
	ne = ibv_poll_cq(context->recvCq,WC_LIST_SIZE,&wc[0]);
//	assert(ne >=0);
	
	if(ne != 0){
		MACHSTATE1(3,"pollRecvCq ne %d",ne);
	}
	
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
	MACHSTATE1(2,"))) pollRecvCq %d",toBuffer);
	return ne;

}

static inline  void processRdmaWC(struct ibv_wc *rdmaWC,const int toBuffer);

static inline int pollSendCq(const int toBuffer){
	int i;
	int ne;
	struct ibv_wc wc[WC_LIST_SIZE];

	ne = ibv_poll_cq(context->sendCq,WC_LIST_SIZE,&wc[0]);
//	assert(ne >=0);
	
	
	for(i=0;i<ne;i++){
		if(wc[i].status != IBV_WC_SUCCESS){
			infiPacket _packet = (infiPacket )wc[i].wr_id;
			printf("[%d] wc[%d] status %d wc[i].opcode %d destnode %d \n",_Cmi_mynode,i,wc[i].status,wc[i].opcode,_packet->destNode->infiData->nodeNo);
			assert(0);
		}
		switch(wc[i].opcode){
			case IBV_WC_SEND:{
				//message received
				processSendWC(&wc[i]);
				
				break;
				}
			case IBV_WC_RDMA_READ:
			{
//				processRdmaWC(&wc[i],toBuffer);
					processRdmaWC(&wc[i],1);
				break;
			}
			default:
				CmiAbort("Wrong type of work completion object in recvq");
				break;
		}
			
	}
	return ne;
}

static void CommunicationServer(int sleepTime, int where){
	if( where == COMM_SERVER_FROM_INTERRUPT){
		return;
	}
	CommunicationServer_nolock(0);
}


static void insertBufferedBcast(char *msg,int size,int broot,int asm_rank);


void static inline handoverMessage(char *newmsg,int total_size,int rank,int broot,int toBuffer);

static inline void processMessage(int nodeNo,int len,char *msg,const int toBuffer){
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
			size = CmiMsgHeaderGetLength(msg);
			MACHSTATE2(3,"START of a new message from node %d of total size %d",nodeNo,size);
//			CmiAssert(size > 0);
//			CmiAssert(nodes_by_pe[srcpe] == node);
			
//			CmiAssert(newmsg == NULL);
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
			if(node->asm_fill + len < node->asm_total && len != dataSize){
				CmiPrintf("from node %d asm_total: %d, asm_fill: %d, len:%d.\n",node->infiData->nodeNo, node->asm_total, node->asm_fill, len);
				CmiAbort("packet in the middle does not have expected length");
			}
			if(node->asm_fill+len > node->asm_total){
				CmiPrintf("asm_total: %d, asm_fill: %d, len:%d.\n", node->asm_total, node->asm_fill, len);
				CmiAbort("\n\n\t\tLength mismatch!!\n\n");
			}
			//tODO: remove this
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
		handoverMessage(newmsg,total_size,node->asm_rank,node->infiData->broot,1);
		MACHSTATE3(3,"Message from node %d of length %d completely received msg %p",nodeNo,total_size,newmsg);
	}
	
};

void static inline handoverMessage(char *newmsg,int total_size,int rank,int broot,int toBuffer){
#if CMK_BROADCAST_SPANNING_TREE
        if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         ){
					 if(toBuffer){
						 	insertBufferedBcast(newmsg,total_size,broot,rank);
					 	}else{
          		SendSpanningChildren(NULL, 0, total_size, newmsg,broot,rank);
						}
					}
#elif CMK_BROADCAST_HYPERCUBE
        if (rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || rank == DGRAM_NODEBROADCAST
#endif
         ){
					 if(toBuffer){
						 	insertBufferedBcast(newmsg,total_size,broot,rank);
					 }else{
          		SendHypercube(NULL, 0, total_size, newmsg,broot,rank);
						}
					}
#endif


		
		switch (rank) {
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
					
          CmiPushPE(rank, newmsg);
				}
  	}    /* end of switch */
		if(!toBuffer){
		processAllBufferedMsgs();
		}
}


static inline void increasePostedRecvs(int nodeNo);
static inline void processRdmaRequest(struct infiRdmaPacket *rdmaPacket,int fromNodeNo,int isBuffered);
static inline void processRdmaAck(struct infiRdmaPacket *rdmaPacket);

struct infiDirectRequestPacket;
static inline void processDirectRequest(struct infiDirectRequestPacket *directRequestPacket);

static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer){
	struct infiBuffer *buffer = (struct infiBuffer *) recvWC->wr_id;	
	struct infiPacketHeader *header = (struct infiPacketHeader *)buffer->buf;
	int nodeNo = header->nodeNo;
		
	int len = recvWC->byte_len-sizeof(struct infiPacketHeader);
#if	CMK_IBVERBS_DEBUG
	MACHSTATE3(3,"packet from node %d len %d psn %d",nodeNo,len,header->psn);
#else
	MACHSTATE2(3,"packet from node %d len %d",nodeNo,len);	
#endif
	
	if(header->code & INFIPACKETCODE_DATA){
			
			processMessage(nodeNo,len,(buffer->buf+sizeof(struct infiPacketHeader)),toBuffer);
	}
	if(header->code & INFIDUMMYPACKET){
		MACHSTATE(3,"Dummy packet");
	}
#if CMK_IBVERBS_INCTOKENS	
	if(header->code & INFIPACKETCODE_INCTOKENS){
		increasePostedRecvs(nodeNo);
	}
#endif	
	if(rdma && header->code & INFIRDMA_START){
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)(buffer->buf+sizeof(struct infiPacketHeader));
//		if(toBuffer){
			//TODO: make a function of this and use for both acks and requests
			struct infiRdmaPacket *copyPacket = malloc(sizeof(struct infiRdmaPacket));
			struct infiRdmaPacket *tmp=context->bufferedRdmaRequests;
			*copyPacket = *rdmaPacket;
			copyPacket->fromNodeNo = nodeNo;
			MACHSTATE1(3,"Buffering Rdma Request %p",copyPacket);
			context->bufferedRdmaRequests = copyPacket;
			copyPacket->next = tmp;
			copyPacket->prev = NULL;
			if(tmp != NULL){
				tmp->prev = copyPacket;
			}
/*		}else{
			processRdmaRequest(rdmaPacket,nodeNo,0);
		}*/
	}
	if(rdma && header->code & INFIRDMA_ACK){
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)(buffer->buf+sizeof(struct infiPacketHeader)) ;
		processRdmaAck(rdmaPacket);
	}
	if(header->code & INFIDIRECT_REQUEST){
		struct infiDirectRequestPacket *directRequestPacket = (struct infiDirectRequestPacket *)(buffer->buf+sizeof(struct infiPacketHeader));
		processDirectRequest(directRequestPacket);
	}
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
	
		if(ibv_post_srq_recv(context->srq,&wr,&bad_wr)){
			assert(0);
		}
	}

};




static inline  void processSendWC(struct ibv_wc *sendWC){

	infiPacket packet = (infiPacket )sendWC->wr_id;
#if CMK_IBVERBS_TOKENS_FLOW
//	packet->destNode->infiData->tokensLeft++;
	context->tokensLeft++;
#endif

	MACHSTATE2(3,"Packet send complete node %d  tokensLeft %d",packet->destNode->infiData->nodeNo,context->tokensLeft);
	if(packet->ogm != NULL){
		packet->ogm->refcount--;
		if(packet->ogm->refcount == 0){
			GarbageCollectMsg(packet->ogm);	
		}
	}else{
		if(packet->header.code == INFIRDMA_START || packet->header.code == INFIRDMA_ACK || packet->header.code ==  INFIDUMMYPACKET){
			CmiFree(packet->buf);
		}
	}

	FreeInfiPacket(packet);
};



/********************************************************************/
static inline void processRdmaRequest(struct infiRdmaPacket *_rdmaPacket,int fromNodeNo,int isBuffered){
	int nodeNo = fromNodeNo;
	OtherNode node = &nodes[nodeNo];
	struct infiRdmaPacket *rdmaPacket;

	getFreeTokens(node->infiData);
#if CMK_IBVERBS_TOKENS_FLOW
//	node->infiData->tokensLeft--;
	context->tokensLeft--;
#endif
	
	struct infiBuffer *buffer = malloc(sizeof(struct infiBuffer));
//	CmiAssert(buffer != NULL);
	
	
	if(isBuffered){
		rdmaPacket = _rdmaPacket;
	}else{
		rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
		*rdmaPacket = *_rdmaPacket;
	}


	rdmaPacket->fromNodeNo = fromNodeNo;
	rdmaPacket->localBuffer = (void *)buffer;
	
	buffer->type = BUFFER_RDMA;
	buffer->size = rdmaPacket->remoteSize;
	
	buffer->buf  = (char *)CmiAlloc(rdmaPacket->remoteSize);
//	CmiAssert(buffer->buf != NULL);

	buffer->key = METADATAFIELD(buffer->buf)->key;

	
	MACHSTATE3(3,"received rdma request from node %d for remoteBuffer %p keyPtr %p",nodeNo,rdmaPacket->remoteBuf,rdmaPacket->keyPtr);
	MACHSTATE3(3,"Local buffer->buf %p buffer->key %p rdmaPacket %p",buffer->buf,buffer->key,rdmaPacket);
//	CmiAssert(buffer->key != NULL);
	
	{
		struct ibv_sge list = {
			.addr = (uintptr_t )buffer->buf,
			.length = buffer->size,
			.lkey 	= buffer->key->lkey
		};

		struct ibv_send_wr *bad_wr;
		struct ibv_send_wr wr = {
			.wr_id = (uint64_t )rdmaPacket,
			.sg_list = &list,
			.num_sge = 1,
			.opcode = IBV_WR_RDMA_READ,
			.send_flags = IBV_SEND_SIGNALED,
			.wr.rdma = {
				.remote_addr = (uint64_t )rdmaPacket->remoteBuf,
				.rkey = rdmaPacket->key.rkey
			}
		};
		/** post and rdma_read that is a rdma get*/
		if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
			assert(0);
		}
	}

};

static inline void EnqueueRdmaAck(struct infiRdmaPacket *rdmaPacket);
static inline void processDirectWC(struct infiRdmaPacket *rdmaPacket);

static inline  void processRdmaWC(struct ibv_wc *rdmaWC,const int toBuffer){
		//rdma get done
#if CMK_IBVERBS_STATS
	double _startRegTime;
#endif	

	struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *) rdmaWC->wr_id;
	if(rdmaPacket->type == INFI_DIRECT){
		processDirectWC(rdmaPacket);
		return;
	}
//	CmiAssert(rdmaPacket->type == INFI_MESG);
	struct infiBuffer *buffer = (struct infiBuffer *)rdmaPacket->localBuffer;

	/*TODO: remove this
	memcpy(buffer->buf,rdmaInBuf,rdmaWC->byte_len);*/
	
/*	CmiAssert(buffer->type == BUFFER_RDMA);
	CmiAssert(rdmaWC->byte_len == buffer->size);*/
	
	{
		int size;
		int rank, srcpe, seqno, magic, i;
		unsigned int broot;
		char *msg = buffer->buf;
		DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot);
		size = CmiMsgHeaderGetLength(msg);
/*		CmiAssert(size == buffer->size);*/
		handoverMessage(buffer->buf,size,rank,broot,toBuffer);
	}
	MACHSTATE2(3,"Rdma done for buffer->buf %p buffer->key %p",buffer->buf,buffer->key);

	
	free(buffer);
	
	OtherNode node=&nodes[rdmaPacket->fromNodeNo];
	//we are sending this ack as a response to a successful
	// rdma_Read.. the token for that rdma_Read needs to be freed
#if CMK_IBVERBS_TOKENS_FLOW	
	//node->infiData->tokensLeft++;
	context->tokensLeft++;
#endif

	//send ack to sender if toBuffer is off otherwise buffer it
	if(toBuffer){
		MACHSTATE1(3,"Buffering Rdma Ack %p",rdmaPacket);
		struct infiRdmaPacket *tmp = context->bufferedRdmaAcks;
		context->bufferedRdmaAcks = rdmaPacket;
		rdmaPacket->next = tmp;
		rdmaPacket->prev = NULL;
		if(tmp != NULL){
			tmp->prev = rdmaPacket;	
		}
	}else{
		EnqueueRdmaAck(rdmaPacket);		
		free(rdmaPacket);
	}
}

static inline void EnqueueRdmaAck(struct infiRdmaPacket *rdmaPacket){
	infiPacket packet;
	OtherNode node=&nodes[rdmaPacket->fromNodeNo];

	
	MallocInfiPacket(packet);
	{
		struct infiRdmaPacket *ackPacket = (struct infiRdmaPacket *)CmiAlloc(sizeof(struct infiRdmaPacket));
		*ackPacket = *rdmaPacket;
		packet->size = sizeof(struct infiRdmaPacket);
		packet->buf = (char *)ackPacket;
		packet->header.code = INFIRDMA_ACK;
		packet->ogm=NULL;
		
		struct ibv_mr *packetKey = METADATAFIELD((void *)ackPacket)->key;
	

		EnqueuePacket(node,packet,sizeof(struct infiRdmaPacket),packetKey);
	}
};


static inline void processRdmaAck(struct infiRdmaPacket *rdmaPacket){
	MACHSTATE2(3,"rdma ack received for remoteBuf %p size %d",rdmaPacket->remoteBuf,rdmaPacket->remoteSize);
	rdmaPacket->ogm->refcount--;
	GarbageCollectMsg(rdmaPacket->ogm);
}


/****************************
 Deal with all the buffered (delayed) messages
 such as processing recvd broadcasts, sending
 rdma acks and processing recvd rdma requests
******************************/


static inline infiBufferedBcastPool createBcastPool(){
	int i;
	infiBufferedBcastPool ret = malloc(sizeof(struct infiBufferedBcastPoolStruct));
	ret->count = 0;
	ret->next = ret->prev = NULL;	
	for(i=0;i<BCASTLIST_SIZE;i++){
		ret->bcastList[i].valid = 0;
	}
	return ret;
};
/****
	The buffered bcast messages are stored in a doubly linked list of 
	arrays or blocks.
	To keep the average insert cost low, a new block is added 
	to the top of the list. (resulting in a reverse seq of blocks)
	Within a block however bcast are stored in increasing order sequence
*****/

static void insertBufferedBcast(char *msg,int size,int broot,int asm_rank){
	if(context->bufferedBcastList == NULL){
		context->bufferedBcastList = createBcastPool();
	}else{
		if(context->bufferedBcastList->count == BCASTLIST_SIZE){
			infiBufferedBcastPool tmp;
			tmp = createBcastPool();
			context->bufferedBcastList->prev = tmp;
			tmp->next = context->bufferedBcastList;
			context->bufferedBcastList = tmp;
		}
	}
	context->bufferedBcastList->bcastList[context->bufferedBcastList->count].msg = msg;
	context->bufferedBcastList->bcastList[context->bufferedBcastList->count].size = size;
	context->bufferedBcastList->bcastList[context->bufferedBcastList->count].broot = broot;
	context->bufferedBcastList->bcastList[context->bufferedBcastList->count].asm_rank = asm_rank;
	context->bufferedBcastList->bcastList[context->bufferedBcastList->count].valid = 1;
	
	MACHSTATE3(3,"Broadcast msg %p of size %d being buffered at count %d ",msg,size,context->bufferedBcastList->count);
	
	context->bufferedBcastList->count++;
}

/*********
	Go through the blocks of buffered bcast messages. process last block first
	processign within a block is in sequence though
*********/
static inline void processBufferedBcast(){
	infiBufferedBcastPool start;

	if(context->bufferedBcastList == NULL){
		return;
	}
	start = context->bufferedBcastList;
	if(context->insideProcessBufferedBcasts==1){
		return;
	}
	context->insideProcessBufferedBcasts=1;

	while(start->next != NULL){
		start = start->next;
	}
	
	while(start != NULL){
		int i=0;
		infiBufferedBcastPool tmp;
		if(start->count != 0){
			MACHSTATE2(3,"start %p start->count %d[[[",start,start->count);
		}
		for(i=0;i<start->count;i++){
			if(start->bcastList[i].valid == 0){
				continue;
			}
			start->bcastList[i].valid=0;
			MACHSTATE3(3,"Buffered broadcast msg %p of size %d being processed at %d",start->bcastList[i].msg,start->bcastList[i].size,i);
#if CMK_BROADCAST_SPANNING_TREE
        if (start->bcastList[i].asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || start->bcastList[i].asm_rank == DGRAM_NODEBROADCAST
#endif
         ){
          	SendSpanningChildren(NULL, 0, start->bcastList[i].size,start->bcastList[i].msg, start->bcastList[i].broot,start->bcastList[i].asm_rank);
					}
#elif CMK_BROADCAST_HYPERCUBE
        if (start->bcastList[i].asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || start->bcastList[i].asm_rank == DGRAM_NODEBROADCAST
#endif
         ){
          	SendHypercube(NULL, 0,start->bcastList[i].size,start->bcastList[i].msg ,start->bcastList[i].broot,start->bcastList[i].asm_rank);
					}
#endif
		}
		if(start->count != 0){
			MACHSTATE2(3,"]]] start %p start->count %d",start,start->count);
		}
		
		tmp = start;
		start = start->prev;
		free(tmp);
		if(start != NULL){
			//not the first one
			start->next = NULL;
		}
	}

	context->bufferedBcastList = NULL;
/*	context->bufferedBcastList->prev = NULL;
	context->bufferedBcastList->count =0;	*/
	context->insideProcessBufferedBcasts=0;
	MACHSTATE(2,"processBufferedBcast done ");
};


static inline void processBufferedRdmaAcks(){
	struct infiRdmaPacket *start = context->bufferedRdmaAcks;
	if(start == NULL){
		return;
	}
	while(start->next != NULL){
		start = start->next;
	}
	while(start != NULL){
		struct infiRdmaPacket *rdmaPacket=start;
		MACHSTATE1(3,"Processing Buffered Rdma Ack %p",rdmaPacket);
		EnqueueRdmaAck(rdmaPacket);
		start = start->prev;
		free(rdmaPacket);
	}
	context->bufferedRdmaAcks=NULL;
}



static inline void processBufferedRdmaRequests(){
	struct infiRdmaPacket *start = context->bufferedRdmaRequests;
	if(start == NULL){
		return;
	}
	
	
	while(start->next != NULL){
		start = start->next;
	}
	while(start != NULL){
		struct infiRdmaPacket *rdmaPacket=start;
		MACHSTATE1(3,"Processing Buffered Rdma Request %p",rdmaPacket);
		processRdmaRequest(rdmaPacket,rdmaPacket->fromNodeNo,1);
		start = start->prev;
	}
	
	context->bufferedRdmaRequests=NULL;
}





static inline void processAllBufferedMsgs(){
#if CMK_IBVERBS_STATS
	double _startTime = CmiWallTimer();
	processBufferedCount++;
#endif
	processBufferedBcast();

	processBufferedRdmaAcks();
	processBufferedRdmaRequests();
#if CMK_IBVERBS_STATS
	processBufferedTime += (CmiWallTimer()-_startTime);
#endif	
};

















/*************************
	Increase tokens when short of them
**********/
static inline void increaseTokens(OtherNode node){
	int err;
	int increase = node->infiData->totalTokens*INCTOKENS_INCREASE;
	if(node->infiData->totalTokens + increase > maxTokens){
		increase = maxTokens-node->infiData->totalTokens;
	}
	node->infiData->totalTokens += increase;
	node->infiData->tokensLeft += increase;
	MACHSTATE3(3,"Increasing tokens for node %d to %d by %d",node->infiData->nodeNo,node->infiData->totalTokens,increase);
	//increase the size of the sendCq
	int currentCqSize = context->sendCqSize;
	if(ibv_resize_cq(context->sendCq,currentCqSize+increase)){
		fprintf(stderr,"[%d] failed to increase cq by %d from %d totalTokens %d \n",_Cmi_mynode,increase,currentCqSize, node->infiData->totalTokens);
		assert(0);
	}
	context->sendCqSize+= increase;
};



static void increasePostedRecvs(int nodeNo){
	OtherNode node = &nodes[nodeNo];
	int tokenIncrease = node->infiData->postedRecvs*INCTOKENS_INCREASE;	
	int recvIncrease = tokenIncrease;
	if(tokenIncrease+node->infiData->postedRecvs > maxTokens){
		tokenIncrease = maxTokens - node->infiData->postedRecvs;
	}
	if(tokenIncrease+context->srqSize > maxRecvBuffers){
		recvIncrease = maxRecvBuffers-context->srqSize;
	}
	node->infiData->postedRecvs+= recvIncrease;
	context->srqSize += recvIncrease;
	MACHSTATE3(3,"Increase tokens by %d to %d for node %d ",tokenIncrease,node->infiData->postedRecvs,nodeNo);
	//increase the size of the recvCq
	int currentCqSize = context->recvCqSize;
	if(ibv_resize_cq(context->recvCq,currentCqSize+tokenIncrease)){
		assert(0);
	}
	context->recvCqSize += tokenIncrease;
	if(recvIncrease > 0){
		//create another bufferPool and attach it to the top of the current one
		struct infiBufferPool *newPool = allocateInfiBufferPool(recvIncrease,packetSize);
		newPool->next = context->recvBufferPool;
		context->recvBufferPool = newPool;
		postInitialRecvs(newPool,recvIncrease,packetSize);
	}

};




/*********************************************
	Memory management routines for RDMA

************************************************/

/**
	There are INFINUMPOOLS of memory.
	The first pool is of size firstBinSize.
	The ith pool is of size firstBinSize*2^i
*/

static void initInfiCmiChunkPools(){
	int i;
	int size = firstBinSize;
	
	for(i=0;i<INFINUMPOOLS;i++){
		infiCmiChunkPools[i].size = size;
		infiCmiChunkPools[i].startBuf = NULL;
		infiCmiChunkPools[i].count = 0;
		size *= 2;
	}
}

static inline void *getInfiCmiChunk(int dataSize){
	//find out to which pool this dataSize belongs to
	// poolIdx = floor(log2(dataSize/firstBinSize))+1
	int ratio = dataSize/firstBinSize;
	int poolIdx=0;
	void *res;
	
	while(ratio > 0){
		ratio  = ratio >> 1;
		poolIdx++;
	}
	MACHSTATE2(3,"getInfiCmiChunk for size %d in poolIdx %d",dataSize,poolIdx);
	if((poolIdx < INFINUMPOOLS && infiCmiChunkPools[poolIdx].startBuf == NULL) || poolIdx >= INFINUMPOOLS){
		infiCmiChunkMetaData *metaData;		
		infiCmiChunkHeader *hdr;
		int allocSize;
		int count=1;
		int i;
		struct ibv_mr *key;
		void *origres;
		
		
		if(poolIdx < INFINUMPOOLS ){
			allocSize = infiCmiChunkPools[poolIdx].size;
		}else{
			allocSize = dataSize;
		}

		if(poolIdx < blockThreshold){
			count = blockAllocRatio;
		}
		res = malloc((allocSize+sizeof(infiCmiChunkHeader))*count);
		hdr = res;
		
		key = ibv_reg_mr(context->pd,res,(allocSize+sizeof(infiCmiChunkHeader))*count,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
		assert(key != NULL);
		
		origres = (res += sizeof(infiCmiChunkHeader));

		for(i=0;i<count;i++){
			metaData = METADATAFIELD(res) = malloc(sizeof(infiCmiChunkMetaData));
			metaData->key = key;
			metaData->owner = hdr;
			metaData->poolIdx = poolIdx;

			if(i == 0){
				metaData->owner->metaData->count = count;
				metaData->nextBuf = NULL;
			}else{
				void *startBuf = res - sizeof(infiCmiChunkHeader);
				metaData->nextBuf = infiCmiChunkPools[poolIdx].startBuf;
				infiCmiChunkPools[poolIdx].startBuf = startBuf;
				infiCmiChunkPools[poolIdx].count++;
				
			}
			if(i != count-1){
				res += (allocSize+sizeof(infiCmiChunkHeader));
			}
	  }	
		
		
		MACHSTATE3(3,"AllocSize %d buf %p key %p",allocSize,res,metaData->key);
		
		return origres;
	}
	if(poolIdx < INFINUMPOOLS){
		infiCmiChunkMetaData *metaData;				
	
		res = infiCmiChunkPools[poolIdx].startBuf;
		res += sizeof(infiCmiChunkHeader);

		MACHSTATE2(3,"Reusing old pool %d buf %p",poolIdx,res);
		metaData = METADATAFIELD(res);

		infiCmiChunkPools[poolIdx].startBuf = metaData->nextBuf;
		MACHSTATE2(1,"Pool %d now has startBuf at %p",poolIdx,infiCmiChunkPools[poolIdx].startBuf);
		
		metaData->nextBuf = NULL;
//		CmiAssert(metaData->poolIdx == poolIdx);

		infiCmiChunkPools[poolIdx].count--;
		return res;
	}

	CmiAssert(0);

	
};




void * infi_CmiAlloc(int size){
	void *res;
/*(	if(size-sizeof(CmiChunkHeader) > firstBinSize){*/
		MACHSTATE1(1,"infi_CmiAlloc for dataSize %d",size-sizeof(CmiChunkHeader));
		res = getInfiCmiChunk(size-sizeof(CmiChunkHeader));	
		res -= sizeof(CmiChunkHeader);
/*	}else{
		res = malloc(size);
	}*/
	return res;
}

void infi_CmiFree(void *ptr){
	int size;
	void *freePtr = ptr;
	
	ptr += sizeof(CmiChunkHeader);
	size = SIZEFIELD (ptr);
/*	if(size > firstBinSize){*/
		infiCmiChunkMetaData *metaData;
		int poolIdx;
		//there is a infiniband specific header
		freePtr = ptr - sizeof(infiCmiChunkHeader);
		metaData = METADATAFIELD(ptr);
		poolIdx = metaData->poolIdx;
		MACHSTATE2(1,"CmiFree buf %p goes back to pool %d",ptr,poolIdx);
//		CmiAssert(poolIdx >= 0);
		if(poolIdx < INFINUMPOOLS && infiCmiChunkPools[poolIdx].count <= INFIMAXPERPOOL){
			metaData->nextBuf = infiCmiChunkPools[poolIdx].startBuf;
			infiCmiChunkPools[poolIdx].startBuf = freePtr;
			infiCmiChunkPools[poolIdx].count++;
			
			MACHSTATE3(3,"Pool %d now has startBuf at %p count %d",poolIdx,infiCmiChunkPools[poolIdx].startBuf,infiCmiChunkPools[poolIdx].count);
		}else{
			MACHSTATE2(3,"Freeing up buf %p poolIdx %d",ptr,poolIdx);
			metaData->owner->metaData->count--;
			if(metaData->owner->metaData == metaData){
				//I am the owner
				if(metaData->owner->metaData->count == 0){
					//all the chunks have been freed
					ibv_dereg_mr(metaData->key);
					free(freePtr);
					free(metaData);
				}
				//if I am the owner and all the chunks have not been
				// freed dont free my metaData. will need later
			}else{
				if(metaData->owner->metaData->count == 0){
					//need to free the owner's buffer and metadata
					freePtr = metaData->owner;
					ibv_dereg_mr(metaData->key);
					free(metaData->owner->metaData);
					free(freePtr);
				}
				free(metaData);
			}
		}	
/*	}else{
		free(freePtr);
	}*/
}


/*********************************************************************************************
This section is for CmiDirect. This is a variant of the  persistent communication in which 
the user can transfer data between processors without using Charm++ messages. This lets the user
send and receive data from the middle of his arrays without any copying on either send or receive
side
*********************************************************************************************/

#define MAXHANDLES 100

struct infiDirectRequestPacket{
	int senderProc;
	int handle;
	struct ibv_mr senderKey;
	void *senderBuf;
	int senderBufSize;
};

typedef struct {
	int id;
	void *buf;
	int size;
	struct ibv_mr *key;
	void (*callbackFnPtr)(void *);
	void *callbackData;
	struct infiDirectRequestPacket *packet;
	struct infiRdmaPacket *rdmaPacket;
}	infiDirectHandle;

typedef struct infiDirectHandleTableStruct{
	infiDirectHandle handles[MAXHANDLES];
	struct infiDirectHandleTableStruct *next;
} infiDirectHandleTable;




static infiDirectHandleTable **sendHandleTable=NULL;
static infiDirectHandleTable **recvHandleTable=NULL;

static int *recvHandleCount=NULL;

static inline infiDirectHandleTable **createHandleTable(){
	infiDirectHandleTable **table = malloc(_Cmi_numnodes*sizeof(infiDirectHandleTable *));
	int i;
	for(i=0;i<_Cmi_numnodes;i++){
		table[i] = NULL;
	}
	return table;
}

static inline void calcHandleTableIdx(int handle,int *tableIdx,int *idx){
	*tableIdx = handle/MAXHANDLES;
	*idx = handle%MAXHANDLES;
};

/**
 To be called on the receiver to create a handle and return its number
**/
int CmiDirect_createHandle(int senderProc,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData){
	int newHandle;
	int tableIdx,idx;
	int i;
	infiDirectHandleTable *table;
	
	if(recvHandleTable == NULL){
		recvHandleTable = createHandleTable();
		recvHandleCount = malloc(sizeof(int)*_Cmi_numnodes);
		for(i=0;i<_Cmi_numnodes;i++){
			recvHandleCount[i] = -1;
		}
	}
	if(recvHandleTable[senderProc] == NULL){
		recvHandleTable[senderProc] = malloc(sizeof(infiDirectHandleTable));
		recvHandleTable[senderProc]->next = NULL;		
	}
	
	newHandle = ++recvHandleCount[senderProc];
	CmiAssert(newHandle >= 0);
	
	calcHandleTableIdx(newHandle,&tableIdx,&idx);
	
	table = recvHandleTable[senderProc];
	for(i=0;i<tableIdx;i++){
		if(table->next ==NULL){
			table->next = malloc(sizeof(infiDirectHandleTable));
			table->next->next = NULL;
		}
		table = table->next;
	}
	table->handles[idx].id = newHandle;
	table->handles[idx].buf = recvBuf;
	table->handles[idx].size = recvBufSize;
	table->handles[idx].callbackFnPtr = callbackFnPtr;
	table->handles[idx].callbackData = callbackData;
	table->handles[idx].key = ibv_reg_mr(context->pd, recvBuf, recvBufSize,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	table->handles[idx].rdmaPacket = CmiAlloc(sizeof(struct infiRdmaPacket));
	table->handles[idx].rdmaPacket->type = INFI_DIRECT;
	table->handles[idx].rdmaPacket->localBuffer = &(table->handles[idx]);
	
	return newHandle;
}

/****
 To be called on the sender to attach the sender's buffer to this handle
******/
void CmiDirect_assocLocalBuffer(int recverProc,int handle,void *sendBuf,int sendBufSize){
	int tableIdx,idx;
	int i;
	infiDirectHandleTable *table;

	if(sendHandleTable == NULL){
		sendHandleTable = createHandleTable();
	}
	if(sendHandleTable[recverProc] == NULL){
		sendHandleTable[recverProc] = malloc(sizeof(infiDirectHandleTable));
		sendHandleTable[recverProc]->next = NULL;
	}
	
	CmiAssert(handle >= 0);
	calcHandleTableIdx(handle,&tableIdx,&idx);
	
	table = sendHandleTable[recverProc];
	for(i=0;i<tableIdx;i++){
		if(table->next ==NULL){
			table->next = malloc(sizeof(infiDirectHandleTable));
			table->next->next = NULL;
		}
		table = table->next;
	}
	table->handles[idx].id = handle;
	table->handles[idx].buf = sendBuf;
	table->handles[idx].size = sendBufSize;
	table->handles[idx].callbackFnPtr = table->handles[idx].callbackData = NULL;
	table->handles[idx].key =  ibv_reg_mr(context->pd, sendBuf, sendBufSize,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	
	table->handles[idx].packet = (struct infiDirectRequestPacket *)CmiAlloc(sizeof(struct infiDirectRequestPacket));
	table->handles[idx].packet->senderProc = _Cmi_mynode;
	table->handles[idx].packet->handle = handle;
	table->handles[idx].packet->senderKey = *(table->handles[idx].key);
	table->handles[idx].packet->senderBuf = sendBuf;
	table->handles[idx].packet->senderBufSize = sendBufSize;
};





/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(int recverProc,int handle){
	if(recverProc == _Cmi_mynode){
		//TODO: Do a copy
		CmiAbort("OOPS!! forgot to do a memcpy");
	}else{
		infiPacket packet;
		int tableIdx,idx;
		int i;
		OtherNode node;
		infiDirectHandleTable *table;

		calcHandleTableIdx(handle,&tableIdx,&idx);

		table = sendHandleTable[recverProc];
		CmiAssert(table != NULL);
		for(i=0;i<tableIdx;i++){
			table = table->next;
		}

		
		MallocInfiPacket (packet);
		{
			packet->size = sizeof(struct infiDirectRequestPacket);
			packet->buf = (char *)&(table->handles[idx].packet);
			packet->header.code = INFIDIRECT_REQUEST;
			packet->header.nodeNo = _Cmi_mynode;
			packet->ogm = NULL;

			node = nodes_by_pe[recverProc];
			struct ibv_mr *packetKey = METADATAFIELD((void *)table->handles[idx].packet)->key;
			EnqueuePacket(node,packet,sizeof(struct infiDirectRequestPacket),packetKey);
		}
	}

};


void processDirectRequest(struct infiDirectRequestPacket *directRequestPacket){
	int senderProc = directRequestPacket->senderProc;
	int handle = directRequestPacket->handle;
	int tableIdx,idx,i;
	infiDirectHandleTable *table;
	OtherNode node = nodes_by_pe[senderProc];
	
	calcHandleTableIdx(handle,&tableIdx,&idx);

	table = recvHandleTable[senderProc];
	CmiAssert(table != NULL);
	for(i=0;i<tableIdx;i++){
		table = table->next;
	}
	
	CmiAssert(table->handles[idx].size == directRequestPacket->senderBufSize);
	
	{
		struct ibv_sge list = {
			.addr = (uintptr_t )table->handles[idx].buf,
			.length = table->handles[idx].size,
			.lkey 	= table->handles[idx].key->lkey
		};

		struct ibv_send_wr *bad_wr;
		struct ibv_send_wr wr = {
			.wr_id = (uint64_t)table->handles[idx].rdmaPacket,
			.sg_list = &list,
			.num_sge = 1,
			.opcode = IBV_WR_RDMA_READ,
			.send_flags = IBV_SEND_SIGNALED,
			.wr.rdma = {
				.remote_addr = (uint64_t )directRequestPacket->senderBuf,
				.rkey = directRequestPacket->senderKey.rkey
			}
		};
		/** post and rdma_read that is a rdma get*/
		if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
			assert(0);
		}
	}
			
	
};

void processDirectWC(struct infiRdmaPacket *rdmaPacket){
	infiDirectHandle *handle = (infiDirectHandle *)rdmaPacket->localBuffer;
	(*(handle->callbackFnPtr))(handle->callbackData);
};


