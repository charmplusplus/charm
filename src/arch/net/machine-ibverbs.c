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
static int dataSize;
static int rdma;
static int rdmaThreshold;


static int maxTokens;
static int tokensPerProcessor; /*number of outstanding sends and receives between any two nodes*/
static int sendPacketPoolSize; /*total number of send buffers created*/

static double _startTime=0;
static int regCount;

static int pktCount;
static int msgCount;


static double regTime;

/*TODO: remove this **/
/*char *rdmaOutBuf,*rdmaInBuf;
struct ibv_mr *outKey,*inKey;*/


#define CMK_IBVERBS_STATS 1
#define CMK_IBVERBS_INCTOKENS 1

#define WC_LIST_SIZE 100
/*#define WC_BUFFER_SIZE 100*/

#define INCTOKENS_FRACTION 0.04
#define INCTOKENS_INCREASE .50

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

struct infiPacketHeader{
	char code;
	int nodeNo;
};

struct infiBuffer;
struct infiRdmaPacket{
	struct infiPacketHeader header;
	struct ibv_mr key;
	struct ibv_mr *keyPtr;
	int remoteSize;
	char *remoteBuf;
	OutgoingMsg ogm;
	struct infiBuffer *localBuffer;
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

#define BCASTLIST_SIZE 50

struct infiBufferedBcastStruct{
	char *msg;
	int size;
	int broot;
	int asm_rank;
};

typedef struct infiBufferedBcastPoolStruct{
	struct infiBufferedBcastStruct bcastList[BCASTLIST_SIZE];
	int count;

	struct infiBufferedBcastPoolStruct *next,*prev;
} *infiBufferedBcastPool;




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
	struct ibv_srq  *srq;
	
	struct ibv_qp		**qp; //Array of qps (numNodes long) to temporarily store the queue pairs
												//It is used between CmiMachineInit and the call to node_addresses_store
												//when the qps are stored in the corresponding OtherNodes

	struct infiAddr *localAddr; //store the lid,qpn,msn address of ur qpair until they are sent

	infiPacket infiPacketFreeList; 
	
	struct infiBufferPool *recvBufferPool;

	struct infiPacketHeader header;

	int srqSize;

	infiBufferedBcastPool bufferedBcastList;
	
/*	infiBufferedWC infiBufferedRecvList;*/
};

static struct infiContext *context;

static inline infiPacket newPacket(int size){
	infiPacket pkt = malloc(sizeof(struct infiPacketStruct));
	pkt->size = size;
	pkt->buf = malloc(sizeof(char)*size);
	memcpy(pkt->buf,(char *)&(context->header),sizeof(struct infiPacketHeader));
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
	int totalTokens;
	int tokensLeft;
	int nodeNo;
	
	int postedRecvs;
	int broot;//needed to store the root of a multi-packet broadcast sent along a spanning tree or hypercube
};


/********************************
Memory management structures and types
*****************/

typedef struct {
	struct ibv_mr *key;
	int poolIdx;
	void *nextBuf;
} infiCmiChunkMetaData;


typedef struct {
	infiCmiChunkMetaData *metaData;
	CmiChunkHeader chunkHeader;
} infiCmiChunkHeader;


#define METADATAFIELD(m) (((infiCmiChunkHeader *)m)[-1].metaData)

typedef struct {
	int size;//without infiCmiChunkHeader
	void *startBuf;
} infiCmiChunkPool;

#define INFINUMPOOLS 15
infiCmiChunkPool infiCmiChunkPools[INFINUMPOOLS];

static void initInfiCmiChunkPools();






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
	
	context->header.nodeNo = _Cmi_mynode;

	mtu_size=4200;
	packetSize = mtu_size;
	dataSize = packetSize-sizeof(struct infiPacketHeader);//infiniband rc header size -estimate
	maxTokens =1000;
	tokensPerProcessor=100;
	createLocalQps(dev,ibPort,_Cmi_mynode,_Cmi_numnodes,context->localAddr);
		
	/*create the pool of arrays*/
	sendPacketPoolSize = (_Cmi_numnodes-1)*(tokensPerProcessor)/4;
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
	
	//TURN ON RDMA
	rdma=1;
//	rdmaThreshold=32768;
	rdmaThreshold=22000;
/*	context->infiBufferedRecvList = NULL;*/
#if CMK_IBVERBS_STATS	
	regCount =0;
	regTime  = 0;

	pktCount=0;
	msgCount=0;
#endif	

	initInfiCmiChunkPools();
/*
	rdmaOutBuf = (char *)CmiAlloc(4000000);
	rdmaInBuf = (char *)CmiAlloc(4000000);
	outKey = ibv_reg_mr(context->pd,rdmaOutBuf,4000000,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
	inKey = ibv_reg_mr(context->pd,rdmaInBuf,4000000,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE| IBV_ACCESS_REMOTE_WRITE);
*/
	

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
	context->sendCq = ibv_create_cq(context->context,(tokensPerProcessor*(numNodes-1))+5,NULL,NULL,0);
	assert(context->sendCq != NULL);
	

	context->recvCq = ibv_create_cq(context->context,(tokensPerProcessor*(numNodes-1))+5,NULL,NULL,0);
	assert(context->recvCq != NULL);
	
	MACHSTATE(3,"cq created");
	
	//array of queue pairs

	context->qp = (struct ibv_qp **)malloc(sizeof(struct ibv_qp *)*numNodes);

	{
		context->srqSize = (maxTokens+2)*(_Cmi_numnodes-1);
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
				.max_send_sge = 1,
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
	ret->totalTokens = tokensPerProcessor;
	ret->tokensLeft = tokensPerProcessor;
	ret->nodeNo = node;
	ret->postedRecvs = tokensPerProcessor;

	
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
	int numPosts = tokensPerProcessor*(_Cmi_numnodes-1);
	context->recvBufferPool = allocateInfiBufferPool(numPosts,packetSize);
	postInitialRecvs(context->recvBufferPool,numPosts,packetSize);


	free(context->qp);
	context->qp = NULL;
	free(context->localAddr);
	context->localAddr= NULL;
}

struct infiBufferPool * allocateInfiBufferPool(int numRecvs,int sizePerBuffer){
	int numBuffers;
	int i;
	struct infiBufferPool *ret;

	MACHSTATE(3,"allocateInfiBufferPool");

	page_size = sysconf(_SC_PAGESIZE);
	ret = malloc(sizeof(struct infiBufferPool));
	ret->next = NULL;
	numBuffers=ret->numBuffers = numRecvs;
	
	ret->buffers = malloc(sizeof(struct infiBuffer)*numBuffers);
	
	for(i=0;i<numBuffers;i++){
		struct infiBuffer *buffer =  &(ret->buffers[i]);
		buffer->type = BUFFER_RECV;
		buffer->size = sizePerBuffer;
		buffer->buf = memalign(page_size,sizePerBuffer);
		buffer->key = ibv_reg_mr(context->pd,buffer->buf,buffer->size,IBV_ACCESS_LOCAL_WRITE);
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
	printf("[%d] msgCount %d pktCount %d packetSize %d total Time %.6lf s # Rdma Reg-Dereg %d Reg-Dereg time %.6lf s \n",_Cmi_mynode,msgCount,pktCount,packetSize,CmiTimer(),regCount,regTime);
#endif
}

static void CmiNotifyStillIdle(CmiIdleState *s) {
	CommunicationServer_nolock(0);
}

static inline void increaseTokens(OtherNode node);

/**
	Packetize this data and send it
**/



static void inline EnqueuePacket(OtherNode node,infiPacket packet,int totalSize){
	int incTokens=0;

	struct ibv_sge sendElement = {
		.addr = (uintptr_t)packet->buf,
		.length = totalSize,
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
	struct infiPacketHeader *header = (struct infiPacketHeader *)packet->buf;
	header->nodeNo = _Cmi_mynode;
	
	packet->destNode = node;
	
#if CMK_IBVERBS_STATS	
	pktCount++;
#endif	
	while(node->infiData->tokensLeft == 0){
		CommunicationServer_nolock(1); 
	}
	

#if CMK_IBVERBS_INCTOKENS	
	if(node->infiData->tokensLeft < INCTOKENS_FRACTION*node->infiData->totalTokens && node->infiData->totalTokens < maxTokens){
		header->code |= INFIPACKETCODE_INCTOKENS;
		incTokens=1;
	}
#endif
	
	if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
		assert(0);
	}
	node->infiData->tokensLeft--;
	
#if CMK_IBVERBS_INCTOKENS	
	if(incTokens){
		increaseTokens(node);
	}
#endif
	MACHSTATE3(3,"Packet send size %d packet %p tokensLeft %d",totalSize,packet,packet->destNode->infiData->tokensLeft);

};

static void inline EnqueueDataPacket(OutgoingMsg ogm, OtherNode node, int rank,char *data,int size,int broot,int copy){
	infiPacket packet;
	MallocInfiPacket(packet);
	
	//the nodeNo is added at time of buffer allocation
	struct infiPacketHeader *header = (struct infiPacketHeader *)packet->buf;
	header->code = INFIPACKETCODE_DATA;
	
	//copy the data
	memcpy((packet->buf+sizeof(struct infiPacketHeader)),data,size);
	EnqueuePacket(node,packet,size+sizeof(struct infiPacketHeader));
};

static inline void EnqueueRdmaPacket(OutgoingMsg ogm, OtherNode node);

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
	MACHSTATE3(3,"DONE Sending ogm %p of size %d to %d",ogm,size,node->infiData->nodeNo);
}


static inline void EnqueueRdmaPacket(OutgoingMsg ogm, OtherNode node){
	infiPacket packet;

	ogm->refcount++;
	
	MallocInfiPacket(packet);
 
 {
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)packet->buf;
		
#if CMK_IBVERBS_STATS
		double _startRegTime = CmiWallTimer();
#endif
/*		struct ibv_mr *key = ibv_reg_mr(context->pd,ogm->data,ogm->size,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);*/
		struct ibv_mr *key = METADATAFIELD(ogm->data)->key;
		MACHSTATE3(3,"ogm->data %p metadata %p key %p",ogm->data,METADATAFIELD(ogm->data),key);
#if CMK_IBVERBS_STATS
		regCount++;
		regTime += CmiWallTimer()-_startRegTime;
#endif
		
		/*TODO:remove this
		memcpy(rdmaOutBuf,ogm->data,ogm->size);
		struct ibv_mr *key = outKey;*/
		
		rdmaPacket->key = *key;
		rdmaPacket->keyPtr = key;
		rdmaPacket->header.code = INFIRDMA_START;
		rdmaPacket->header.nodeNo = _Cmi_mynode;
		rdmaPacket->ogm = ogm;
		rdmaPacket->remoteBuf = ogm->data;
		rdmaPacket->remoteSize = ogm->size;
		
		/*TODO: remove
		rdmaPacket->remoteBuf = rdmaOutBuf;*/
		
		MACHSTATE3(3,"rdmaRequest being sent to node %d buf %p size %d",node->infiData->nodeNo,ogm->data,ogm->size);
		EnqueuePacket(node,packet,sizeof(struct infiRdmaPacket));
	}
}


static inline void pollRecvCq(const int toBuffer);
static inline void pollSendCq(const int toBuffer);

static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer);
static inline void processSendWC(struct ibv_wc *sendWC);
static int _count=0;

static inline  void CommunicationServer_nolock(int toBuffer) {
	_count++;
	MACHSTATE(2,"CommServer_nolock{");
	

	pollRecvCq(toBuffer);
	

	pollSendCq(toBuffer);
	
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

static inline  void processRdmaWC(struct ibv_wc *rdmaWC,const int toBuffer);

static inline void pollSendCq(const int toBuffer){
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
			case IBV_WC_RDMA_READ:
			{
				processRdmaWC(&wc[i],toBuffer);
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


static inline infiBufferedBcastPool createBcastPool(){
	infiBufferedBcastPool ret = malloc(sizeof(struct infiBufferedBcastPoolStruct));
	ret->count = 0;
	ret->next = ret->prev = NULL;	
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
}

/*********
	Go through the blocks of buffered bcast messages. process last block first
	processign within a block is in sequence though
*********/
static inline void processBufferedBcast(){
	infiBufferedBcastPool start;
	start = context->bufferedBcastList;

	while(start->next != NULL){
		start = start->next;
	}
	
	while(start != NULL){
		int i=0;
		infiBufferedBcastPool tmp;
		for(i=0;i<start->count;i++){
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
		
		if(start != context->bufferedBcastList){
			//not the first one
			tmp = start;
			start = start->prev;
			free(tmp);
			start->next = NULL;
		}else{
			start = start->prev;
		}
	}

	context->bufferedBcastList->next = NULL;
	context->bufferedBcastList->prev = NULL;
	context->bufferedBcastList->count =0;	
	
};


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
			if(node->asm_fill + len < node->asm_total && len != dataSize){
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
		handoverMessage(newmsg,total_size,node->asm_rank,node->infiData->broot,toBuffer);
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
}


static inline void increasePostedRecvs(int nodeNo);
static inline void processRdmaRequest(struct infiRdmaPacket *rdmaPacket);
static inline void processRdmaAck(struct infiRdmaPacket *rdmaPacket);


static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer){
	struct infiBuffer *buffer = (struct infiBuffer *) recvWC->wr_id;	
	struct infiPacketHeader *header = (struct infiPacketHeader *)buffer->buf;
	int nodeNo = header->nodeNo;
	
	int len = recvWC->byte_len-sizeof(struct infiPacketHeader);
	
	if(header->code & INFIPACKETCODE_DATA){
			processMessage(nodeNo,len,(buffer->buf+sizeof(struct infiPacketHeader)),toBuffer);
	}
#if CMK_IBVERBS_INCTOKENS	
	if(header->code & INFIPACKETCODE_INCTOKENS){
		increasePostedRecvs(nodeNo);
	}
#endif	
	if(rdma && header->code & INFIRDMA_START){
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)buffer->buf;
		processRdmaRequest(rdmaPacket);
	}
	if(rdma && header->code & INFIRDMA_ACK){
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)buffer->buf;
		processRdmaAck(rdmaPacket);
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

	packet->destNode->infiData->tokensLeft++;

	MACHSTATE2(3,"Packet send complete packet %p tokensLeft %d",packet,packet->destNode->infiData->tokensLeft);

	FreeInfiPacket(packet);
};



/********************************************************************/
//TODO: get token for rdma later
static inline void processRdmaRequest(struct infiRdmaPacket *_rdmaPacket){
#if CMK_IBVERBS_STATS
	double _startRegTime;
#endif	
	int nodeNo = _rdmaPacket->header.nodeNo;
	OtherNode node = &nodes[nodeNo];

	struct infiBuffer *buffer = malloc(sizeof(struct infiBuffer));
//	CmiAssert(buffer != NULL);
	struct infiRdmaPacket *rdmaPacket = malloc(sizeof(struct infiRdmaPacket));
	
	*rdmaPacket = *_rdmaPacket;
	rdmaPacket->localBuffer = buffer;
	
	buffer->type = BUFFER_RDMA;
	buffer->size = rdmaPacket->remoteSize;
	
	buffer->buf  = (char *)CmiAlloc(rdmaPacket->remoteSize);
//	CmiAssert(buffer->buf != NULL);

#if CMK_IBVERBS_STATS
		_startRegTime = CmiWallTimer();
#endif
/*	buffer->key = ibv_reg_mr(context->pd,buffer->buf,buffer->size,IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |IBV_ACCESS_REMOTE_READ );*/
		buffer->key = METADATAFIELD(buffer->buf)->key;
#if CMK_IBVERBS_STATS
		regCount++;
		regTime += CmiWallTimer()-_startRegTime;
#endif

	/*TODO: remove this
	buffer->key = inKey;*/
	
	MACHSTATE3(3,"received rdma request from node %d for remoteBuffer %p keyPtr %p",nodeNo,rdmaPacket->remoteBuf,rdmaPacket->keyPtr);
	MACHSTATE3(3,"Local buffer->buf %p buffer->key %p rdmaPacket %p",buffer->buf,buffer->key,rdmaPacket);
//	CmiAssert(buffer->key != NULL);
	
	{
		struct ibv_sge list = {
			.addr = (uintptr_t )buffer->buf,
			/*TODO: change this
			.addr = (uintptr_t )rdmaInBuf,*/
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

static inline  void processRdmaWC(struct ibv_wc *rdmaWC,const int toBuffer){
		//rdma get done
#if CMK_IBVERBS_STATS
	double _startRegTime;
#endif	

	struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *) rdmaWC->wr_id;
	struct infiBuffer *buffer = rdmaPacket->localBuffer;

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

#if CMK_IBVERBS_STATS
		_startRegTime = CmiWallTimer();
#endif
	
//		ibv_dereg_mr(buffer->key);
#if CMK_IBVERBS_STATS
		regCount++;
		regTime += CmiWallTimer()-_startRegTime;
#endif
	
	free(buffer);

	//send ack to sender 
	EnqueueRdmaAck(rdmaPacket);
	free(rdmaPacket);
}

static inline void EnqueueRdmaAck(struct infiRdmaPacket *rdmaPacket){
	infiPacket packet;
	OtherNode node=&nodes[rdmaPacket->header.nodeNo];
	MallocInfiPacket(packet);
	
	{
		struct infiRdmaPacket *ackPacket = (struct infiRdmaPacket *) packet->buf;
		*ackPacket = *rdmaPacket;
		ackPacket->header.code = INFIRDMA_ACK;
	

		EnqueuePacket(node,packet,sizeof(struct infiRdmaPacket));
	}
};


static inline void processRdmaAck(struct infiRdmaPacket *rdmaPacket){
#if CMK_IBVERBS_STATS
	double _startRegTime=CmiWallTimer();
#endif	
	
//	ibv_dereg_mr(rdmaPacket->keyPtr);

#if CMK_IBVERBS_STATS
		regCount++;
		regTime += CmiWallTimer()-_startRegTime;
#endif

	MACHSTATE2(3,"rdma ack received for remoteBuf %p size %d",rdmaPacket->remoteBuf,rdmaPacket->remoteSize);
	rdmaPacket->ogm->refcount--;
	GarbageCollectMsg(rdmaPacket->ogm);
}


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
	//increase the size of the sendCq
	int currentCqSize = context->sendCq->cqe;
	if(ibv_resize_cq(context->sendCq,currentCqSize+increase)){
		assert(0);
	}
};



static void increasePostedRecvs(int nodeNo){
	OtherNode node = &nodes[nodeNo];
	int increase = node->infiData->postedRecvs*INCTOKENS_INCREASE;	
	if(increase+node->infiData->postedRecvs > maxTokens){
		increase = maxTokens - node->infiData->postedRecvs;
	}
	node->infiData->postedRecvs+= increase;
	MACHSTATE3(3,"Increase tokens by %d to %d for node %d ",increase,node->infiData->postedRecvs,nodeNo);
	//increase the size of the recvCq
	int currentCqSize = context->recvCq->cqe;
	if(ibv_resize_cq(context->recvCq,currentCqSize+increase)){
		assert(0);
	}

	//create another bufferPool and attach it to the top of the current one
	struct infiBufferPool *newPool = allocateInfiBufferPool(increase,packetSize);
	newPool->next = context->recvBufferPool;
	context->recvBufferPool = newPool;
	postInitialRecvs(newPool,increase,packetSize);

};




/*********************************************
	Memory management routines for RDMA

************************************************/


static void initInfiCmiChunkPools(){
	int i;
	int size = rdmaThreshold*2;
	
	for(i=0;i<INFINUMPOOLS;i++){
		infiCmiChunkPools[i].size = size; // pool i has buffers of size rdmaThreshold*2^(i+1)
		infiCmiChunkPools[i].startBuf = NULL;
		size *= 2;
	}
}

static inline void *getInfiCmiChunk(int dataSize){
	//find out to which pool this dataSize belongs to
	// poolIdx = rint(log2(dataSize/rdmaThreshold))
	int ratio = dataSize/rdmaThreshold;
	int poolIdx=-1;
	void *res;
	
	CmiAssert(ratio >= 1);
	while(ratio > 0){
		ratio  = ratio >> 1;
		poolIdx++;
	}
	MACHSTATE2(3,"getInfiCmiChunk for size %d in poolIdx %d",dataSize,poolIdx);
	if((poolIdx < INFINUMPOOLS && infiCmiChunkPools[poolIdx].startBuf == NULL) || poolIdx >= INFINUMPOOLS){
		infiCmiChunkMetaData *metaData;		
		int allocSize;
		if(poolIdx < INFINUMPOOLS ){
			allocSize = infiCmiChunkPools[poolIdx].size;
		}else{
			allocSize = dataSize;
		}
		res = malloc(allocSize+sizeof(infiCmiChunkHeader));
		res += sizeof(infiCmiChunkHeader);
		
		
		metaData = METADATAFIELD(res) = malloc(sizeof(infiCmiChunkMetaData));
		metaData->key = ibv_reg_mr(context->pd,res,allocSize,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
		
		MACHSTATE3(3,"AllocSize %d buf %p key %p",allocSize,res,metaData->key);
		
		CmiAssert(metaData->key != NULL);
		metaData->poolIdx = poolIdx;
		metaData->nextBuf = NULL;
		return res;
	}
	if(poolIdx < INFINUMPOOLS){
		infiCmiChunkMetaData *metaData;				
	
		res = infiCmiChunkPools[poolIdx].startBuf;
		res += sizeof(infiCmiChunkHeader);

		MACHSTATE2(3,"Reusing old pool %d buf %p",poolIdx,res);
		metaData = METADATAFIELD(res);

		infiCmiChunkPools[poolIdx].startBuf = metaData->nextBuf;
		MACHSTATE2(3,"Pool %d now has startBuf at %p",poolIdx,infiCmiChunkPools[poolIdx].startBuf);
		
		metaData->nextBuf = NULL;
		CmiAssert(metaData->poolIdx == poolIdx);
		return res;
	}

	CmiAssert(0);

	
};




void * infi_CmiAlloc(int size){
	void *res;
	if(size-sizeof(CmiChunkHeader) > rdmaThreshold){
		MACHSTATE1(3,"infi_CmiAlloc for dataSize %d",size-sizeof(CmiChunkHeader));
		res = getInfiCmiChunk(size-sizeof(CmiChunkHeader));	
		res -= sizeof(CmiChunkHeader);
	}else{
		res = malloc(size);
	}
	return res;
}

void infi_CmiFree(void *ptr){
	int size;
	void *freePtr = ptr;
	
	ptr += sizeof(CmiChunkHeader);
	size = SIZEFIELD (ptr);
	if(size > rdmaThreshold){
		infiCmiChunkMetaData *metaData;
		int poolIdx;
		//there is a infiniband specific header
		freePtr = ptr - sizeof(infiCmiChunkHeader);
		metaData = METADATAFIELD(ptr);
		poolIdx = metaData->poolIdx;
		MACHSTATE2(3,"CmiFree buf %p goes back to pool %d",ptr,poolIdx);
		CmiAssert(poolIdx >= 0);
		if(poolIdx < INFINUMPOOLS){
			metaData->nextBuf = infiCmiChunkPools[poolIdx].startBuf;
			infiCmiChunkPools[poolIdx].startBuf = freePtr;
			
			MACHSTATE2(3,"Pool %d now has startBuf at %p",poolIdx,infiCmiChunkPools[poolIdx].startBuf);
		}else{			
			ibv_dereg_mr(metaData->key);
			free(metaData);
			free(freePtr);
		}	
	}else{
		free(freePtr);
	}
}












