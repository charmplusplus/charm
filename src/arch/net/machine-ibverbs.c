/** @file
 * Ibverbs (infiniband)  implementation of Converse NET version
 * @ingroup NET
 * contains only Ibverbs specific code for:
 * - CmiMachineInit()
 * - CmiCommunicationInit()
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
#if CMK_HAS_MALLOC_H
#include <malloc.h>
#endif
#include <getopt.h>
#include <time.h>

#include <infiniband/verbs.h>

//#define QLOGIC
#if ! QLOGIC
enum ibv_mtu mtu = IBV_MTU_2048;
#else
enum ibv_mtu mtu = IBV_MTU_4096;
#endif
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
//#define NON_SRQ
#ifdef NON_SRQ
static int minPerProcessorRecvs;
#endif
static double _startTime=0;
static int regCount;

static int pktCount;
static int msgCount;
static int minTokensLeft;


static double regTime;

static double processBufferedTime;
static int processBufferedCount;

#define CMK_IBVERBS_STATS 0
#define CMK_IBVERBS_TOKENS_FLOW 1
#define CMK_IBVERBS_INCTOKENS 0 //never turn this on 
#define CMK_IBVERBS_DEBUG 0
#define CMI_DIRECT_DEBUG 0
#define WC_LIST_SIZE 32
/*#define WC_BUFFER_SIZE 100*/

#if CMK_IBVERBS_STATS
static int numReg=0;
static int numUnReg=0;
static int numCurReg=0;
static int numAlloc=0;
static int numFree=0;
static int numMultiSendUnreg=0;
static int numMultiSend=0;
static int numMultiSendFree=0;
#endif


#define INCTOKENS_FRACTION 0.04
#define INCTOKENS_INCREASE .50

// flag for using a pool for every thread in SMP mode
#if CMK_SMP
#define THREAD_MULTI_POOL 1
#endif

#if THREAD_MULTI_POOL 
#include "pcqueue.h"
PCQueue **queuePool;
void infi_CmiFreeDirect(void *ptr);
static inline void fillBufferPools();
#endif

#define INFIBARRIERPACKET 128

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
#ifndef NON_SRQ
	struct ibv_srq  *srq;
#endif	
	struct ibv_qp		**qp; //Array of qps (numNodes long) to temporarily store the queue pairs
												//It is used between CmiMachineInit and the call to node_addresses_store
												//when the qps are stored in the corresponding OtherNodes

	struct infiAddr *localAddr; //store the lid,qpn,msn address of ur qpair until they are sent

	infiPacket infiPacketFreeList; 
	
	struct infiBufferPool *recvBufferPool;

	struct infiPacketHeader header;

#ifndef NON_SRQ
	int srqSize;
#endif	
	int sendCqSize,recvCqSize;
	int tokensLeft;

	infiBufferedBcastPool bufferedBcastList;
	
	struct infiRdmaPacket *bufferedRdmaAcks;
	
	struct infiRdmaPacket *bufferedRdmaRequests;
/*	infiBufferedWC infiBufferedRecvList;*/
	
	int insideProcessBufferedBcasts;
};

static struct infiContext *context = NULL;




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
	int recvPsn;
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

#if THREAD_MULTI_POOL
	int parentPe;						// the PE that allocated the buffer and must release it
#endif
} infiCmiChunkMetaData;




#define METADATAFIELD(m) (((infiCmiChunkHeader *)m)[-1].metaData)

typedef struct {
	int size;//without infiCmiChunkHeader
	void *startBuf;
	int count;
} infiCmiChunkPool;

#define INFINUMPOOLS 14
#define INFIMAXPERPOOL 100
#define INFIMULTIPOOL 0xDEAFB00D

#if THREAD_MULTI_POOL
static infiCmiChunkPool **infiCmiChunkPools;
//TODO Find proper place to dispose the memory acquired by infiCmiChunkPool
#else
static infiCmiChunkPool infiCmiChunkPools[INFINUMPOOLS];
#endif

static void initInfiCmiChunkPools();


static inline infiPacket newPacket(){
	infiPacket pkt = (infiPacket )CmiAlloc(sizeof(struct infiPacketStruct));
	pkt->size = -1;
	pkt->header = context->header;
	pkt->next = NULL;
	pkt->destNode = NULL;
	pkt->keyHeader = METADATAFIELD(pkt)->key;
	pkt->ogm=NULL;
	CmiAssert(pkt->keyHeader!=NULL);
	pkt->buf=NULL;
	
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
	pkt->buf=NULL;\
	pkt->next = context->infiPacketFreeList; \
	context->infiPacketFreeList = pkt; \
}

#define MallocInfiPacket(pkt) { \
	infiPacket p = context->infiPacketFreeList; \
	if(p == NULL){ p = newPacket();} \
	         else{context->infiPacketFreeList = p->next; } \
	pkt = p;\
}



void infi_unregAndFreeMeta(void *md)
{
  if(md!=NULL && (((infiCmiChunkMetaData *)md)->poolIdx == INFIMULTIPOOL))
    {
      int unregstat=ibv_dereg_mr(((infiCmiChunkMetaData*)md)->key);
      CmiAssert(unregstat==0);
      free(((infiCmiChunkMetaData *)md));
#if CMK_IBVERBS_STATS
      numUnReg++;
      numCurReg--;
      numMultiSendUnreg++;
#endif
    }
}


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
				CmiEnforce(0);
			}
		}
	}
}

#if CMK_IBVERBS_FAST_START
static void send_partial_init();
#endif

static void CmiMachineInit(char **argv){
	struct ibv_device **devList;
	struct ibv_device *dev;
	int ibPort;
	int i;
        int MAXPORT = 8;
	int calcMaxSize;
	infiPacket *pktPtrs;
	struct infiRdmaPacket **rdmaPktPtrs;
        int num_devices, idev;
#define MAX_DEVICE_NAME 120
        char *usr_ibv_device_name=NULL;
        int ibv_device_name_set=0;

#if CMK_SMP
        ibv_fork_init();
#endif
	MACHSTATE(3,"CmiMachineInit {");
	MACHSTATE2(3,"_Cmi_numnodes %d CmiNumNodes() %d",_Cmi_numnodes,CmiNumNodes());
	MACHSTATE1(3,"CmiMyNodeSize() %d",CmiMyNodeSize());
	
	devList =  ibv_get_device_list(&num_devices);
        CmiEnforce(num_devices > 0);
	CmiEnforce(devList != NULL);
	if (CmiGetArgStringDesc(argv,"+IBVDeviceName",&usr_ibv_device_name,"User set IBV device name"))
          {
	    MACHSTATE1(3,"IBVDeviceName set %s",usr_ibv_device_name);
	    ibv_device_name_set=1;
          }

	context = (struct infiContext *)malloc(sizeof(struct infiContext));
	MACHSTATE1(3,"context allocated %p",context);
	
	//localAddr will store the local addresses of all the qps
	context->localAddr = (struct infiAddr *)malloc(sizeof(struct infiAddr)*_Cmi_numnodes);
	MACHSTATE1(3,"context->localAddr allocated %p",context->localAddr);

        idev = 0;
        // try all devices, can't assume device 0 is IB, it may be ethernet
loop:
	dev = devList[idev];
	CmiEnforce(dev != NULL);

	MACHSTATE2(3,"device name %s for %d",ibv_get_device_name(dev), idev);
	//the context for this infiniband device 
	context->context = ibv_open_device(dev);
	CmiEnforce(context->context != NULL);

        // test ibPort
        for (ibPort = 1; ibPort < MAXPORT; ibPort++) {
          struct ibv_port_attr attr;
          if (ibv_query_port(context->context, ibPort, &attr) != 0) continue;
#if CMK_IBV_PORT_ATTR_HAS_LINK_LAYER
          if (attr.link_layer == IBV_LINK_LAYER_INFINIBAND)  break;
#else
          break;
#endif
          
        }
        if (ibPort == MAXPORT) {
          if (++idev == num_devices)
            CmiAbort("No valid IB port found!");
          else
            goto loop;
        }
	if(ibv_device_name_set)
	  {
	    if(strcmp(usr_ibv_device_name,ibv_get_device_name(dev))==0)
	      {
		MACHSTATE2(3, "device %d selected for user requested IBVDeviceName %s\n",idev, ibv_get_device_name(dev));
	      }
	    else
	      { // force increment to next device
		if(ibPort != MAXPORT) ++idev;
		goto loop;
	      }
	  }
	context->ibPort = ibPort;
	MACHSTATE1(3,"use port %d", ibPort);
	
	MACHSTATE1(3,"device opened %p",context->context);

/*	FD_ZERO(&context->asyncFds);
	FD_SET(context->context->async_fd,&context->asyncFds);
	context->tmo.tv_sec=0;
	context->tmo.tv_usec=0;
	
	MACHSTATE(3,"asyncFds zeroed and set");*/

	//protection domain
	context->pd = ibv_alloc_pd(context->context);
	CmiEnforce(context->pd != NULL);
	MACHSTATE2(3,"pd %p pd->handle %d",context->pd,context->pd->handle);

  /******** At this point we know that this node is more or less serviceable
	So, this is a good point for sending the partial init message for the fast
	start case
	Moreover, no work dependent on the number of nodes has started yet.
	************/

#if CMK_IBVERBS_FAST_START
  send_partial_init();
#endif


	context->header.nodeNo = _Cmi_mynode;

	mtu_size=1200;
	packetSize = mtu_size*4;
	dataSize = packetSize-sizeof(struct infiPacketHeader);
	
	calcMaxSize=8000;
/*	if(_Cmi_numnodes*50 > calcMaxSize){
		calcMaxSize = _Cmi_numnodes*50;
		if(calcMaxSize > 10000){
			calcMaxSize = 10000;
		}
	}*/
	maxRecvBuffers=calcMaxSize;
	if (CmiGetArgIntDesc(argv,"+IBVMaxSendTokens",&maxTokens,"User Set IBV Max Outstanding Send Tokens") == 0)
	  maxTokens = 1000; // this value may need to be tweaked later
	context->tokensLeft=maxTokens;
	context->qp=NULL;
	//tokensPerProcessor=4;
	if(_Cmi_numnodes > 1){
#if !CMK_IBVERBS_FAST_START
		/* a barrier to make sure all nodes initialized the device */
  		ChMessage msg;
    		ctrl_sendone_nolock("barrier",NULL,0,NULL,0);
  		ChMessage_recv(Cmi_charmrun_fd,&msg);
#endif
		createLocalQps(dev,ibPort,_Cmi_mynode,_Cmi_numnodes,context->localAddr);
	}
	
        if (Cmi_charmrun_fd == -1) return;
	
	//TURN ON RDMA
	rdma=1;
//	rdmaThreshold=32768;
	rdmaThreshold=22000;
	firstBinSize = 128;
	CmiAssert(rdmaThreshold > firstBinSize);
	/*	blockAllocRatio=16;
		blockThreshold=8;*/

	blockAllocRatio=64;
	blockThreshold=9;



#if !THREAD_MULTI_POOL
	initInfiCmiChunkPools();
#endif

	/*create the pool of send packets*/
	sendPacketPoolSize = maxTokens/2;	
	if(sendPacketPoolSize > 2000){
		sendPacketPoolSize = 2000;
	}
	
	context->infiPacketFreeList=NULL;
	pktPtrs = malloc(sizeof(infiPacket)*sendPacketPoolSize);

	//Silly way of allocating the memory buffers (slow as well) but simplifies the code
#if !THREAD_MULTI_POOL
	for(i=0;i<sendPacketPoolSize;i++){
		MallocInfiPacket(pktPtrs[i]);	
	}

	for(i=0;i<sendPacketPoolSize;i++){
		FreeInfiPacket(pktPtrs[i]);	
	}
	free(pktPtrs);
#endif
	
	context->bufferedBcastList=NULL;
	context->bufferedRdmaAcks = NULL;
	context->bufferedRdmaRequests = NULL;
	context->insideProcessBufferedBcasts=0;
	
	
	if(rdma){
/*		int numPkts;
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
		free(rdmaPktPtrs);*/
	}
	
/*	context->infiBufferedRecvList = NULL;*/
#if CMK_IBVERBS_STATS	
	regCount =0;
	regTime  = 0;

	pktCount=0;
	msgCount=0;

	processBufferedCount=0;
	processBufferedTime=0;

	minTokensLeft = maxTokens;
#endif	

	

	MACHSTATE(3,"} CmiMachineInit");
}

void CmiCommunicationInit(char **argv)
{
#if THREAD_MULTI_POOL
	initInfiCmiChunkPools();
	fillBufferPools();
#endif
}

/*********
	Open a qp for every processor
*****/
void createLocalQps(struct ibv_device *dev,int ibPort, int myNode,int numNodes,struct infiAddr *localAddr){
	int myLid;
	int i;
	int err;
	
	//find my lid
	myLid = getLocalLid(context->context,ibPort);
	
	MACHSTATE2(3,"myLid %d numNodes %d",myLid,numNodes);

	context->sendCqSize = maxTokens+2;
	context->sendCq = ibv_create_cq(context->context,context->sendCqSize,NULL,NULL,0);
	CmiAssert(context->sendCq != NULL);
	
	MACHSTATE1(3,"sendCq created %p",context->sendCq);
	
	
	context->recvCqSize = maxRecvBuffers+2;
	context->recvCq = ibv_create_cq(context->context,context->recvCqSize,NULL,NULL,0);
	
	MACHSTATE2(3,"recvCq created %p %d",context->recvCq,context->recvCqSize);
	CmiAssert(context->recvCq != NULL);
	
	//array of queue pairs

	context->qp = (struct ibv_qp **)malloc(sizeof(struct ibv_qp *)*numNodes);

	if(numNodes > 1)
	{
		struct ibv_qp_attr attr;
#ifndef NON_SRQ
		context->srqSize = (maxRecvBuffers+2);
              {
		struct ibv_srq_init_attr srqAttr = {
			.attr = {
			.max_wr  = context->srqSize,
			.max_sge = 1
			}
		};
		context->srq = ibv_create_srq(context->pd,&srqAttr);
		CmiAssert(context->srq != NULL);
              }
#endif	
              
              {
		struct ibv_qp_init_attr initAttr = {
			.qp_type = IBV_QPT_RC,
			.send_cq = context->sendCq,
			.recv_cq = context->recvCq,
#ifndef NON_SRQ
			.srq		 = context->srq,
#endif	
			.sq_sig_all = 0,
			.qp_context = NULL,
			.cap     = {
				.max_send_wr  = maxTokens,
				.max_send_sge = 2,
#ifdef NON_SRQ
				.max_recv_wr  = maxRecvBuffers, // or maxRecvBuffers
				.max_recv_sge = 1,
#endif	
			},
		};

		attr.qp_state        = IBV_QPS_INIT;
		attr.pkey_index      = 0;
		attr.port_num        = ibPort;
         	attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

/*		MACHSTATE1(3,"context->pd %p",context->pd);
		struct ibv_qp *qp = ibv_create_qp(context->pd,&initAttr);
		MACHSTATE1(3,"TEST QP %p",qp);*/

		for( i=1;i<numNodes;i++){
                        int n = (myNode + i)%numNodes;
			if(n == myNode){
			}else{
				localAddr[n].lid = myLid;
				context->qp[n] = ibv_create_qp(context->pd,&initAttr);
			
				MACHSTATE2(3,"qp[%d] created %p",n,context->qp[n]);
				CmiAssert(context->qp[n] != NULL);
			
				if(err= ibv_modify_qp(context->qp[n], &attr,
					  IBV_QP_STATE              |
					  IBV_QP_PKEY_INDEX         |
				  	IBV_QP_PORT               |
				  	IBV_QP_ACCESS_FLAGS)) {
					     		  	MACHSTATE1(3,"ERROR modifying  to INIT %d",err);
                                                      	        CmiAbort("failed to change qp state to INIT ");
                                                                }
				localAddr[n].qpn = context->qp[n]->qp_num;
				localAddr[n].psn = lrand48() & 0xffffff;
				MACHSTATE4(3,"i %d lid Ox%x qpn 0x%x psn 0x%x",n,localAddr[n].lid,localAddr[n].qpn,localAddr[n].psn);
			}
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
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= mtu,
		.dest_qp_num		= addr[1],
		.rq_psn 		= addr[2],
		.max_dest_rd_atomic	= 1,
		.min_rnr_timer		= 31,
		.ah_attr		= {
			.is_global	= 0,
			.dlid		= addr[0],
			.sl		= 0,
			.src_path_bits	= 0,
			.port_num	= context->ibPort
		}
	};
	
	ret->state = INFI_HEADER_DATA;
	ret->qp = context->qp[node];
//	ret->totalTokens = tokensPerProcessor;
//	ret->tokensLeft = tokensPerProcessor;
	ret->nodeNo = node;
//	ret->postedRecvs = tokensPerProcessor;
#if	CMK_IBVERBS_DEBUG	
	ret->psn = 0;
	ret->recvPsn = 0;
#endif
	
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
// Here NON_SRQ is for QLOGIC
#if ! QLOGIC
	attr.timeout 	    = 26;
	attr.retry_cnt 	    = 20;
#else
	attr.timeout 	    = 14;
	attr.retry_cnt 	    = 7;
#endif
	attr.rnr_retry 	    = 7;
	attr.sq_psn 	    = context->localAddr[node].psn;
	attr.max_rd_atomic  = 1;

	MACHSTATE3(3,"dlid 0x%x qp 0x%x psn 0x%x",attr.ah_attr.dlid,attr.dest_qp_num,attr.sq_psn);
	

	if (err=ibv_modify_qp(ret->qp, &attr,
	  IBV_QP_STATE              |
	  IBV_QP_TIMEOUT            |
	  IBV_QP_RETRY_CNT          |
	  IBV_QP_RNR_RETRY          |
	  IBV_QP_SQ_PSN             |
	  IBV_QP_MAX_QP_RD_ATOMIC)) {
			MACHSTATE1(3,"ERROR changing qp state to RTS %d: will retry",err);
	}
	// Error code 22 means that there was an invalid parameter when calling to this verbs, try with alternate parameters
	if (err == 22)
	{
          //use inverted logic
#if QLOGIC
          mtu = IBV_MTU_2048;
          attr.path_mtu             = mtu;
          attr.timeout              = 26;
          attr.retry_cnt            = 20;
#else
          mtu = IBV_MTU_4096;
          attr.path_mtu             = mtu;
          attr.timeout              = 14;
          attr.retry_cnt            = 7;
#endif

          MACHSTATE3(3,"Retry:dlid 0x%x qp 0x%x psn 0x%x",attr.ah_attr.dlid,attr.dest_qp_num,attr.sq_psn);
          if (err=ibv_modify_qp(ret->qp, &attr,
                IBV_QP_STATE              |
                IBV_QP_TIMEOUT            |
                IBV_QP_RETRY_CNT          |
                IBV_QP_RNR_RETRY          |
                IBV_QP_SQ_PSN             |
                IBV_QP_MAX_QP_RD_ATOMIC)) {
            MACHSTATE1(3,"ERROR changing qp state to RTS %d",err);
            CmiAbort("Failed to change qp state to RTS: you may need some device-specific parameters in machine-ibverbs");
          }

        } else if(err) {
          CmiAbort("Failed to change qp state to RTS");
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
#ifdef NON_SRQ
// This is resulting in the total recv buffers to grow as the number of nodes. What could be the alternative? May be adaptively increase number of buffers for the most communicating nodes. Need a mechanism for such flow control, existing does not claim to work.
	minPerProcessorRecvs = 10;
	if(minPerProcessorRecvs*(_Cmi_numnodes-1) <= maxRecvBuffers){
		numPosts = minPerProcessorRecvs*(_Cmi_numnodes-1);
	}
#endif
//        numPosts=1000; 
	if(numPosts > 0){
		context->recvBufferPool = allocateInfiBufferPool(numPosts,packetSize);
		postInitialRecvs(context->recvBufferPool,numPosts,packetSize);
	}


	if (context->qp) {
          free(context->qp);
	  context->qp = NULL;
	}
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
#if CMK_IBVERBS_STATS
	numCurReg++;
	numReg++;
#endif

	CmiAssert(bigKey != NULL);
	
	for(i=0;i<numBuffers;i++){
		struct infiBuffer *buffer =  &(ret->buffers[i]);
		buffer->type = BUFFER_RECV;
		buffer->size = sizePerBuffer;
		

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
#ifndef NON_SRQ
	workRequests[numRecvs-1].next = NULL;
	MACHSTATE(3,"About to call ibv_post_srq_recv");
	if(ibv_post_srq_recv(context->srq,workRequests,&bad_wr)){
		CmiEnforce(0);
	}
#else 
// create a pool per processor and post initial receives to processor queue similar to send, split the buffer pool Equi-partitioning recv pool
       { 
        int myNode;
	int numNodes;
        int perNodeRecvs,k,i,n;
	numNodes = _Cmi_numnodes;
	myNode = _Cmi_mynode;
  	perNodeRecvs = numRecvs/(numNodes-1);
	k =0;
	for( i=1;i<numNodes;i++){
                n = (myNode + i)%numNodes;
		if(n  != myNode){ 
				if (k==numNodes-2) 

					workRequests[numRecvs-1].next = NULL;
				else
					workRequests[(k+1)*perNodeRecvs-1].next = NULL;
				if(ibv_post_recv(context->qp[n],&workRequests[k*perNodeRecvs],&bad_wr)){CmiEnforce(0);}
				k++;
				}
          }
        }
#endif

	free(workRequests);
	free(sgElements);
}




static inline void CommunicationServer_nolock(int toBuffer); //if buffer ==1 recvd messages are buffered but not processed

void CmiMachineExit()
{
#if CMK_IBVERBS_STATS	
	printf("[%d] numReg %d numUnReg %d numCurReg %d msgCount %d pktCount %d packetSize %d total Time %.6lf s processBufferedCount %d processBufferedTime %.6lf s maxTokens %d tokensLeft %d \n",_Cmi_mynode,numReg, numUnReg, numCurReg, msgCount,pktCount,packetSize,CmiTimer(),processBufferedCount,processBufferedTime,maxTokens,context->tokensLeft);
#endif
}

void CmiMachineCleanup(){
	MACHSTATE(3, "CmiMachineCleanup")
	int num_devices;
	struct ibv_device **devList;
	ibv_dealloc_pd(context->pd);
	ibv_close_device(context->context);
	devList = ibv_get_device_list(&num_devices);
	ibv_free_device_list(devList);
	MACHSTATE(3, "CmiMachineCleanup END")
}
static void ServiceCharmrun_nolock();

static void CmiNotifyStillIdle(CmiIdleState *s) {
#if CMK_SMP
	CmiCommLock();
	inProgress[CmiMyRank()] += 1;
/*	if(where == COMM_SERVER_FROM_SMP)*/
#endif
/*		ServiceCharmrun_nolock();*/

	CommunicationServer_nolock(0);
#if CMK_SMP
	CmiCommUnlock();
	inProgress[CmiMyRank()] -= 1;
#endif
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
	struct ibv_send_wr *bad_wr=NULL;
#if	CMK_IBVERBS_DEBUG
	packet->header.psn = (++node->infiData->psn);
#endif	



	packet->elemList[1].addr = (uintptr_t)packet->buf;
	packet->elemList[1].length = size;
	packet->elemList[1].lkey = dataKey->lkey;
	
	
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
		CmiEnforce(0);
	}*/

	if(retval = ibv_post_send(node->infiData->qp,&(packet->wr),&bad_wr)){
		//CmiPrintf("[%d] Sending to node %d failed with return value %d\n",_Cmi_mynode,node->infiData->nodeNo,retval);
                CmiAbort("Sending to node failed\n");
	}
#if	CMK_IBVERBS_TOKENS_FLOW
	context->tokensLeft--;
#if 	CMK_IBVERBS_STATS
	if(context->tokensLeft < minTokensLeft){
		minTokensLeft = context->tokensLeft;
	}
#endif
#endif

/*	if(!checkQp(node->infiData->qp)){
		pollSendCq(1);
		CmiEnforce(0);
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
	struct ibv_mr *key;
	infiPacket packet;
	MallocInfiPacket(packet);
	packet->size = size;
	packet->buf = CmiAlloc(size);
	
	packet->header.code = INFIDUMMYPACKET;

	key = METADATAFIELD(packet->buf)->key;
	
	MACHSTATE2(3,"Dummy packet to %d size %d",node->infiData->nodeNo,size);
	EnqueuePacket(node,packet,size,key);
}






static void inline EnqueueDataPacket(OutgoingMsg ogm, OtherNode node, int rank,char *data,int size,int broot,int copy){
	struct ibv_mr *key;
	infiPacket packet;
	MallocInfiPacket(packet);
	packet->size = size;
	packet->buf=data;
	
	//the nodeNo is added at time of packet allocation
	packet->header.code = INFIPACKETCODE_DATA;
	
	ogm->refcount++;
	packet->ogm = ogm;
	
	key = METADATAFIELD(ogm->data)->key;
	CmiAssert(key != NULL);
	
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
//#if	!CMK_SMP
	processAllBufferedMsgs();
//#endif
	ogm->refcount--;
	MACHSTATE3(3,"DONE Sending ogm %p of size %d to %d",ogm,ogm->size,node->infiData->nodeNo);
}


static inline void EnqueueRdmaPacket(OutgoingMsg ogm, OtherNode node){
	infiPacket packet;

	ogm->refcount++;
	
	MallocInfiPacket(packet);
 
 {
		struct infiRdmaPacket *rdmaPacket = (struct infiRdmaPacket *)CmiAlloc(sizeof(struct infiRdmaPacket));
		struct ibv_mr *key;
		struct ibv_mr *packetKey;

		
		packet->size = sizeof(struct infiRdmaPacket);
		packet->buf = (char *)rdmaPacket;
		
		key = METADATAFIELD(ogm->data)->key;

		CmiAssert(key!=NULL);

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
		
		
		packetKey = METADATAFIELD((void *)rdmaPacket)->key;
		
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
	CmiAssert(FD_ISSET(context->context->async_fd,&context->asyncFds));
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

static void pollCmiDirectQ();

static inline  void CommunicationServer_nolock(int toBuffer) {
	int processed;
	if(_Cmi_numnodes <= 1){
		pollCmiDirectQ();
		return;
	}
	MACHSTATE(2,"CommServer_nolock{");
	
//	processAsyncEvents();
	
//	checkAllQps();

	pollCmiDirectQ();

	processed = pollRecvCq(toBuffer);
	

	processed += pollSendCq(toBuffer);
	
	if(toBuffer == 0){
//		if(processed != 0)
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
//	CmiAssert(ne >=0);
	
	if(ne != 0){
		MACHSTATE1(3,"pollRecvCq ne %d",ne);
	}
	
	for(i=0;i<ne;i++){
		if(wc[i].status != IBV_WC_SUCCESS){
			CmiEnforce(0);
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
//	CmiAssert(ne >=0);
	
	
	for(i=0;i<ne;i++){
		if(wc[i].status != IBV_WC_SUCCESS){
			printf("[%d] wc[%d] status %d wc[i].opcode %d\n",_Cmi_mynode,i,wc[i].status,wc[i].opcode);
#if CMK_IBVERBS_STATS
	printf("[%d] msgCount %d pktCount %d packetSize %d total Time %.6lf s processBufferedCount %d processBufferedTime %.6lf s maxTokens %d tokensLeft %d minTokensLeft %d \n",_Cmi_mynode,msgCount,pktCount,packetSize,CmiTimer(),processBufferedCount,processBufferedTime,maxTokens,context->tokensLeft,minTokensLeft);
#endif
			CmiEnforce(0);
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
			case IBV_WC_RDMA_WRITE:
			{
				/*** used for CmiDirect puts 
				Nothing needs to be done on the sender side once send is done **/
				break;
			}
			default:
				CmiAbort("Wrong type of work completion object in recvq");
				break;
		}
			
	}
	return ne;
}


/******************
Check the communication server socket and

*****************/
int CheckSocketsReady(int withDelayMs)
{   
  int nreadable;
  CMK_PIPE_DECL(withDelayMs);

  CmiStdoutAdd(CMK_PIPE_SUB);
  if (Cmi_charmrun_fd!=-1) CMK_PIPE_ADDREAD(Cmi_charmrun_fd);

  nreadable=CMK_PIPE_CALL();
  ctrlskt_ready_read = 0;
  dataskt_ready_read = 0;
  dataskt_ready_write = 0;
  
  if (nreadable == 0) {
    MACHSTATE(1,"} CheckSocketsReady (nothing readable)")
    return nreadable;
  }
  if (nreadable==-1) {
    CMK_PIPE_CHECKERR();
    MACHSTATE(2,"} CheckSocketsReady (INTERRUPTED!)")
    return CheckSocketsReady(0);
  }
  CmiStdoutCheck(CMK_PIPE_SUB);
  if (Cmi_charmrun_fd!=-1) 
          ctrlskt_ready_read = CMK_PIPE_CHECKREAD(Cmi_charmrun_fd);
  MACHSTATE(1,"} CheckSocketsReady")
  return nreadable;
}


/*** Service the charmrun socket
*************/

static void ServiceCharmrun_nolock()
{
  int again = 1;
  MACHSTATE(2,"ServiceCharmrun_nolock begin {")
  while (again)
  {
  again = 0;
  CheckSocketsReady(0);
  if (ctrlskt_ready_read) { ctrl_getone(); again=1; }
  if (CmiStdoutNeedsService()) { CmiStdoutService(); }
  }
  MACHSTATE(2,"} ServiceCharmrun_nolock end")
}



static void CommunicationServer(int sleepTime, int where){
	if( where == COMM_SERVER_FROM_INTERRUPT){
#if CMK_IMMEDIATE_MSG
		CmiHandleImmediate();
#endif
		return;
	}
#if CMK_SMP
	if(where == COMM_SERVER_FROM_WORKER){
		return;
	}
	CmiCommLock();
	inProgress[CmiMyRank()] += 1;
	if(where == COMM_SERVER_FROM_SMP){
#endif
	        ServiceCharmrun_nolock();
#if CMK_SMP
	}
#endif
	CommunicationServer_nolock(0);
#if CMK_SMP
	CmiCommUnlock();
	inProgress[CmiMyRank()] -= 1;
#endif

	/* when called by communication thread or in interrupt */
#if CMK_IMMEDIATE_MSG
	if (where == COMM_SERVER_FROM_SMP) {
		CmiHandleImmediate();
	}
#endif
}


static void insertBufferedBcast(char *msg,int size,int broot,int asm_rank);


void static inline handoverMessage(char *newmsg,int total_size,int rank,int broot,int toBuffer);

static inline void processMessage(int nodeNo,int len,char *msg,const int toBuffer){
	char *newmsg;
	OtherNode node = &nodes[nodeNo];
	MACHSTATE2(3,"Processing packet from node %d len %d",nodeNo,len);
	
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
				//CmiPrintf("size: %d, len:%d.\n", size, len);
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
				//CmiPrintf("from node %d asm_total: %d, asm_fill: %d, len:%d.\n",node->infiData->nodeNo, node->asm_total, node->asm_fill, len);
				CmiAbort("packet in the middle does not have expected length");
			}
			if(node->asm_fill+len > node->asm_total){
				//CmiPrintf("asm_total: %d, asm_fill: %d, len:%d.\n", node->asm_total, node->asm_fill, len);
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
		 	insertBufferedBcast(CopyMsg(newmsg,total_size),total_size,broot,rank);
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
		 	insertBufferedBcast(CopyMsg(newmsg,total_size),total_size,broot,rank);
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
//#if !CMK_SMP		
		processAllBufferedMsgs();
//#endif
		}
}


static inline void increasePostedRecvs(int nodeNo);
static inline void processRdmaRequest(struct infiRdmaPacket *rdmaPacket,int fromNodeNo,int isBuffered);
static inline void processRdmaAck(struct infiRdmaPacket *rdmaPacket);

//struct infiDirectRequestPacket;
//static inline void processDirectRequest(struct infiDirectRequestPacket *directRequestPacket);

static inline void processRecvWC(struct ibv_wc *recvWC,const int toBuffer){
	struct infiBuffer *buffer = (struct infiBuffer *) recvWC->wr_id;	
	struct infiPacketHeader *header = (struct infiPacketHeader *)buffer->buf;
	int nodeNo = header->nodeNo;
#if	CMK_IBVERBS_DEBUG
	OtherNode node = &nodes[nodeNo];
#endif
	
	int len = recvWC->byte_len-sizeof(struct infiPacketHeader);
#if	CMK_IBVERBS_DEBUG
	/*
	if(node->infiData->recvPsn == 0){
		node->infiData->recvPsn = header->psn;
	}else{
	  		CmiAssert(header->psn == (node->infiData->recvPsn)+1);
	   *	   	node->infiData->recvPsn++; 
	}
	*/
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
	if(header->code & INFIBARRIERPACKET){
                MACHSTATE(3,"Barrier packet");
                CmiAbort("Should not receive Barrier packet in normal polling loop.  Your Barrier is broken");
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
/*	if(header->code & INFIDIRECT_REQUEST){
		struct infiDirectRequestPacket *directRequestPacket = (struct infiDirectRequestPacket *)(buffer->buf+sizeof(struct infiPacketHeader));
		processDirectRequest(directRequestPacket);
	}*/
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
#ifndef NON_SRQ	
		if(ibv_post_srq_recv(context->srq,&wr,&bad_wr))
#else
		OtherNode node1 = &nodes[nodeNo];
		if(ibv_post_recv(node1->infiData->qp,&wr,&bad_wr))
#endif 
		{
			CmiEnforce(0);
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
                   if (packet->buf) CmiFree(packet->buf);  /* gzheng */
		}
	}

	FreeInfiPacket(packet);
};



/********************************************************************/
static inline void processRdmaRequest(struct infiRdmaPacket *_rdmaPacket,int fromNodeNo,int isBuffered){
	int nodeNo = fromNodeNo;
	OtherNode node = &nodes[nodeNo];
	struct infiRdmaPacket *rdmaPacket;
	struct infiBuffer *buffer = malloc(sizeof(struct infiBuffer));

	getFreeTokens(node->infiData);
#if CMK_IBVERBS_TOKENS_FLOW
//	node->infiData->tokensLeft--;
	context->tokensLeft--;
#if 	CMK_IBVERBS_STATS
	if(context->tokensLeft < minTokensLeft){
		minTokensLeft = context->tokensLeft;
	}
#endif
#endif
	
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
			CmiEnforce(0);
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
/*	if(rdmaPacket->type == INFI_DIRECT){
		processDirectWC(rdmaPacket);
		return;
	}*/
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
	
	//OtherNode node=&nodes[rdmaPacket->fromNodeNo];
	//we are sending this ack as a response to a successful
	// rdma_Read.. the token for that rdma_Read needs to be freed
#if CMK_IBVERBS_TOKENS_FLOW	
	//node->infiData->tokensLeft++;
	context->tokensLeft++;
#endif

	//send ack to sender if toBuffer is off otherwise buffer it
	if(toBuffer){
		struct infiRdmaPacket *tmp = context->bufferedRdmaAcks;
		MACHSTATE1(3,"Buffering Rdma Ack %p",rdmaPacket);
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
		struct ibv_mr *packetKey;
		*ackPacket = *rdmaPacket;
		packet->size = sizeof(struct infiRdmaPacket);
		packet->buf = (char *)ackPacket;
		packet->header.code = INFIRDMA_ACK;
		packet->ogm=NULL;
		
		packetKey = METADATAFIELD((void *)ackPacket)->key;
	

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
		CmiFree(start->bcastList[i].msg);           /* gzheng */
					}
#elif CMK_BROADCAST_HYPERCUBE
        if (start->bcastList[i].asm_rank == DGRAM_BROADCAST
#if CMK_NODE_QUEUE_AVAILABLE
          || start->bcastList[i].asm_rank == DGRAM_NODEBROADCAST
#endif
         ){
          	SendHypercube(NULL, 0,start->bcastList[i].size,start->bcastList[i].msg ,start->bcastList[i].broot,start->bcastList[i].asm_rank);
		CmiFree(start->bcastList[i].msg);           /* gzheng */
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
        int currentCqSize;
	if(node->infiData->totalTokens + increase > maxTokens){
		increase = maxTokens-node->infiData->totalTokens;
	}
	node->infiData->totalTokens += increase;
	node->infiData->tokensLeft += increase;
	MACHSTATE3(3,"Increasing tokens for node %d to %d by %d",node->infiData->nodeNo,node->infiData->totalTokens,increase);
	//increase the size of the sendCq
	currentCqSize = context->sendCqSize;
	if(ibv_resize_cq(context->sendCq,currentCqSize+increase)){
		fprintf(stderr,"[%d] failed to increase cq by %d from %d totalTokens %d \n",_Cmi_mynode,increase,currentCqSize, node->infiData->totalTokens);
		CmiEnforce(0);
	}
	context->sendCqSize+= increase;
};
// Should not be used with NON-SRQ (at this time I am not sure INCTOKEN  works, at the top it says never turn it on), so I am not modifying these to make them work with Non-SRQ version
static void increasePostedRecvs(int nodeNo){
	OtherNode node = &nodes[nodeNo];
	int tokenIncrease = node->infiData->postedRecvs*INCTOKENS_INCREASE;	
	int recvIncrease = tokenIncrease;
	int currentCqSize;
	if(tokenIncrease+node->infiData->postedRecvs > maxTokens){
		tokenIncrease = maxTokens - node->infiData->postedRecvs;
	}
#ifndef NON_SRQ
	if(tokenIncrease+context->srqSize > maxRecvBuffers){
		recvIncrease = maxRecvBuffers-context->srqSize;
	}
	node->infiData->postedRecvs+= recvIncrease;
	context->srqSize += recvIncrease;
#endif
	MACHSTATE3(3,"Increase tokens by %d to %d for node %d ",tokenIncrease,node->infiData->postedRecvs,nodeNo);
	//increase the size of the recvCq
	currentCqSize = context->recvCqSize;
	if(ibv_resize_cq(context->recvCq,currentCqSize+tokenIncrease)){
		CmiEnforce(0);
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
	int i,j;
	int size = firstBinSize;
	int nodeSize;

#if THREAD_MULTI_POOL
	nodeSize = CmiMyNodeSize() + 1;
	infiCmiChunkPools = malloc(sizeof(infiCmiChunkPool *) * nodeSize);
	for(i = 0; i < nodeSize; i++){
		infiCmiChunkPools[i] = malloc(sizeof(infiCmiChunkPool) * INFINUMPOOLS);
	}
	for(j = 0; j < nodeSize; j++){
		size = firstBinSize;
		for(i=0;i<INFINUMPOOLS;i++){
			infiCmiChunkPools[j][i].size = size;
			infiCmiChunkPools[j][i].startBuf = NULL;
			infiCmiChunkPools[j][i].count = 0;
			size *= 2;
		}
	}

	// creating the n^2 system of queues
	queuePool = malloc(sizeof(PCQueue *) * nodeSize);
	for(i = 0; i < nodeSize; i++){
		queuePool[i] = malloc(sizeof(PCQueue) * nodeSize);
	}
	for(i = 0; i < nodeSize; i++)
		for(j = 0; j < nodeSize; j++)
			queuePool[i][j] = PCQueueCreate();

#else

	size = firstBinSize;	
	for(i=0;i<INFINUMPOOLS;i++){
		infiCmiChunkPools[i].size = size;
		infiCmiChunkPools[i].startBuf = NULL;
		infiCmiChunkPools[i].count = 0;
		size *= 2;
	}
#endif

}

/***********
Register memory for a part of a received multisend message
*************/
infiCmiChunkMetaData *registerMultiSendMesg(char *msg,int size){
	infiCmiChunkMetaData *metaData = malloc(sizeof(infiCmiChunkMetaData));
	char *res=msg-sizeof(infiCmiChunkHeader);
	metaData->key = ibv_reg_mr(context->pd,res,(size+sizeof(infiCmiChunkHeader)),IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
#if CMK_IBVERBS_STATS
	numCurReg++;
	numReg++;
	numMultiSend++;
#endif
	CmiAssert(metaData->key!=NULL);
	metaData->owner = NULL;
	metaData->poolIdx = INFIMULTIPOOL;

	return metaData;
};


#if THREAD_MULTI_POOL

// Fills up the buffer pools for every thread in the node
static inline void fillBufferPools(){
	int nodeSize, poolIdx, thread;
	infiCmiChunkMetaData *metaData;		
	infiCmiChunkHeader *hdr;
	int allocSize;
	int count=1;
	int i;
	struct ibv_mr *key;
	void *res;

	// initializing values
	nodeSize = CmiMyNodeSize() + 1;

	// iterating over all threads and all pools
	for(thread = 0; thread < nodeSize; thread++){
		for(poolIdx = 0; poolIdx < INFINUMPOOLS; poolIdx++){
			allocSize = infiCmiChunkPools[thread][poolIdx].size;
			if(poolIdx < blockThreshold){
				count = blockAllocRatio;
			}else{
				count = 1;
			}
                        posix_memalign(&res, ALIGN_BYTES, (allocSize+sizeof(infiCmiChunkHeader))*count);
			hdr = res;
			key = ibv_reg_mr(context->pd,res,(allocSize+sizeof(infiCmiChunkHeader))*count,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
			CmiAssert(key != NULL);
#if CMK_IBVERBS_STATS
		numCurReg++;
		numReg++;
#endif
			res += sizeof(infiCmiChunkHeader);
			for(i=0;i<count;i++){
				metaData = METADATAFIELD(res) = malloc(sizeof(infiCmiChunkMetaData));
				metaData->key = key;
				metaData->owner = hdr;
				metaData->poolIdx = poolIdx;
				metaData->parentPe = thread;						// setting the parent PE
				if(i == 0){
					metaData->owner->metaData->count = count;
					metaData->nextBuf = NULL;
                                        infiCmiChunkPools[thread][poolIdx].startBuf =  res - sizeof(infiCmiChunkHeader);
                                        infiCmiChunkPools[thread][poolIdx].count++;
				}else{
					void *startBuf = res - sizeof(infiCmiChunkHeader);
					metaData->nextBuf = infiCmiChunkPools[thread][poolIdx].startBuf;
					infiCmiChunkPools[thread][poolIdx].startBuf = startBuf;
					infiCmiChunkPools[thread][poolIdx].count++;
				}
				if(i != count-1){
					res += (allocSize+sizeof(infiCmiChunkHeader));
				}
			}
		}
	}	
}

static inline void *getInfiCmiChunkThread(int dataSize){
	//find out to which pool this dataSize belongs to
	// poolIdx = floor(log2(dataSize/firstBinSize))+1
	int ratio = dataSize/firstBinSize;
	int poolIdx=0;
	void *res;
	int i,j,nodeSize;
	void *pointer;

	//printf("Hi\n");
	MACHSTATE1(2,"Rank=%d",CmiMyRank());
	MACHSTATE1(3,"INFI_ALLOC %d",CmiMyRank());
	
	while(ratio > 0){
		ratio  = ratio >> 1;
		poolIdx++;
	}
	MACHSTATE1(2,"This is %d",CmiMyRank());
	MACHSTATE2(2,"getInfiCmiChunk for size %d in poolIdx %d",dataSize,poolIdx);

	// checking whether to analyze the free queues to reuse buffers	
	nodeSize = CmiMyNodeSize() + 1;
	if(poolIdx < INFINUMPOOLS && infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf == NULL){
		MACHSTATE1(3,"Disposing memory %d",CmiMyRank());
		for(i = 0; i < nodeSize; i++){
			if(!PCQueueEmpty(queuePool[CmiMyRank()][i])){
				for(j = 0; j < PCQueueLength(queuePool[CmiMyRank()][i]); j++){
					pointer = (void *)PCQueuePop(queuePool[CmiMyRank()][i]);
					infi_CmiFreeDirect(pointer);	
				}
			}
		}	
	}

	if((poolIdx < INFINUMPOOLS && infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf == NULL) || poolIdx >= INFINUMPOOLS){
		infiCmiChunkMetaData *metaData;		
		infiCmiChunkHeader *hdr;
		int allocSize;
		int count=1;
		int i;
		struct ibv_mr *key;
		void *origres;
		
		
		if(poolIdx < INFINUMPOOLS ){
			allocSize = infiCmiChunkPools[CmiMyRank()][poolIdx].size;
		}else{
			allocSize = dataSize;
		}

		if(poolIdx < blockThreshold){
			count = blockAllocRatio;
		}
                posix_memalign(&res, ALIGN_BYTES, (allocSize+sizeof(infiCmiChunkHeader))*count);
		_MEMCHECK(res);
		hdr = res;
		
		key = ibv_reg_mr(context->pd,res,(allocSize+sizeof(infiCmiChunkHeader))*count,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
		CmiAssert(key != NULL);
#if CMK_IBVERBS_STATS
		numCurReg++;
		numReg++;
#endif
		
		origres = (res += sizeof(infiCmiChunkHeader));

		for(i=0;i<count;i++){
			metaData = METADATAFIELD(res) = malloc(sizeof(infiCmiChunkMetaData));
			_MEMCHECK(metaData);
			metaData->key = key;
			metaData->owner = hdr;
			metaData->poolIdx = poolIdx;
			metaData->parentPe = CmiMyRank();						// setting the parent PE

			if(i == 0){
				metaData->owner->metaData->count = count;
				metaData->nextBuf = NULL;
			}else{
				void *startBuf = res - sizeof(infiCmiChunkHeader);
				metaData->nextBuf = infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf;
				infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf = startBuf;
				infiCmiChunkPools[CmiMyRank()][poolIdx].count++;
				
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
	
		res = infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf;
		res += sizeof(infiCmiChunkHeader);

		MACHSTATE2(2,"Reusing old pool %d buf %p",poolIdx,res);
		metaData = METADATAFIELD(res);

		infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf = metaData->nextBuf;
		MACHSTATE2(1,"Pool %d now has startBuf at %p",poolIdx,infiCmiChunkPools[CmiMyRank()][poolIdx].startBuf);
		
		metaData->nextBuf = NULL;
//		CmiAssert(metaData->poolIdx == poolIdx);

		infiCmiChunkPools[CmiMyRank()][poolIdx].count--;
		return res;
	}

	CmiEnforce(0);

	
};
#else /* not MULTIPOOL case */
static inline void *getInfiCmiChunk(int dataSize){
        //find out to which pool this dataSize belongs to
        // poolIdx = floor(log2(dataSize/firstBinSize))+1
        int ratio = dataSize/firstBinSize;
        int poolIdx=0;
        char *res;
#if CMK_IBVERBS_STATS
	if(numAlloc>10000 && numAlloc%1000==0)
	  {
	  printf("[%d] numReg %d numUnReg %d numCurReg %d numAlloc %d numFree %d msgCount %d pktCount %d packetSize %d total Time %.6lf s processBufferedCount %d processBufferedTime %.6lf s maxTokens %d tokensLeft %d \n",_Cmi_mynode,numReg, numUnReg, numCurReg, numAlloc, numFree, msgCount,pktCount,packetSize,CmiTimer(),processBufferedCount,processBufferedTime,maxTokens,context->tokensLeft);
	  /*	  printf("[%d]  numMultiSendUnreg %d numMultiSend %d  numMultiSendFree %d\n",_Cmi_mynode, numMultiSendUnreg, numMultiSend, numMultiSendFree);*/
	  }
#endif
        while(ratio > 0){
                ratio  = ratio >> 1;
                poolIdx++;
        }
        MACHSTATE2(2,"getInfiCmiChunk for size %d in poolIdx %d",dataSize,poolIdx);
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
                posix_memalign(&res, ALIGN_BYTES, (allocSize+sizeof(infiCmiChunkHeader))*count);
                hdr = (infiCmiChunkHeader *)res;

                key = ibv_reg_mr(context->pd,res,(allocSize+sizeof(infiCmiChunkHeader))*count,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                CmiAssert(key != NULL);
#if CMK_IBVERBS_STATS
		numCurReg++;
		numReg++;
#endif
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


                MACHSTATE3(2,"AllocSize %d buf %p key %p",allocSize,res,metaData->key);

                return origres;
        }
        if(poolIdx < INFINUMPOOLS){
                infiCmiChunkMetaData *metaData;

                res = infiCmiChunkPools[poolIdx].startBuf;
                res += sizeof(infiCmiChunkHeader);

                MACHSTATE2(2,"Reusing old pool %d buf %p",poolIdx,res);
                metaData = METADATAFIELD(res);

                infiCmiChunkPools[poolIdx].startBuf = metaData->nextBuf;
                MACHSTATE2(1,"Pool %d now has startBuf at %p",poolIdx,infiCmiChunkPools[poolIdx].startBuf);

                metaData->nextBuf = NULL;
//              CmiAssert(metaData->poolIdx == poolIdx);

                infiCmiChunkPools[poolIdx].count--;
                return res;
        }

        CmiEnforce(0);


};
#endif


void * infi_CmiAlloc(int size){
	char *res;
#if CMK_IBVERBS_STATS
	numAlloc++;
#endif
        if (Cmi_charmrun_fd == -1) {
          posix_memalign(&res, ALIGN_BYTES, size + sizeof(void*));
          res += sizeof(void*);
          return res;
        }
#if THREAD_MULTI_POOL
	res = getInfiCmiChunkThread(size-sizeof(CmiChunkHeader));
	res -= sizeof(CmiChunkHeader);

	return res;
#else
#if CMK_SMP
	CmiMemLock();
#endif
/*(	if(size-sizeof(CmiChunkHeader) > firstBinSize){*/
		MACHSTATE1(1,"infi_CmiAlloc for dataSize %d",size-sizeof(CmiChunkHeader));

		res = (char*)getInfiCmiChunk(size-sizeof(CmiChunkHeader));	
		res -= sizeof(CmiChunkHeader);
#if CMK_SMP	
	CmiMemUnlock();
#endif
/*	}else{
		res = malloc(size);
	}*/
	
	return res;
#endif
}

#if THREAD_MULTI_POOL
//Note: this function receives a pointer to the data, so that it is not necessary to add any sizeof(CmiChunkHeader) to it.
void infi_CmiFreeDirect(void *ptr){
        int size;
        int parentPe;
        void *freePtr = ptr;
        infiCmiChunkMetaData *metaData;
        int poolIdx;
        infiCmiChunkPool *pool;
#if CMK_IBVERBS_STATS
	numFree++;
#endif

        //ptr += sizeof(CmiChunkHeader);
        size = SIZEFIELD (ptr);
/*      if(size > firstBinSize){*/
        //there is a infiniband specific header
        freePtr = ptr - sizeof(infiCmiChunkHeader);
        metaData = METADATAFIELD(ptr);
        poolIdx = metaData->poolIdx;
	pool = infiCmiChunkPools[CmiMyRank()] + poolIdx;
        MACHSTATE2(1,"CmiFree buf %p goes back to pool %d",ptr,poolIdx);
//      CmiAssert(poolIdx >= 0);
	if(poolIdx < INFINUMPOOLS && pool->count < INFIMAXPERPOOL &&
	   pool->count < ((1 << INFINUMPOOLS) >> poolIdx) ){
	  metaData->nextBuf = pool->startBuf;
	  pool->startBuf = freePtr;
	  pool->count++;
	  MACHSTATE3(2,"Pool %d now has startBuf at %p count %d",poolIdx,pool->startBuf,pool->count);
	}else{
	  MACHSTATE2(2,"Freeing up buf %p poolIdx %d",ptr,poolIdx);
	  metaData->owner->metaData->count--;
	  if(metaData->owner->metaData == metaData){
	    //I am the owner
	    if(metaData->owner->metaData->count == 0){
	      //all the chunks have been freed
	      int unregstat=ibv_dereg_mr(metaData->key);
#if CMK_IBVERBS_STATS
	      numUnReg++;
	      numCurReg--;
#endif

	      CmiAssert(unregstat==0);
	      free(freePtr);
	      free(metaData);
	    }
	    //if I am the owner and all the chunks have not been
	    // freed dont free my metaData. will need later
	  }else{
	    if(metaData->owner->metaData->count == 0){
              int unregstat;
	      //need to free the owner's buffer and metadata
	      freePtr = metaData->owner;
	      unregstat=ibv_dereg_mr(metaData->key);
#if CMK_IBVERBS_STATS
	      numUnReg++;
	      numCurReg--;
#endif

	      CmiAssert(unregstat==0);
	      free(metaData->owner->metaData);
	      free(freePtr);
	    }
	    free(metaData);
	  }
	}
}


void infi_CmiFree(void *ptr){

	int i,j;
        int size;
	int parentPe;
	int nodeSize;
	void *pointer;
        void *freePtr = ptr;
	infiCmiChunkMetaData *metaData;
        int poolIdx;
	nodeSize = CmiMyNodeSize() + 1;

	MACHSTATE(3,"Freeing");

        if (Cmi_charmrun_fd == -1) { char *res = ptr; res -= sizeof(void*); free(res); return; }
        ptr += sizeof(CmiChunkHeader);
        size = SIZEFIELD (ptr);
/*      if(size > firstBinSize){*/
        //there is a infiniband specific header
        freePtr = ptr - sizeof(infiCmiChunkHeader);
        metaData = METADATAFIELD(ptr);
        poolIdx = metaData->poolIdx;

        if(poolIdx == INFIMULTIPOOL){
        	/** this is a part of a received mult message  
                    it will be freed correctly later
                **/
#if CMK_IBVERBS_STATS
	  numMultiSendFree++;
#endif
	  return;
        }


	// checking if this free operation is my responsibility
	parentPe = metaData->parentPe;
	if(parentPe != CmiMyRank()){
		PCQueuePush(queuePool[parentPe][CmiMyRank()],(char *)ptr);
		return;
	}


	infi_CmiFreeDirect(ptr);

}

#else
void infi_CmiFree(void *ptr){
	int size;
	void *freePtr = ptr;
	int poolIdx;
	infiCmiChunkMetaData *metaData;
#if CMK_IBVERBS_STATS
	numFree++;
#endif
	
        if (Cmi_charmrun_fd == -1) { char *res = ptr; res -= sizeof(void*); free(res); return; }
#if CMK_SMP	
	CmiMemLock();
#endif
	ptr += sizeof(CmiChunkHeader);
	size = SIZEFIELD (ptr);
/*	if(size > firstBinSize){*/
		//there is a infiniband specific header
		freePtr = (char*)ptr - sizeof(infiCmiChunkHeader);
		metaData = METADATAFIELD(ptr);
		poolIdx = metaData->poolIdx;
		if(poolIdx == INFIMULTIPOOL){
			/** this is a part of a received mult message  
			it will be freed correctly later
			**/
#if CMK_IBVERBS_STATS
		        numMultiSendFree++;
#endif
			return;
		}
		MACHSTATE2(1,"CmiFree buf %p goes back to pool %d",ptr,poolIdx);
//		CmiAssert(poolIdx >= 0);
		if(poolIdx < INFINUMPOOLS &&
		   infiCmiChunkPools[poolIdx].count <= INFIMAXPERPOOL &&
		   infiCmiChunkPools[poolIdx].count < ((1 << INFINUMPOOLS) >> poolIdx) ){
		  metaData->nextBuf = infiCmiChunkPools[poolIdx].startBuf;
		  infiCmiChunkPools[poolIdx].startBuf = freePtr;
		  infiCmiChunkPools[poolIdx].count++;
			
		  MACHSTATE3(2,"Pool %d now has startBuf at %p count %d",poolIdx,infiCmiChunkPools[poolIdx].startBuf,infiCmiChunkPools[poolIdx].count);
		}else{
			MACHSTATE2(2,"Freeing up buf %p poolIdx %d",ptr,poolIdx);
			metaData->owner->metaData->count--;
			if(metaData->owner->metaData == metaData){
				//I am the owner
				if(metaData->owner->metaData->count == 0){
					//all the chunks have been freed
					int unregstat=ibv_dereg_mr(metaData->key);
#if CMK_IBVERBS_STATS
                                        numUnReg++;
                                        numCurReg--;
#endif

					CmiAssert(unregstat==0);
					free(freePtr);
					free(metaData);
				}
				//if I am the owner and all the chunks have not been
				// freed dont free my metaData. will need later
			}else{
				if(metaData->owner->metaData->count == 0){
					//need to free the owner's buffer and metadata
					int unregstat=ibv_dereg_mr(metaData->key);
					freePtr = metaData->owner;
#if CMK_IBVERBS_STATS
                                        numUnReg++;
                                        numCurReg--;
#endif

					CmiAssert(unregstat==0);
					free(metaData->owner->metaData);
					free(freePtr);
				}
				free(metaData);
			}
		}	
#if CMK_SMP	
	CmiMemUnlock();
#endif
/*	}else{
		free(freePtr);
	}*/
}
#endif

/*********************************************************************************************
This section is for CmiDirect. This is a variant of the  persistent communication in which 
the user can transfer data between processors without using Charm++ messages. This lets the user
send and receive data from the middle of his arrays without any copying on either send or receive
side
*********************************************************************************************/

struct infiDirectRequestPacket{
	int senderProc;
	int handle;
	struct ibv_mr senderKey;
	void *senderBuf;
	int senderBufSize;
};

#include "cmidirect.h"

#define MAXHANDLES 512

struct infiDirectHandleStruct;


typedef struct directPollingQNodeStruct {
	struct infiDirectHandleStruct *handle;
	struct directPollingQNodeStruct *next;
	double *lastDouble;
} directPollingQNode;

typedef struct infiDirectHandleStruct{
	int id;
	void *buf;
	int size;
	struct ibv_mr *key;
	void (*callbackFnPtr)(void *);
	void *callbackData;
//	struct infiDirectRequestPacket *packet;
	struct infiDirectUserHandle userHandle;
	struct infiRdmaPacket *rdmaPacket;
	directPollingQNode pollingQNode;
}	infiDirectHandle;

typedef struct infiDirectHandleTableStruct{
	infiDirectHandle handles[MAXHANDLES];
	struct infiDirectHandleTableStruct *next;
} infiDirectHandleTable;


// data structures 

directPollingQNode *headDirectPollingQ=NULL,*tailDirectPollingQ=NULL;

static infiDirectHandleTable **sendHandleTable=NULL;
static infiDirectHandleTable **recvHandleTable=NULL;

static int *recvHandleCount=NULL;

void addHandleToPollingQ(infiDirectHandle *handle){
//	directPollingQNode *newNode = malloc(sizeof(directPollingQNode));
	directPollingQNode *newNode = &(handle->pollingQNode);
	newNode->handle = handle;
	newNode->next = NULL;
	if(headDirectPollingQ==NULL){
		/*empty pollingQ*/
		headDirectPollingQ = newNode;
		tailDirectPollingQ = newNode;
	}else{
		tailDirectPollingQ->next = newNode;
		tailDirectPollingQ = newNode;
	}
};
/*
infiDirectHandle *removeHandleFromPollingQ(){
	if(headDirectPollingQ == NULL){
		//polling Q is empty
		return NULL;
	}
	directPollingQNode *retNode = headDirectPollingQ;
	if(headDirectPollingQ == tailDirectPollingQ){
		//PollingQ has one node 
		headDirectPollingQ = tailDirectPollingQ = NULL;
	}else{
		headDirectPollingQ = headDirectPollingQ->next;
	}
	infiDirectHandle *retHandle = retNode->handle;
	free(retNode);
	return retHandle;
}*/

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

static inline void initializeLastDouble(void *recvBuf,int recvBufSize,double initialValue)
{
	/** initialize the last double in the buffer to bufize***/
	int index = recvBufSize - sizeof(double);
	double *lastDouble = (double *)(((char *)recvBuf)+index);
	*lastDouble = initialValue;
}


/**
 To be called on the receiver to create a handle and return its number
**/
struct infiDirectUserHandle CmiDirect_createHandle(int senderNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue){
	int newHandle;
	int tableIdx,idx;
	int i;
	infiDirectHandleTable *table;
	struct infiDirectUserHandle userHandle;
	
	CmiAssert(recvBufSize > sizeof(double));
	
	if(recvHandleTable == NULL){
		recvHandleTable = createHandleTable();
		recvHandleCount = malloc(sizeof(int)*_Cmi_numnodes);
		for(i=0;i<_Cmi_numnodes;i++){
			recvHandleCount[i] = -1;
		}
	}
	if(recvHandleTable[senderNode] == NULL){
		recvHandleTable[senderNode] = malloc(sizeof(infiDirectHandleTable));
		recvHandleTable[senderNode]->next = NULL;		
	}
	
	newHandle = ++recvHandleCount[senderNode];
	CmiAssert(newHandle >= 0);
	
	calcHandleTableIdx(newHandle,&tableIdx,&idx);
	
	table = recvHandleTable[senderNode];
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
#if CMI_DIRECT_DEBUG
	CmiPrintf("[%d] RDMA create addr %p %d sizeof(struct ibv_mr) %d\n",CmiMyNode(),table->handles[idx].buf,recvBufSize,sizeof(struct ibv_mr));
#endif
	table->handles[idx].callbackFnPtr = callbackFnPtr;
	table->handles[idx].callbackData = callbackData;
	table->handles[idx].key = ibv_reg_mr(context->pd, recvBuf, recvBufSize,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	CmiAssert(table->handles[idx].key != NULL);
#if CMK_IBVERBS_STATS
	numCurReg++;
	numReg++;
#endif
/*	table->handles[idx].rdmaPacket = CmiAlloc(sizeof(struct infiRdmaPacket));
	table->handles[idx].rdmaPacket->type = INFI_DIRECT;
	table->handles[idx].rdmaPacket->localBuffer = &(table->handles[idx]);*/
	
	userHandle.handle = newHandle;
	userHandle.recverNode = _Cmi_mynode;
	userHandle.senderNode = senderNode;
	userHandle.recverBuf = recvBuf;
	userHandle.recverBufSize = recvBufSize;
	memcpy(userHandle.recverKey,table->handles[idx].key,sizeof(struct ibv_mr));
	userHandle.initialValue = initialValue;
	
	table->handles[idx].userHandle = userHandle;
	
	initializeLastDouble(recvBuf,recvBufSize,initialValue);

  {
	 int index = table->handles[idx].size - sizeof(double);
   table->handles[idx].pollingQNode.lastDouble = (double *)(((char *)table->handles[idx].buf)+index);
	} 
	
	addHandleToPollingQ(&(table->handles[idx]));
	
//	MACHSTATE4(3," Newhandle created %d senderProc %d recvBuf %p recvBufSize %d",newHandle,senderProc,recvBuf,recvBufSize);
	
	return userHandle;
}

/****
 To be called on the sender to attach the sender's buffer to this handle
******/
void CmiDirect_assocLocalBuffer(struct infiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize){
	int tableIdx,idx;
	int i;
	int handle = userHandle->handle;
	int recverNode  = userHandle->recverNode;

	infiDirectHandleTable *table;

	if(sendHandleTable == NULL){
		sendHandleTable = createHandleTable();
	}
	if(sendHandleTable[recverNode] == NULL){
		sendHandleTable[recverNode] = malloc(sizeof(infiDirectHandleTable));
		sendHandleTable[recverNode]->next = NULL;
	}
	
	CmiAssert(handle >= 0);
	calcHandleTableIdx(handle,&tableIdx,&idx);
	
	table = sendHandleTable[recverNode];
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
#if CMI_DIRECT_DEBUG
	CmiPrintf("[%d] RDMA assoc addr %p %d remote addr %p \n",CmiMyPe(),table->handles[idx].buf,sendBufSize,userHandle->recverBuf);
#endif
	table->handles[idx].callbackFnPtr = NULL;
	table->handles[idx].callbackData = NULL;
	table->handles[idx].key =  ibv_reg_mr(context->pd, sendBuf, sendBufSize,IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	CmiAssert(table->handles[idx].key != NULL);
#if CMK_IBVERBS_STATS
	numCurReg++;
	numReg++;
#endif
	table->handles[idx].userHandle = *userHandle;
	CmiAssert(sendBufSize == table->handles[idx].userHandle.recverBufSize);
	
	table->handles[idx].rdmaPacket = CmiAlloc(sizeof(struct infiRdmaPacket));
	table->handles[idx].rdmaPacket->type = INFI_DIRECT;
	table->handles[idx].rdmaPacket->localBuffer = &(table->handles[idx]);
	
	
/*	table->handles[idx].packet = (struct infiDirectRequestPacket *)CmiAlloc(sizeof(struct infiDirectRequestPacket));
	table->handles[idx].packet->senderProc = _Cmi_mynode;
	table->handles[idx].packet->handle = handle;
	table->handles[idx].packet->senderKey = *(table->handles[idx].key);
	table->handles[idx].packet->senderBuf = sendBuf;
	table->handles[idx].packet->senderBufSize = sendBufSize;*/
	
	MACHSTATE4(3,"idx %d recverProc %d handle %d sendBuf %p",idx,recverNode,handle,sendBuf);
};





/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(struct infiDirectUserHandle *userHandle){
	int handle = userHandle->handle;
	int recverNode = userHandle->recverNode;
	if(recverNode == _Cmi_mynode){
		/*when the sender and receiver are on the same
		processor, just look up the sender and receiver
		buffers and do a memcpy*/

		infiDirectHandleTable *senderTable;
		infiDirectHandleTable *recverTable;
		
		int tableIdx,idx,i;

		
		/*find entry for this handle in sender table*/
		calcHandleTableIdx(handle,&tableIdx,&idx);
		CmiAssert(sendHandleTable!= NULL);
		senderTable = sendHandleTable[_Cmi_mynode];
		CmiAssert(senderTable != NULL);
		for(i=0;i<tableIdx;i++){
			senderTable = senderTable->next;
		}

		/**find entry for this handle in recver table*/
		recverTable = recvHandleTable[recverNode];
		CmiAssert(recverTable != NULL);
		for(i=0;i< tableIdx;i++){
			recverTable = recverTable->next;
		}
		
		CmiAssert(senderTable->handles[idx].size == recverTable->handles[idx].size);
		memcpy(recverTable->handles[idx].buf,senderTable->handles[idx].buf,senderTable->handles[idx].size);
#if CMI_DIRECT_DEBUG
		CmiPrintf("[%d] RDMA memcpy put addr %p receiver %p, size %d\n",CmiMyPe(),senderTable->handles[idx].buf,recverTable->handles[idx].buf,senderTable->handles[idx].size);
#endif
		// The polling Q should find you and handle the callback and pollingq entry
		//		(*(recverTable->handles[idx].callbackFnPtr))(recverTable->handles[idx].callbackData);
		

	}else{
		infiPacket packet;
		int tableIdx,idx;
		int i;
		OtherNode node;
		infiDirectHandleTable *table;

		calcHandleTableIdx(handle,&tableIdx,&idx);

		table = sendHandleTable[recverNode];
		CmiAssert(table != NULL);
		for(i=0;i<tableIdx;i++){
			table = table->next;
		}

//		MACHSTATE2(3,"CmiDirect_put to recverProc %d handle %d",recverProc,handle);
#if CMI_DIRECT_DEBUG
		CmiPrintf("[%d] RDMA put addr %p\n",CmiMyPe(),table->handles[idx].buf);
#endif

		
		{
			
			OtherNode node = &nodes[table->handles[idx].userHandle.recverNode];
			struct ibv_sge list = {
				.addr = (uintptr_t )table->handles[idx].buf,
				.length = table->handles[idx].size,
				.lkey 	= table->handles[idx].key->lkey
			};
			
			struct ibv_mr *remoteKey = (struct ibv_mr *)table->handles[idx].userHandle.recverKey;

			struct ibv_send_wr *bad_wr;
			struct ibv_send_wr wr = {
				.wr_id = (uint64_t)table->handles[idx].rdmaPacket,
				.sg_list = &list,
				.num_sge = 1,
				.opcode = IBV_WR_RDMA_WRITE,
				.send_flags = IBV_SEND_SIGNALED,
				
				.wr.rdma = {
					.remote_addr = (uint64_t )table->handles[idx].userHandle.recverBuf,
					.rkey = remoteKey->rkey
				}
			};
			/** post and rdma_read that is a rdma get*/
			if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
				CmiEnforce(0);
			}
		}

	/*	MallocInfiPacket (packet);
		{
			packet->size = sizeof(struct infiDirectRequestPacket);
			packet->buf = (char *)(table->handles[idx].packet);
			struct ibv_mr *packetKey = METADATAFIELD((void *)table->handles[idx].packet)->key;
			EnqueuePacket(node,packet,sizeof(struct infiDirectRequestPacket),packetKey);
		}*/
	}

};

/**** need not be called the first time *********/
void CmiDirect_readyMark(struct infiDirectUserHandle *userHandle){
  initializeLastDouble(userHandle->recverBuf,userHandle->recverBufSize,userHandle->initialValue);
}

/**** need not be called the first time *********/
void CmiDirect_readyPollQ(struct infiDirectUserHandle *userHandle){
	int handle = userHandle->handle;
	int tableIdx,idx,i;
	infiDirectHandleTable *table;
	calcHandleTableIdx(handle,&tableIdx,&idx);
	
	table = recvHandleTable[userHandle->senderNode];
	CmiAssert(table != NULL);
	for(i=0;i<tableIdx;i++){
		table = table->next;
	}
#if CMI_DIRECT_DEBUG
  CmiPrintf("[%d] CmiDirect_ready receiver %p\n",CmiMyNode(),userHandle->recverBuf);
#endif	
	addHandleToPollingQ(&(table->handles[idx]));
	

}

/**** need not be called the first time *********/
void CmiDirect_ready(struct infiDirectUserHandle *userHandle){
	int handle = userHandle->handle;
	int tableIdx,idx,i;
	infiDirectHandleTable *table;
	
	initializeLastDouble(userHandle->recverBuf,userHandle->recverBufSize,userHandle->initialValue);

	calcHandleTableIdx(handle,&tableIdx,&idx);
	
	table = recvHandleTable[userHandle->senderNode];
	CmiAssert(table != NULL);
	for(i=0;i<tableIdx;i++){
		table = table->next;
	}
#if CMI_DIRECT_DEBUG
  CmiPrintf("[%d] CmiDirect_ready receiver %p\n",CmiMyNode(),userHandle->recverBuf);
#endif	
	addHandleToPollingQ(&(table->handles[idx]));
	
}


static int receivedDirectMessage(infiDirectHandle *handle){
//	int index = handle->size - sizeof(double);
//	double *lastDouble = (double *)(((char *)handle->buf)+index);
	if(*(handle->pollingQNode.lastDouble) == handle->userHandle.initialValue){
		return 0;
	}else{
		(*(handle->callbackFnPtr))(handle->callbackData);	
		return 1;
	}
	
}


static void pollCmiDirectQ(){
	directPollingQNode *ptr = headDirectPollingQ, *prevPtr=NULL;
	while(ptr != NULL){
		if(receivedDirectMessage(ptr->handle)){
#if CMI_DIRECT_DEBUG
      CmiPrintf("[%d] polling detected recvd message at buf %p\n",CmiMyNode(),ptr->handle->userHandle.recverBuf);
#endif
			directPollingQNode *delPtr = ptr;
			/** has been received and delete this node***/
			if(prevPtr == NULL){
				/** first in the pollingQ**/
				if(headDirectPollingQ == tailDirectPollingQ){
					/**only node in pollingQ****/
					headDirectPollingQ = tailDirectPollingQ = NULL;
				}else{
					headDirectPollingQ = headDirectPollingQ->next;
				}
			}else{
				if(ptr == tailDirectPollingQ){
					/**last node is being deleted**/
					tailDirectPollingQ = prevPtr;
				}
				prevPtr->next = ptr->next;
			}
			ptr = ptr->next;
		//	free(delPtr);
		}else{
			prevPtr = ptr;
			ptr = ptr->next;
		}
	}
}


/*void processDirectRequest(struct infiDirectRequestPacket *directRequestPacket){
	int senderProc = directRequestPacket->senderProc;
	int handle = directRequestPacket->handle;
	int tableIdx,idx,i;
	infiDirectHandleTable *table;
	OtherNode node = nodes_by_pe[senderProc];

	MACHSTATE2(3,"processDirectRequest from proc %d handle %d",senderProc,handle);

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
//	 post and rdma_read that is a rdma get
		if(ibv_post_send(node->infiData->qp,&wr,&bad_wr)){
			CmiEnforce(0);
		}
	}
			
	
};*/
/*
void processDirectWC(struct infiRdmaPacket *rdmaPacket){
	MACHSTATE(3,"processDirectWC");
	infiDirectHandle *handle = (infiDirectHandle *)rdmaPacket->localBuffer;
	(*(handle->callbackFnPtr))(handle->callbackData);
};
*/

#if 0

// use the common one

static void sendBarrierMessage(int pe)
{
  /* we will only need one packet */
  int size=32;
  OtherNode  node = nodes + pe;
  infiPacket packet;
  MallocInfiPacket(packet);
  packet->size = size;
  packet->buf = CmiAlloc(size);
  packet->header.code = INFIBARRIERPACKET;
  struct ibv_mr *key = METADATAFIELD(packet->buf)->key;
  MACHSTATE2(3,"Barrier packet to %d size %d",node->infiData->nodeNo,size);
  /*  pollSendCq(0);*/
  EnqueuePacket(node,packet,size,key);
}

static void recvBarrierMessage()
{
  int i;
  int ne;
  /*  struct ibv_wc wc[WC_LIST_SIZE];*/
  struct ibv_wc wc[1];
  struct ibv_wc *recvWC;
  /* block on the recvq, this is lazy and evil in the general case because we abuse buffers but should be ok for startup barriers */
  int toBuffer=1; // buffer without processing recvd messages
  int barrierReached=0;
  struct infiBuffer *buffer = NULL;
  struct infiPacketHeader *header = NULL;
  int nodeNo=-1;
  int len=-1;
  while(!barrierReached)
    {
      /* gengbin's semantic will implode if more than one q is polled at a time */
      ne = ibv_poll_cq(context->recvCq,1,&wc[0]);
      //	CmiAssert(ne >=0);
      if(ne != 0){
	MACHSTATE1(3,"recvBarrier ne %d",ne);
      }
      pollSendCq(1); 
      for(i=0;i<ne;i++){
	if(wc[i].status != IBV_WC_SUCCESS){
	  CmiEnforce(0);
	}
	switch(wc[i].opcode){
	case IBV_WC_RECV: /* we have something to consider*/
	  recvWC=&wc[i];
	  buffer = (struct infiBuffer *) recvWC->wr_id;	
	  header = (struct infiPacketHeader *)buffer->buf;
	  nodeNo = header->nodeNo;
	  len = recvWC->byte_len-sizeof(struct infiPacketHeader);

	  if(header->code & INFIPACKETCODE_DATA){
	    processMessage(nodeNo,len,(buffer->buf+sizeof(struct infiPacketHeader)),toBuffer);
	  }
	  if(header->code & INFIDUMMYPACKET){
	    MACHSTATE(3,"Dummy packet");
	  }
	  if(header->code & INFIBARRIERPACKET){
	    MACHSTATE2(3,"Barrier packet from node %d len %d",nodeNo,len);	
	    // now we are done
	    barrierReached=1;
	    /* semantically questionable */
	    //processAllBufferedMsgs();
	    //return;
	  }
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
	      CmiEnforce(0);
	    }
	  }

	  break;
	default:
	  CmiAbort("Wrong type of work completion object in recvq");
	  break;
	}
      }
    }
  /* semantically questionable */
  //  processAllBufferedMsgs();
}


/* happen at node level */
int CmiBarrier()
{
  int len, size, i;
  int status;
  int count = 0;
  OtherNode node;
  int numnodes = CmiNumNodes();
  if (CmiMyRank() == 0) {
    /* every one send to pe 0 */
    if (CmiMyNode() != 0) {
      sendBarrierMessage(0);
    }
    /* printf("[%d] HERE\n", CmiMyPe()); */
    if (CmiMyNode() == 0) 
    {
      for (count = 1; count < numnodes; count ++) 
      {
        recvBarrierMessage();
      }
      /* pe 0 broadcast */
      for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = i;
        if (p > numnodes - 1) break;
        /* printf("[%d] BD => %d \n", CmiMyPe(), p); */
        sendBarrierMessage(p);
      }
    }
    /* non 0 node waiting */
    if (CmiMyNode() != 0) 
    {
      recvBarrierMessage();
      for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int p = CmiMyNode();
        p = BROADCAST_SPANNING_FACTOR*p + i;
        if (p > numnodes - 1) break;
        p = p%numnodes;
        /* printf("[%d] RELAY => %d \n", CmiMyPe(), p); */
        sendBarrierMessage(p);
      }
    }
  }
  CmiNodeAllBarrier();
  processAllBufferedMsgs();
  /* printf("[%d] OUT of barrier \n", CmiMyPe()); */
}

/* everyone sends a message to pe 0 and go on */
int CmiBarrierZero()
{
  int i;

  if (CmiMyRank() == 0) {
    if (CmiMyNode()) {
      sendBarrierMessage(0);
    }
    else {
      for (i=0; i<CmiNumNodes()-1; i++)
      {
        recvBarrierMessage();
      }
    }
  }
  CmiNodeAllBarrier();
  processAllBufferedMsgs();
}


#endif
