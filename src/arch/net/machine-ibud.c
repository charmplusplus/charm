/** @file
 * ibverbs unreliable datagram implementation of Converse NET version
 * @ingroup NET
 * contains only ibverbs specific code for:
 * - CmiMachineInit()
 * - CmiCommunicationInit()
 * - CmiNotifyIdle()
 * - DeliverViaNetwork()
 * - CommunicationServer()

   Eric Shook and Esteban Meneses - Jul 22, 2008
*/

/**
 * @addtogroup NET
 * @{
 */

/** NOTES
- Every message sent using the unreliable layer of infiniband must include the GRH (Global Routing Header). The GRH are the first 40 bytes of every packet and the machine layer has no other responsibility over it than reserving the space in the packet.
*/


// FIXME: Note: Charm does not guarantee in order messages - can use for bettter performance


#include <infiniband/verbs.h>

#define WC_LIST_SIZE 32

#define INFIPACKETCODE_DATA 1
#define INFIDUMMYPACKET 64
#define INFIBARRIERPACKET 128

#define METADATAFIELD(m) (((infiCmiChunkHeader *)m)[-1].metaData)


enum ibv_mtu mtu = IBV_MTU_2048;
static int mtu_size;
static int maxrecvbuffers;
static int maxtokens;
static int firstBinSize;
static int blockThreshold;
static int blockAllocRatio;
static int packetsize;


struct ibudstruct {
	struct ibv_device **devlist;
	struct ibv_device *dev;
};

struct ibudstruct ibud;

struct infiPacketHeader{
	char code;
	int nodeNo;
};

/** Represents a qp used to send messages to another node
 There is one for each remote node */
struct infiAddr {
	int lid,qpn,psn;
};

typedef struct infiPacketStruct {
	char *buf;
	int size;
	char extra[40];			// FIXME: check this 40 extra stuff
	struct infiPacketHeader header;
	struct ibv_mr *keyHeader;
	struct OtherNodeStruct *destNode;
	struct infiPacketStruct *next;
	OutgoingMsg ogm;
	struct ibv_sge elemList[2];
	struct ibv_send_wr wr;
}* infiPacket;

struct infiContext {
	struct ibv_context	*context;
	
	fd_set  asyncFds;
	struct timeval tmo;
	
	int ibPort;
	struct ibv_pd		*pd;
	struct ibv_cq		*sendCq;
	struct ibv_cq   	*recvCq;
	struct ibv_srq  	*srq;
	struct ibv_mr		*mr;
	struct ibv_ah		**ah;

	struct ibv_qp		*qp;

	struct infiAddr localAddr; //store the lid,qpn,msn address of ur qpair until they are sent

    struct infiBufferPool *recvBufferPool;

	infiPacket infiPacketFreeList;

	struct infiPacketHeader header;
	int sendCqSize,recvCqSize;

	void *buffer; // Registered memory buffer for msg's

};

static struct infiContext *context;



struct infiOtherNodeData{
	int state;// does it expect a packet with a header (first packet) or one without
	int totalTokens;
	int tokensLeft;
	int nodeNo;

	int postedRecvs;
	int broot;//needed to store the root of a multi-packet broadcast sent along a spanning tree or hypercube
	struct infiAddr qp;
};

enum { INFI_HEADER_DATA=21,INFI_DATA};

typedef struct {
  int sleepMs; /*Milliseconds to sleep while idle*/
  int nIdles;  /*Number of times we've been idle in a row*/
  CmiState cs; /*Machine state*/
} CmiIdleState;

#define BUFFER_RECV 1
struct infiBuffer{
	int type;
	char *buf;
	int size;
	struct ibv_mr *key;
};

// FIXME: This is defined in converse.h for ibverbs
//struct infiCmiChunkMetaDataStruct;

typedef struct infiCmiChunkMetaDataStruct {
        struct ibv_mr *key;
        int poolIdx;
        void *nextBuf;
        struct infiCmiChunkHeaderStruct *owner;
        int count;
} infiCmiChunkMetaData;

struct infiBufferPool{
    int numBuffers;
    struct infiBuffer *buffers;
    struct infiBufferPool *next;
};


/*
typedef struct infiCmiChunkHeaderStruct{
	struct infiCmiChunkMetaDataStruct *metaData;
	CmiChunkHeader chunkHeader;
} infiCmiChunkHeader;

struct infiCmiChunkMetaDataStruct *registerMultiSendMesg(char *msg,int msgSize);
*/

// FIXME: temp for error reading
static const char *const __ibv_wc_status_str[] = {
   "Success",
   "Local Length Error",
   "Local QP Operation Error",
   "Local EE Context Operation Error",
   "Local Protection Error",
   "Work Request Flushed Error",
   "Memory Management Operation Error",
   "Bad Response Error",
   "Local Access Error",
   "Remote Invalid Request Error",
   "Remote Access Error",
   "Remote Operation Error",
   "Transport Retry Counter Exceeded",
   "RNR Retry Counter Exceeded",
   "Local RDD Violation Error",
   "Remote Invalid RD Request",
   "Aborted Error",
   "Invalid EE Context Number",
   "Invalid EE Context State",
   "Fatal Error",
   "Response Timeout Error",
   "General Error"
};
const char *ibv_wc_status_str(enum ibv_wc_status status) {
   if (status < IBV_WC_SUCCESS || status > IBV_WC_GENERAL_ERR)
       status = IBV_WC_GENERAL_ERR;
   return (__ibv_wc_status_str[status]);
}

/***** BEGIN MEMORY MANAGEMENT STUFF *****/
typedef struct {
        int size;//without infiCmiChunkHeader
        void *startBuf;
        int count;
} infiCmiChunkPool;


#define INFIMULTIPOOL -5
#define INFINUMPOOLS 20
#define INFIMAXPERPOOL 100

infiCmiChunkPool infiCmiChunkPools[INFINUMPOOLS];

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

void infi_unregAndFreeMeta(void *md) {
	if(md!=NULL && (((infiCmiChunkMetaData *)md)->poolIdx == INFIMULTIPOOL)) {
		ibv_dereg_mr(((infiCmiChunkMetaData*)md)->key);
		free(((infiCmiChunkMetaData *)md));
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
//                        CmiAssert(allocSize>=dataSize); // FIXME: added this assertion
                }else{
                        allocSize = dataSize;
                }

                if(poolIdx < blockThreshold){
                        count = blockAllocRatio;
                }
                res = malloc((allocSize+sizeof(infiCmiChunkHeader))*count);
                hdr = res;

                key = ibv_reg_mr(context->pd,res,(allocSize+sizeof(infiCmiChunkHeader))*count,IBV_ACCESS_LOCAL_WRITE);
                CmiAssert(key != NULL);

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

        CmiAssert(0);
}

void * infi_CmiAlloc(int size){
	void *res;

	#if CMK_SMP
	CmiMemLock();
	#endif
	MACHSTATE1(1,"infi_CmiAlloc for dataSize %d",size-sizeof(CmiChunkHeader));

	res = getInfiCmiChunk(size-sizeof(CmiChunkHeader));
	res -= sizeof(CmiChunkHeader);
	#if CMK_SMP     
	CmiMemUnlock();
	#endif
	return res;
}


void infi_CmiFree(void *ptr){
	int size;
	void *freePtr = ptr;
	infiCmiChunkMetaData *metaData;
	int poolIdx;

	#if CMK_SMP     
	CmiMemLock();
	#endif
	ptr += sizeof(CmiChunkHeader);
	size = SIZEFIELD (ptr);
	//there is a infiniband specific header
	freePtr = ptr - sizeof(infiCmiChunkHeader);
	metaData = METADATAFIELD(ptr);
	poolIdx = metaData->poolIdx;
	if(poolIdx == INFIMULTIPOOL){
		/** this is a part of a received mult message  
		it will be freed correctly later **/
		return;
	}
	MACHSTATE2(1,"CmiFree buf %p goes back to pool %d",ptr,poolIdx);
	if(poolIdx < INFINUMPOOLS && infiCmiChunkPools[poolIdx].count <= INFIMAXPERPOOL){
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
				ibv_dereg_mr(metaData->key);
				free(freePtr);
				free(metaData);
			}
			//if I am the owner and all the chunks have not been
			// freed dont free my metaData. will need later
		}else {
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
	#if CMK_SMP
	CmiMemUnlock();
	#endif
}


static void initInfiCmiChunkPools(){
    int i,j;
    int size = firstBinSize;
    int nodeSize;

    size = firstBinSize;    
    for(i=0;i<INFINUMPOOLS;i++){
        infiCmiChunkPools[i].size = size;
        infiCmiChunkPools[i].startBuf = NULL;
        infiCmiChunkPools[i].count = 0;
        size *= 2;
    }

} 
  


/***** END MEMORY MANAGEMENT STUFF *****/



//     Post the buffers as recv work requests
void postInitialRecvs(struct infiBufferPool *recvBufferPool,int numRecvs,int sizePerBuffer){
    int j,err;
    struct ibv_recv_wr *workRequests = malloc(sizeof(struct ibv_recv_wr)*numRecvs);
    struct ibv_sge *sgElements = malloc(sizeof(struct ibv_sge)*numRecvs);
    struct ibv_recv_wr *bad_wr;

    int startBufferIdx=0;
    MACHSTATE2(3,"posting %d receives of size %d",numRecvs,sizePerBuffer);
    for(j=0;j<numRecvs;j++){
        sgElements[j].addr = (uint64_t) recvBufferPool->buffers[startBufferIdx+j].buf;
        sgElements[j].length = sizePerBuffer + 40;						// we add the 40 bytes of the GRH
        sgElements[j].lkey = recvBufferPool->buffers[startBufferIdx+j].key->lkey;
        workRequests[j].wr_id = (uint64_t)&(recvBufferPool->buffers[startBufferIdx+j]);
        workRequests[j].sg_list = &sgElements[j];
        workRequests[j].num_sge = 1;
        if(j != numRecvs-1){
            workRequests[j].next = &workRequests[j+1];
        }
    }
    workRequests[numRecvs-1].next = NULL;
    MACHSTATE(3,"About to call ibv_post_recv");
    CmiAssert(ibv_post_recv(context->qp,workRequests,&bad_wr)==0); 

    free(workRequests);
    free(sgElements);

}

struct infiBufferPool * allocateInfiBufferPool(int numRecvs,int sizePerBuffer){
	int numBuffers;
	int i;
	int bigSize;
	char *bigBuf;
	struct infiBufferPool *ret;
	struct ibv_mr *bigKey;
	int page_size;

	MACHSTATE2(3,"allocateInfiBufferPool numRecvs %d sizePerBuffer%d ",numRecvs,sizePerBuffer);

	page_size = sysconf(_SC_PAGESIZE);
	ret = malloc(sizeof(struct infiBufferPool));
	ret->next = NULL;
	numBuffers=ret->numBuffers = numRecvs;
	ret->buffers = malloc(sizeof(struct infiBuffer)*numBuffers);
	bigSize = numBuffers*sizePerBuffer;
	bigBuf = memalign(page_size,bigSize);
	bigKey = ibv_reg_mr(context->pd,bigBuf,bigSize,IBV_ACCESS_LOCAL_WRITE);
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



void infiPostInitialRecvs(){
	//create the pool and post the receives 
	int numPosts;
    
	// we add 40 to the buffer size to handle administrative information
	context->recvBufferPool = allocateInfiBufferPool(maxrecvbuffers, packetsize + 40); 	// we add 40 bytes to hold the GRH of Infiniband 
	postInitialRecvs(context->recvBufferPool,maxrecvbuffers,packetsize);

}   

/*
*/




static CmiIdleState *CmiNotifyGetState(void) { 
	return NULL; 
}

static void CmiNotifyStillIdle(CmiIdleState *s); 
static void CmiNotifyBeginIdle(CmiIdleState *s) {
  CmiNotifyStillIdle(s);
}


static inline  void CommunicationServer_lock(int toBuffer);
static inline  void CommunicationServer_nolock(int toBuffer);

static void CmiNotifyStillIdle(CmiIdleState *s) { 
#if CMK_SMP
        CommunicationServer_lock(0);
#else
        CommunicationServer_nolock(0);
#endif
}

/******************************************************************************
 *
 * CmiNotifyIdle()-- wait until a packet comes in
 *
 *****************************************************************************/

void CmiNotifyIdle(void) {
  CmiNotifyStillIdle(NULL);
}

/****************************************************************************
 *                                                                          
 * CheckSocketsReady
 *
 * Checks both sockets to see which are readable and which are writeable.
 * We check all these things at the same time since this can be done for
 * free with ``select.'' The result is stored in global variables, since
 * this is essentially global state information and several routines need it.
 *
 ***************************************************************************/

int CheckSocketsReady(int withDelayMs)
{   
  int nreadable,dataWrite=writeableDgrams || writeableAcks;
  CMK_PIPE_DECL(withDelayMs);


#if CMK_USE_KQUEUE && 0
  // This implementation doesn't yet work, but potentially is much faster

  /* Only setup the CMK_PIPE structures the first time they are used. 
     This makes the kqueue implementation much faster.
  */
  static int first = 1;
  if(first){
    first = 0;
    CmiStdoutAdd(CMK_PIPE_SUB);
    if (Cmi_charmrun_fd!=-1) { CMK_PIPE_ADDREAD(Cmi_charmrun_fd); }
    else return 0; /* If there's no charmrun, none of this matters. */
    if (dataskt!=-1) {
      CMK_PIPE_ADDREAD(dataskt); 
      CMK_PIPE_ADDWRITE(dataskt);
    }
  }

#else  
  CmiStdoutAdd(CMK_PIPE_SUB);  
  if (Cmi_charmrun_fd!=-1) { CMK_PIPE_ADDREAD(Cmi_charmrun_fd); }  
  else return 0; /* If there's no charmrun, none of this matters. */  
  if (dataskt!=-1) {  
    { CMK_PIPE_ADDREAD(dataskt); }  
    if (dataWrite)  
      CMK_PIPE_ADDWRITE(dataskt);  
  }  
#endif 

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
  if (dataskt!=-1) {
	dataskt_ready_read = CMK_PIPE_CHECKREAD(dataskt);
	if (dataWrite)
		dataskt_ready_write = CMK_PIPE_CHECKWRITE(dataskt);
  }
  return nreadable;
}



static inline infiPacket newPacket(){
	infiPacket pkt=(infiPacket )CmiAlloc(sizeof(struct infiPacketStruct));

	pkt->size = -1;
	pkt->header = context->header;
	pkt->next = NULL;
	pkt->destNode = NULL;
	pkt->keyHeader = METADATAFIELD(pkt)->key;
	pkt->ogm=NULL;
	CmiAssert(pkt->keyHeader!=NULL);
	
	pkt->elemList[0].addr = (uintptr_t)&(pkt->header);
	pkt->elemList[0].length = sizeof(struct infiPacketHeader);
	pkt->elemList[0].lkey = pkt->keyHeader->lkey;
	
	pkt->wr.wr_id = (uint64_t)pkt; 
	pkt->wr.sg_list = &(pkt->elemList[0]);
	pkt->wr.num_sge = 2; //FIXME: should be 2 here
	pkt->wr.opcode = IBV_WR_SEND;
	pkt->wr.send_flags = IBV_SEND_SIGNALED;
	pkt->wr.next = NULL;

	return pkt;
};

static void inline EnqueuePacket(OtherNode node,infiPacket packet,int size,struct ibv_mr *dataKey){
/*
struct ibv_send_wr wr,*bad_wr=NULL;
    struct ibv_sge list;
    void *buffer;
    struct ibv_mr *mr;
    int pe=node->infiData->nodeNo;
    int retval;

buffer=malloc(128);
mr=ibv_reg_mr(context->pd, buffer, 128, IBV_ACCESS_LOCAL_WRITE);

    //memset(&list, 0, sizeof(struct ibv_sge));
    list.addr = (uintptr_t) buffer + 40;
    list.length = 128;
    list.lkey = mr->lkey;

        memset(&wr, 0, sizeof(struct ibv_send_wr));
        //wr.wr_id = 1234;
        wr.wr_id = (uint64_t)packet;
        wr.sg_list = &list;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        //wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_SOLICITED;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.ud.ah = context->ah[pe]; 
        wr.wr.ud.remote_qpn = nodes[pe].infiData->qp.qpn; 
        wr.wr.ud.remote_qkey = 0;

        MACHSTATE3(3,"   wr_id=%i qp_num=%i lkey=%p",wr.wr_id,wr.wr.ud.remote_qpn,mr->lkey); 

	if(retval = ibv_post_send(context->qp,&wr,&bad_wr)){ 
		CmiPrintf("[%d] Sending to node %d failed with return value %d\n",_Cmi_mynode,node->infiData->nodeNo,retval);
		CmiAssert(0);
    }
        ibv_dereg_mr(mr);
        free(buffer);
*/

	int retval;
	struct ibv_send_wr *bad_wr=NULL;
    MACHSTATE(2," here");

// FIXME: these were originally [1], but I don't know why 
	packet->elemList[1].addr = (uintptr_t)packet->buf; // FIXME: It works if I add 40 here
	packet->elemList[1].length = size;
	packet->elemList[1].lkey = dataKey->lkey;
    MACHSTATE(2," here");

	packet->destNode = node;
	
    MACHSTATE(2," here1");
    MACHSTATE1(2," here qp=%i",context->qp);
    MACHSTATE1(2," here wr=%i",&(packet->wr));
    MACHSTATE1(2," here wr=%i",&bad_wr);


	if(ibv_post_send(context->qp,&(packet->wr),&bad_wr)){ 
		MACHSTATE(2," problem sending");
		CmiPrintf("[%d] Sending to node %d failed with return value %d\n",_Cmi_mynode,node->infiData->nodeNo,retval);
		CmiAssert(0);
	}
    MACHSTATE(2," here");
    MACHSTATE2(3,"Packet send size %d node %d ",size,packet->destNode->infiData->nodeNo);
    MACHSTATE2(2,"            addr %p lkey %p ",(uintptr_t)packet->buf,dataKey->lkey);

}

static void inline EnqueueDataPacket(OutgoingMsg ogm, char *data, int size, OtherNode node, int rank, int broot) {
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
}




/***********************************************************************
 * DeliverViaNetwork()
 *
 * This function is responsible for all non-local transmission. This
 * function takes the outgoing messages, splits it into datagrams and
 * enqueues them into the Send Queue.
 ***********************************************************************/
void DeliverViaNetwork(OutgoingMsg ogm, OtherNode node, int rank, unsigned int broot, int copy) {
	int size; char *data;
        MACHSTATE(3,"DeliverViaNetwork");
	size=ogm->size;
	data=ogm->data;
	DgramHeaderMake(data, rank, ogm->src, Cmi_charmrun_pid, 1, broot); // May not be needed
	CmiMsgHeaderSetLength(data,size);
	while(size>Cmi_dgram_max_data) {
		EnqueueDataPacket(ogm, data, Cmi_dgram_max_data, node, rank, broot);
		size -= Cmi_dgram_max_data;
		data += Cmi_dgram_max_data;

	}
	if(size>0)
		EnqueueDataPacket(ogm, data, size, node, rank, broot);

}


static void ServiceCharmrun_nolock() {
	int again = 1;
	MACHSTATE(2,"ServiceCharmrun_nolock begin {")
	while (again) {
		again = 0;
		CheckSocketsReady(0);
		if (ctrlskt_ready_read) { 	// FIXME: this is set in another call 
			ctrl_getone();
			again=1; 
		}
		if (CmiStdoutNeedsService())
			CmiStdoutService();
	}
	MACHSTATE(2,"} ServiceCharmrun_nolock end")
}


void static inline handoverMessage(char *newmsg,int total_size,int rank,int broot,int toBuffer){
	#if CMK_BROADCAST_SPANNING_TREE | CMK_BROADCAST_HYPERCUBE
	if (rank == DGRAM_BROADCAST
		#if CMK_NODE_QUEUE_AVAILABLE
		|| rank == DGRAM_NODEBROADCAST
		#endif
		){
		if(toBuffer){
			insertBufferedBcast(CopyMsg(newmsg,total_size),total_size,broot,rank);
		}else{
			#if CMK_BROADCAST_SPANNING_TREE
			SendSpanningChildren(NULL, 0, total_size, newmsg,broot,rank);
			#else
			SendHypercube(NULL, 0, total_size, newmsg,broot,rank);
			#endif
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
		default: {
			CmiPushPE(rank, newmsg);
		}
	}    /* end of switch */
//	if(!toBuffer){
//		processAllBufferedMsgs();
//	}
}



static inline void processMessage(int nodeNo,int len,char *msg,const int toBuffer){
        char *newmsg;
        OtherNode node = &nodes[nodeNo];
        newmsg = node->asm_msg;

	MACHSTATE2(3,"Processing packet from node %d len %d",nodeNo,len);

	switch(node->infiData->state){
		case INFI_HEADER_DATA: {
			int size;
			int rank, srcpe, seqno, magic, i;
			unsigned int broot;
			DgramHeaderBreak(msg, rank, srcpe, magic, seqno, broot); //FIXME: what does this do?
			size = CmiMsgHeaderGetLength(msg);
			MACHSTATE2(3,"START of a new message from node %d of total size %d",nodeNo,size);
			newmsg = (char *)CmiAlloc(size); // FIXME: is there a better way than to do an alloc?
			_MEMCHECK(newmsg);
			memcpy(newmsg, msg, len);
			node->asm_rank = rank;
			node->asm_total = size;
			node->asm_fill = len;
			node->asm_msg = newmsg;
			node->infiData->broot = broot;
			if(len>size) {
				//there are more packets following
				node->infiData->state = INFI_DATA;
			} else if(len == size){
				//this is the only packet for this message 
				node->infiData->state = INFI_HEADER_DATA;
			} else { //len < size 
				CmiPrintf("size: %d, len:%d.\n", size, len);
				CmiAbort("\n\n\t\tLength mismatch!!\n\n");
			}

			break;
		}
		case INFI_DATA: {
			if(node->asm_fill+len<node->asm_total&&len!=Cmi_dgram_max_data){
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
	if(node->infiData->state == INFI_HEADER_DATA){ // then the entire message is ready so hand it over
		int total_size = node->asm_total;
		node->asm_msg = NULL;
//		handoverMessage(newmsg,total_size,node->asm_rank,node->infiData->broot,1); // FIXME: handover the message!
		MACHSTATE3(3,"Message from node %d of length %d completely received msg %p",nodeNo,total_size,newmsg);
	}
}


void processSendWC(struct ibv_wc *sendWC) {
	MACHSTATE(3,"processSendWC {");
	infiPacket packet = (infiPacket )sendWC->wr_id;
	FreeInfiPacket(packet);
	MACHSTATE(3,"} processSendWC ");
}

void processRecvWC(struct ibv_wc *recvWC,const int toBuffer) {
	//ibv_post_recv ...
	struct infiBuffer *buffer = (struct infiBuffer *) recvWC->wr_id;
	struct infiPacketHeader *header = (struct infiPacketHeader *)buffer->buf;
	int nodeNo = header->nodeNo;

	int len = recvWC->byte_len-sizeof(struct infiPacketHeader);
	MACHSTATE(3,"processRecvWC {");
	MACHSTATE2(3,"packet from node %d len %d",nodeNo,len);

	if(header->code & INFIPACKETCODE_DATA){
		processMessage(nodeNo,len,(buffer->buf+sizeof(struct infiPacketHeader)),toBuffer);
	}
	else if(header->code & INFIDUMMYPACKET){
		MACHSTATE(3,"Dummy packet");
	}
	else if(header->code & INFIBARRIERPACKET){
		MACHSTATE(3,"Barrier packet");
		CmiAbort("Should not receive Barrier packet in normal polling loop.  Your Barrier is broken");
	}

	{
		struct ibv_sge list = {
			.addr   = (uintptr_t) buffer->buf,
			.length = buffer->size,
			.lkey   = buffer->key->lkey,
		};
	
		struct ibv_recv_wr wr = {
			.wr_id = (uint64_t)buffer,
			.sg_list = &list,
			.num_sge = 1,
			.next = NULL
		};
		struct ibv_recv_wr *bad_wr;
	
		CmiAssert(ibv_post_recv(context->qp,&wr,&bad_wr)==0);
	}
	MACHSTATE(3,"} processRecvWC ");
}


static inline int pollCq(const int toBuffer,struct ibv_cq *cq) {
	/* toBuffer ignored for sendCq and is used for recvCq */
	int i;
	int ne;
	struct ibv_wc wc[WC_LIST_SIZE];

	MACHSTATE1(2,"pollCq %d (((",toBuffer);
	ne = ibv_poll_cq(cq,WC_LIST_SIZE,&wc[0]);
	
	if(ne != 0){
		MACHSTATE1(3,"pollCq ne %d",ne);
		if(ne<0)
			CmiAbort("ibv_poll_cq error");
	}
	
	for(i=0;i<ne;i++){
//		CmiAssert(wc[i].status==IBV_WC_SUCCESS);
        if(wc[i].status!=IBV_WC_SUCCESS) {
                MACHSTATE3(3,"wc[%i].status=%i (%s)",i,wc[i].status,ibv_wc_status_str(wc[i].status)); 
                MACHSTATE3(3,"   wr_id=%i qp_num=%i vendor_err=%i",wc[i].wr_id,wc[i].qp_num,wc[i].vendor_err); 
                MACHSTATE1(3,"  key=%p ",
                    ((struct infiBuffer *)((&wc[i])->wr_id))->key);
/*                MACHSTATE4(3,"  lkey=%p buffer=%d length=%d end=%d",
                    ((struct infiBuffer *)((&wc[i])->wr_id))->key->lkey,
                    ((struct infiBuffer *)((&wc[i])->wr_id))->buf, 
                    ((struct infiBuffer *)((&wc[i])->wr_id))->size,
                    ((struct infiBuffer *)((&wc[i])->wr_id))->buf+((struct infiBuffer *)((&wc[i])->wr_id))->size); 

*/		        CmiAssert(wc[i].status==IBV_WC_SUCCESS);
        }


		switch(wc[i].opcode){
			case IBV_WC_SEND: //sending message
				processSendWC(&wc[i]);
				break;
			case IBV_WC_RECV: // recving message
				processRecvWC(&wc[i],toBuffer);
				break;
			default:
				CmiAbort("Wrong type of work completion object in cq");
				break;
		}
			
	}
	MACHSTATE1(2,"))) pollCq %d",toBuffer);
	return ne;

}

static inline  void CommunicationServer_lock(int toBuffer) {
        CmiCommLock();
        CommunicationServer_nolock(0);
        CmiCommUnlock();
}

static inline  void CommunicationServer_nolock(int toBuffer) {
/*
	if(_Cmi_numnodes <= 1){
		pollCmiDirectQ();
		return;
	} 
*/
	MACHSTATE(2,"CommServer_nolock{");
        
//	pollCmiDirectQ();	// FIXME: not sure what this does...
  
	pollCq(toBuffer,context->sendCq);
	pollCq(toBuffer,context->recvCq);

//	if(toBuffer == 0)
//		processAllBufferedMsgs();	// FIXME : I don't think we need buf'ed msgs

	MACHSTATE(2,"} CommServer_nolock ne");
}




static uint16_t getLocalLid(struct ibv_context *dev_context, int port){
	struct ibv_port_attr attr;
	if (ibv_query_port(dev_context, port, &attr))
		return 0;

	return attr.lid;
}


struct infiAddr* initinfiAddr(int node,int lid,int qpn,int psn) {
	struct infiAddr *addr=malloc(sizeof(struct infiAddr));

	addr->lid=lid;
	addr->qpn=qpn;
	addr->psn=psn;

	return addr;
}

struct infiOtherNodeData *initinfiData(int node,int lid,int qpn,int psn) {
//struct infiOtherNodeData *initInfiOtherNodeData(int node,int addr[3]){
        struct infiOtherNodeData *ret=malloc(sizeof(struct infiOtherNodeData));
	//set qp
	ret->qp.lid=lid;
	ret->qp.qpn=qpn;
	ret->qp.psn=psn;

        ret->nodeNo = node;
        ret->state = INFI_HEADER_DATA; 
	
    MACHSTATE4(3,"Storing node[%i] (lid=%i qpn=%i psn=%i)",node,lid,qpn,psn);

//        ret->qp = context->qp;
//      ret->totalTokens = tokensPerProcessor;
//      ret->tokensLeft = tokensPerProcessor;
//      ret->postedRecvs = tokensPerProcessor;
    return ret;
}


/***********************************************************************
 * CommunicationServer()
 * 
 * This function does the scheduling of the tasks related to the
 * message sends and receives. It is called from the CmiGeneralSend()
 * function, and periodically from the CommunicationInterrupt() (in case 
 * of the single processor version), and from the comm_thread (for the
 * SMP version). Based on which of the data/control read/write sockets
 * are ready, the corresponding tasks are called
 *
 ***********************************************************************/
void CmiHandleImmediate();
static void CommunicationServer(int sleepTime, int where) {
/*  0: from smp thread
    1: from interrupt
    2: from worker thread
*/

	if(where==COMM_SERVER_FROM_INTERRUPT)
		return;
#if CMK_SMP
	if(where == COMM_SERVER_FROM_WORKER)
		return;
	if(where == COMM_SERVER_FROM_SMP) {
		ServiceCharmrun_nolock();
	}
        CommunicationServer_lock(0);
#else
	ServiceCharmrun_nolock(); 
	CommunicationServer_nolock(0);
#endif
}







static void sendBarrierMessage(int pe) {
	/* we will only need one packet */
	int size=32;
	OtherNode node=nodes+pe;
	infiPacket packet;
	MallocInfiPacket(packet); 
	packet->size = size;
	packet->buf = CmiAlloc(size); 
	packet->header.code=INFIBARRIERPACKET;
	packet->wr.wr.ud.ah=context->ah[pe];
	packet->wr.wr.ud.remote_qpn=nodes[pe].infiData->qp.qpn;
	packet->wr.wr.ud.remote_qkey = 0x11111111;

	MACHSTATE1(3,"HERE -> %d",packet->header.code);
MACHSTATE2(3,"sending to qpn=%i pe=%i",nodes[pe].infiData->qp.qpn,pe);	
	struct ibv_mr *key=METADATAFIELD(packet->buf)->key;
	MACHSTATE3(3,"Barrier packet to %d size %d wr_id %d",node->infiData->nodeNo,size,packet->wr.wr_id);
	EnqueuePacket(node,packet,size,key);
}

// FIXME: haven't looked at yet
static void recvBarrierMessage() {
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
int count=0; // FIXME: remove debug
MACHSTATE(3,"recvBarrierMessage 0"); // FIXME: REMOVE this debug
	while(!barrierReached) {
		/* gengbin's semantic will implode if more than one q is polled at a time */
		pollCq(toBuffer,context->sendCq); // FIXME: just put this in to fix pollSendCq req - not sure if its correct
		ne = ibv_poll_cq(context->recvCq,1,&wc[0]);
		if(ne!=0){
			MACHSTATE1(3,"recvBarrier ne %d",ne);
            CmiAssert(ne>0);
		}
		for(i=0;i<ne;i++){
			if(wc[i].status != IBV_WC_SUCCESS){
                MACHSTATE3(3,"wc[%i].status=%i (%s)",i,wc[i].status,ibv_wc_status_str(wc[i].status)); 
                MACHSTATE3(3,"   wr_id=%i qp_num=%i vendor_err=%i",wc[i].wr_id,wc[i].qp_num,wc[i].vendor_err); 
/*
                MACHSTATE4(3,"  lkey=%d buffer=%d length=%d end=%d",
                    ((struct infiBuffer *)((&wc[i])->wr_id))->key->lkey,
                    ((struct infiBuffer *)((&wc[i])->wr_id))->buf, 
                    ((struct infiBuffer *)((&wc[i])->wr_id))->size,
                    ((struct infiBuffer *)((&wc[i])->wr_id))->buf+((struct infiBuffer *)((&wc[i])->wr_id))->size); 
                MACHSTATE1(3,"  key=%p ",
                    ((struct infiBuffer *)((&wc[i])->wr_id))->key);
*/
                
				CmiAbort("wc.status !=IBV_WC_SUCCESS"); 
			}
			switch(wc[i].opcode){
				case IBV_WC_RECV: /* we have something to consider*/
				    MACHSTATE(3," IN HERE !!!!!!!!!!");
					recvWC=&wc[i];

					buffer = (struct infiBuffer *) recvWC->wr_id;
					header = (struct infiPacketHeader *)(buffer->buf + 40);		// add 40 bytes to skip the GRH

					nodeNo = header->nodeNo;
					len = recvWC->byte_len-sizeof(struct infiPacketHeader);
					if(header->code & INFIPACKETCODE_DATA){
						processMessage(nodeNo,len,(buffer->buf+sizeof(struct infiPacketHeader)),toBuffer);
					} else if(header->code & INFIDUMMYPACKET){
						MACHSTATE(3,"Dummy packet");
					} else if(header->code & INFIBARRIERPACKET){
						MACHSTATE2(3,"Barrier packet from node %d len %d",nodeNo,len);
						barrierReached=1;
					}else // FIXME: erase this else clause
						MACHSTATE2(3,"Ups... %d %d",header->code,nodeNo);
					{
						struct ibv_sge list = {
							.addr     = (uintptr_t) buffer->buf,
							.length = buffer->size,
							.lkey     = buffer->key->lkey
						};
						
						struct ibv_recv_wr wr = {
							.wr_id = (uint64_t)buffer,
							.sg_list = &list,
							.num_sge = 1,
							.next = NULL
						};
						struct ibv_recv_wr *bad_wr;

    						CmiAssert(ibv_post_recv(context->qp,&wr,&bad_wr)==0); 

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



// FIXME: haven't looked at yet
/* happen at node level */
int CmiBarrier() {
	int len, size, i;
	int status;
	int count = 0;
	OtherNode node;
	int numnodes = CmiNumNodes();
MACHSTATE1(3,"Barrier 1 rank=%i",CmiMyRank());
	if (CmiMyRank() == 0) { /* every one send to pe 0 */
		if (CmiMyNode() != 0) { 

MACHSTATE(3,"Barrier sendmsg");
			sendBarrierMessage(0);
			recvBarrierMessage();
			for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
				int p = CmiMyNode();
				p = BROADCAST_SPANNING_FACTOR*p + i;
				if (p > numnodes - 1) break;
				p = p%numnodes;
				/* printf("[%d] RELAY => %d \n", CmiMyPe(), p); */
				sendBarrierMessage(p);
			}
		} else {
MACHSTATE(3,"Barrier else");
			for (count = 1; count < numnodes; count ++) {
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
MACHSTATE(3,"Barrier 3");
	}
MACHSTATE(3,"Barrier 4");
	CmiNodeAllBarrier();
	//  processAllBufferedMsgs();
	/* printf("[%d] OUT of barrier \n", CmiMyPe()); */
MACHSTATE(3,"Barrier e");
}

// FIXME: haven't looked at yet
/* everyone sends a message to pe 0 and go on */
int CmiBarrierZero() {
	int i;

	if (CmiMyRank() == 0) {
		if (CmiMyNode()) {
			sendBarrierMessage(0);
		} else {
			for (i=0; i<CmiNumNodes()-1; i++) {
				recvBarrierMessage();
			}
		}
	}
	CmiNodeAllBarrier();
	//  processAllBufferedMsgs();
}


void createqp(struct ibv_device *dev){

	context->sendCq = ibv_create_cq(context->context,context->sendCqSize,NULL,NULL,0);
	CmiAssert(context->sendCq != NULL);
	MACHSTATE1(3,"sendCq created %p",context->sendCq);
	
	context->recvCq = ibv_create_cq(context->context,context->recvCqSize,NULL,NULL,0);
	CmiAssert(context->recvCq != NULL);
	MACHSTATE2(3,"recvCq created %p %d",context->recvCq,context->recvCqSize);

    {
    	struct ibv_qp_init_attr attr = {
            .qp_context = context->context,
		    .qp_type = IBV_QPT_UD,
	    	.send_cq = context->sendCq,
    		.recv_cq = context->recvCq,
            .srq = NULL,
            .sq_sig_all=0,
    		.cap     = {
		    	.max_send_wr  = context->sendCqSize, // FIXME: this isn't right - need to make a smaller number
	    		.max_recv_wr  = context->recvCqSize, // FIXME: this isn't right - need to make a smaller number
    			.max_send_sge = 1,
			    .max_recv_sge = 1,
		    },
	    };
    	context->qp = ibv_create_qp(context->pd,&attr);
        CmiAssert(context->qp != NULL);
        MACHSTATE1(3,"qp created %p",context->qp);
    }
    {
    	struct ibv_qp_attr attr;
    	attr.qp_state        = IBV_QPS_INIT;
    	attr.pkey_index      = 0;
    	attr.port_num        = context->ibPort; 
        attr.qkey            = 0x11111111;
	    if(ibv_modify_qp(context->qp, &attr,
		    IBV_QP_STATE              |
    		IBV_QP_PKEY_INDEX         |
	    	IBV_QP_PORT               |
		    IBV_QP_QKEY))
            CmiAbort("Could not modify QP to INIT");
    }
    {
    	struct ibv_qp_attr attr;
        attr.qp_state = IBV_QPS_RTR;
        if(ibv_modify_qp(context->qp, &attr, IBV_QP_STATE))
            CmiAbort("Could not modify QP to RTR");
    }
    {
    	struct ibv_qp_attr attr;
        attr.qp_state = IBV_QPS_RTS;
        attr.sq_psn=context->localAddr.psn;
        if(ibv_modify_qp(context->qp, &attr, IBV_QP_STATE|IBV_QP_SQ_PSN))
            CmiAbort("Could not modify QP to RTS");
    }

	context->localAddr.lid=getLocalLid(context->context,context->ibPort);
	context->localAddr.qpn = context->qp->qp_num;
	context->localAddr.psn = lrand48() & 0xffffff;

	MACHSTATE3(4,"qp information (lid=%i qpn=%i psn=%i)",context->localAddr.lid,context->localAddr.qpn,context->localAddr.psn);
}

void createah() {
	int i,numnodes;

	numnodes=_Cmi_numnodes;
	context->ah=(struct ibv_ah **)malloc(sizeof(struct ibv_ah *)*numnodes);

	for(i=0;i<numnodes;i++) { 
//		if(i!=_Cmi_mynode) {
            {
    			struct ibv_ah_attr ah_attr = {
    				.is_global     = 0,
    				.dlid          = nodes[i].infiData->qp.lid,
    				.sl            = 0,
    				.src_path_bits = 0,
    				.port_num      = context->ibPort,
    			};
    			context->ah[i]=ibv_create_ah(context->pd,&ah_attr);
    			CmiAssert(context->ah[i]!=0);
                MACHSTATE2(4,"ah for node %i lid=%i ",i,ah_attr.dlid);
            }
//		}
	}
}


void CmiMachineInit(char **argv)
{
	int i;
	int calcmaxsize;
	int lid;

	MACHSTATE(3,"CmiMachineInit {");
	MACHSTATE2(3,"_Cmi_numnodes %d CmiNumNodes() %d",_Cmi_numnodes,CmiNumNodes());
	MACHSTATE1(3,"CmiMyNodeSize() %d",CmiMyNodeSize());

	/* copied from ibverbs.c */
	firstBinSize = 120;
	blockThreshold=8;
	blockAllocRatio=16;

	mtu_size=1200;
	packetsize = mtu_size*4;
	Cmi_dgram_max_data=packetsize-sizeof(struct infiPacketHeader);
	CmiAssert(Cmi_dgram_max_data>1);
	
	calcmaxsize=8000;

	maxrecvbuffers=calcmaxsize;
	maxtokens = calcmaxsize;

    initInfiCmiChunkPools();
	
	ibud.devlist = ibv_get_device_list(NULL);
	CmiAssert(ibud.devlist != NULL);

	ibud.dev = *(ibud.devlist);
	CmiAssert(ibud.dev != NULL);

	MACHSTATE1(3,"device name %s",ibv_get_device_name(ibud.dev));

	context = (struct infiContext *)malloc(sizeof(struct infiContext));
	
	MACHSTATE1(3,"context allocated %p",context);
	
	context->sendCqSize = 2; // FIXME: 1?
	context->recvCqSize = calcmaxsize+1; 
	context->ibPort = 1;
	context->context = ibv_open_device(ibud.dev);  //the context for this infiniband device 
	CmiAssert(context->context != NULL);
	
	MACHSTATE1(3,"device opened %p",context->context);

	context->pd = ibv_alloc_pd(context->context); //protection domain
	CmiAssert(context->pd != NULL);

	context->header.nodeNo = _Cmi_mynode;

	if(_Cmi_numnodes>1) {
		createqp(ibud.dev);
//MACHSTATE1(3,"pp post recv=%i",pp_post_recv());	
    }
	
	MACHSTATE(3,"} CmiMachineInit");
}

void CmiCommunicationInit(char **argv) {
	MACHSTATE(3,"CmiCommunicationInit {");
	if(_Cmi_numnodes>1) {
        	infiPostInitialRecvs();
	 	createah();
	}
	MACHSTATE(3,"} CmiCommunicationInit");
}

void CmiMachineExit()
{
	ibv_destroy_qp(context->qp);
	ibv_dealloc_pd(context->pd); 
	ibv_close_device(context->context);
	ibv_free_device_list(ibud.devlist);
}

