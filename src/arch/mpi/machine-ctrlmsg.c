/* An alternative way to implement MPI-based machine layer */
/* Control flows of this scheme:
 * SEND SIDE:
 * 1. send a fixed-size small control msg to destination (MPI_Send)
 *     so the ctrl msg buffer can be reused every time.
 * 2. immediately following the 1st step, send the actual msg (MPI_Isend)
 * 3. free the buffer for the sent msg
 * 
 * RECV SIDE:
 * 1. Pre-post buffers for those small ctrl msgs (MPI_Irecv)
 * 2. If any ctrl msg is received, issue a (i)recv call for the actual msg 
 *     (differentiate small/large msgs)
 *
 * MEMORY ALLOCATION:
 * use MPI_Alloc_mem and MPI_Free_mem so that CmiAllloc/CmiFree needs to be changed
 */

/* This file contains variables and function declarations that are used for this alternative implementation */

/* This file contains function and variables definitions that are used for this alternative implementation */

#if USE_MPI_CTRLMSG_SCHEME

#define CTRL_MSG_TAG         (TAG-13)
#define USE_NUM_TAGS            1000

static int MPI_CTRL_MSG_CNT=10;
static int tags;

typedef struct MPICtrlMsgEntry{
	int src;
	int size;
        int tag;
}MPICtrlMsgEntry;

typedef struct RecvCtrlMsgEntry{
	int bufCnt;
	MPI_Request *ctrlReqs; /* sizeof(MPI_Request* bufCnt */
	MPICtrlMsgEntry *bufs; /*sizeof(MPICtrlMsgEntry)*bufCnt*/
}RecvCtrlMsgEntry;

static RecvCtrlMsgEntry recvCtrlMsgList;

static void createCtrlMsgIrecvBufs(){
	int i;
	MPICtrlMsgEntry *bufPtr = NULL;
	MPI_Request *reqPtr = NULL;
	int count = MPI_CTRL_MSG_CNT;

        tags = 0;
	
	recvCtrlMsgList.bufCnt = count;
	recvCtrlMsgList.ctrlReqs = (MPI_Request *)malloc(sizeof(MPI_Request)*count);
	recvCtrlMsgList.bufs = (MPICtrlMsgEntry *)malloc(sizeof(MPICtrlMsgEntry)*count);
	
	bufPtr = recvCtrlMsgList.bufs;
	reqPtr = recvCtrlMsgList.ctrlReqs;
	
	for(i=0; i<count; i++, bufPtr++, reqPtr++){
		if(MPI_SUCCESS != MPI_Irecv(bufPtr, sizeof(MPICtrlMsgEntry), 
                              MPI_BYTE, MPI_ANY_SOURCE, CTRL_MSG_TAG, charmComm, reqPtr)){
			CmiAbort("MPI_Irecv failed in creating pre-posted ctrl msg buffers\n");
		}
	}
}

static void sendViaCtrlMsg(int node, int size, char *msg, SMSG_LIST *smsg){
	MPICtrlMsgEntry one;

	one.src = CmiMyNode();
	one.size = size;
        one.tag = TAG + 100 + tags;

        tags = (tags+1)%USE_NUM_TAGS;
	
	START_TRACE_SENDCOMM(msg);
	if(MPI_SUCCESS != MPI_Send((void *)&one, sizeof(MPICtrlMsgEntry), MPI_BYTE, node, CTRL_MSG_TAG, charmComm)){
		CmiAbort("MPI_Send failed in sending ctrl msg\n");
	}
	if (MPI_SUCCESS != MPI_Isend((void *)msg,size,MPI_BYTE,node,one.tag,charmComm,&(smsg->req)))
            CmiAbort("MPISendOneMsg: MPI_Isend failed!\n");
	END_TRACE_SENDCOMM(msg);
}

/* returns the size of msg to be received. If there's no msg to be received, then -1 is returned */
static int recvViaCtrlMsg(){
	int count = recvCtrlMsgList.bufCnt;
	MPI_Request *ctrlReqs = recvCtrlMsgList.ctrlReqs;
	MPICtrlMsgEntry *ctrlMsgs = recvCtrlMsgList.bufs;
	
	int completed_index = -1;
	int flg = 0;
	int nbytes = -1;
	MPI_Status sts;
	if(MPI_SUCCESS != MPI_Testany(count, ctrlReqs, &completed_index, &flg, &sts)){
		CmiAbort("MPI_Testany failed for checking if ctrl msg is received\n");
	}
	
	if(flg){
		int src = ctrlMsgs[completed_index].src;
		int msgsize = ctrlMsgs[completed_index].size;
		nbytes = msgsize;
		char *actualMsg = (char *)CmiAlloc(msgsize);
		
		IRecvList one = irecvListEntryAllocate();
		
		/* irecv the actual msg */
		if(MPI_SUCCESS != MPI_Irecv(actualMsg, msgsize, MPI_BYTE, src, ctrlMsgs[completed_index].tag, charmComm, &(one->req))){
			CmiAbort("MPI_Irecv failed after a ctrl msg is received\n");
		}

		/* repost the ctrl msg */
		if(MPI_SUCCESS != MPI_Irecv(ctrlMsgs+completed_index, sizeof(MPICtrlMsgEntry), MPI_BYTE,
			                                          MPI_ANY_SOURCE, CTRL_MSG_TAG, charmComm, ctrlReqs+completed_index)){
			CmiAbort("MPI_Irecv failed in re-posting a ctrl msg is received\n");
		}
		
		one->msg = actualMsg;
		one->size = msgsize;
		one->next = NULL;
		waitIrecvListTail->next = one;
		waitIrecvListTail = one;
	}
	
	return nbytes;
}

#endif
