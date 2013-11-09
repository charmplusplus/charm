/**
 *        functions for broadcast
**/

CmiCommHandle CmiSendNetworkFunc(int destPE, int size, char *msg, int mode);

static void handleOneBcastMsg(int size, char *msg) {
    CmiAssert(CMI_BROADCAST_ROOT(msg)!=0);
#if CMK_OFFLOAD_BCAST_PROCESS
    if (CMI_BROADCAST_ROOT(msg)>0) {
        CMIQueuePush(CsvAccess(procBcastQ), msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
        CMIQueuePush(CsvAccess(nodeBcastQ), msg);
#endif
    }
#else
    if (CMI_BROADCAST_ROOT(msg)>0) {
        processProcBcastMsg(size, msg);
    } else {
#if CMK_NODE_QUEUE_AVAILABLE
        processNodeBcastMsg(size, msg);
#endif
    }
#endif
}

static void processBcastQs() {
#if CMK_OFFLOAD_BCAST_PROCESS
    char *msg;
    do {
        msg = CMIQueuePop(CsvAccess(procBcastQ));
        if (!msg) break;
        MACHSTATE2(4, "[%d]: process a proc-level bcast msg %p begin{", CmiMyNode(), msg);
        processProcBcastMsg(CMI_MSG_SIZE(msg), msg);
        MACHSTATE2(4, "[%d]: process a proc-level bcast msg %p end}", CmiMyNode(), msg);
    } while (1);
#if CMK_NODE_QUEUE_AVAILABLE
    do {
        msg = CMIQueuePop(CsvAccess(nodeBcastQ));
        if (!msg) break;
        MACHSTATE2(4, "[%d]: process a node-level bcast msg %p begin{", CmiMyNode(), msg);
        processNodeBcastMsg(CMI_MSG_SIZE(msg), msg);
        MACHSTATE2(4, "[%d]: process a node-level bcast msg %p end}", CmiMyNode(), msg);
    } while (1);
#endif
#endif
}

static INLINE_KEYWORD void processProcBcastMsg(int size, char *msg) {
    /* Since this function is only called on intermediate nodes,
     * the rank of this msg should be 0.
     */
    CmiAssert(CMI_DEST_RANK(msg)==0);
    /*CmiPushPE(CMI_DEST_RANK(msg), msg);*/

#if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildrenProc(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    SendHyperCubeProc(size, msg);
#endif
#if CMK_BROADCAST_SPANNING_TREE && CMK_BROADCAST_USE_CMIREFERENCE
      /* same message may be sent out, make a copy of it */
    if (CmiNumNodes()>1 && CmiGetReference(msg)>1) {
      void *newmsg;
      newmsg = CopyMsg(msg, size);
      CmiFree(msg);
      msg = newmsg;
    }
#endif
    CmiPushPE(0, msg);

}


#if CMK_NODE_QUEUE_AVAILABLE
static INLINE_KEYWORD void processNodeBcastMsg(int size, char *msg) {
#if CMK_BROADCAST_SPANNING_TREE
    SendSpanningChildrenNode(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    SendHyperCubeNode(size, msg);
#endif

    /* In SMP mode, this push operation needs to be executed
     * after forwarding broadcast messages. If it is executed
     * earlier, then during the bcast msg forwarding period,
     * the msg could be already freed on the worker thread.
     * As a result, the forwarded message could be wrong!
     * 
     */
    CmiPushNode(msg);
}
#endif

static void SendSpanningChildren(int size, char *msg, int rankToAssign, int startNode) {
#if CMK_BROADCAST_SPANNING_TREE
    int i, oldRank;
    char *newmsg;

    oldRank = CMI_DEST_RANK(msg);
    /* doing this is to avoid the multiple assignment in the following for loop */
    CMI_DEST_RANK(msg) = rankToAssign;
    /* first send msgs to other nodes */
    CmiAssert(startNode >=0 &&  startNode<CmiNumNodes());
    for (i=1; i<=BROADCAST_SPANNING_FACTOR; i++) {
        int nd = CmiMyNode()-startNode;
        if (nd<0) nd+=CmiNumNodes();
        nd = BROADCAST_SPANNING_FACTOR*nd + i;
        if (nd > CmiNumNodes() - 1) break;
        nd += startNode;
        nd = nd%CmiNumNodes();
        CmiAssert(nd>=0 && nd!=CmiMyNode());
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        CmiSendNetworkFunc(CmiNodeFirst(nd), size, msg, BCAST_SYNC);
#else
        newmsg = CopyMsg(msg, size);
        CmiSendNetworkFunc(CmiNodeFirst(nd), size, newmsg, BCAST_SYNC);
#endif
    }
    CMI_DEST_RANK(msg) = oldRank;
#endif
}

static void SendHyperCube(int size,  char *msg, int rankToAssign, int startNode) {
#if CMK_BROADCAST_HYPERCUBE
    int i, cnt, tmp, relDist, oldRank;
    const int dims=CmiNodesDim;

    oldRank = CMI_DEST_RANK(msg);
    /* doing this is to avoid the multiple assignment in the following for loop */
    CMI_DEST_RANK(msg) = rankToAssign;

    /* first send msgs to other nodes */
    relDist = CmiMyNode()-startNode;
    if (relDist < 0) relDist += CmiNumNodes();

    /* Sending scheme example: say we have 9 nodes, and the msg is sent from 0
     * The overall sending steps will be as follows:
     * 0-->8, 0-->4, 0-->2, 0-->1
     *               4-->6, 4-->5
     *                      2-->3
     *                      6-->7
     * So for node id as N=A+2^B, it will forward the broadcast (B-1) msg to in
     * the order as: N+2^(B-1), N+2^(B-2),..., N+1 except node 0, where B is
     * the first position of bit 1 in the binary format of the number of N
     * counting from the right with count starting from 0.
     * On node 0, the value "B" should be CmiNodesDim
     */
    /* Calculate 2^B */
    if(relDist==0) cnt = 1<<dims;
    else cnt = relDist & ((~relDist)+1);
    /*CmiPrintf("ND[%d]: send bcast msg with cnt=%d\n", CmiMyNode(), cnt);*/
    /* Begin to send msgs */
    for(cnt>>=1; cnt>0; cnt>>=1){
        int nd = relDist + cnt;
        char *newmsg;
        if (nd >= CmiNumNodes()) continue;
        nd = (nd+startNode)%CmiNumNodes();
        /*CmiPrintf("ND[%d]: send to node %d\n", CmiMyNode(), nd);*/
        CmiAssert(nd>=0 && nd!=CmiMyNode());
#if CMK_BROADCAST_USE_CMIREFERENCE
        CmiReference(msg);
        CmiSendNetworkFunc(CmiNodeFirst(nd), size, msg, BCAST_SYNC);
#else
        newmsg = CopyMsg(msg, size);
        CmiSendNetworkFunc(CmiNodeFirst(nd), size, newmsg, BCAST_SYNC);
#endif
    }
    CMI_DEST_RANK(msg) = oldRank;
#endif
}

static void SendSpanningChildrenProc(int size, char *msg) {
    int startnode = CMI_BROADCAST_ROOT(msg)-1;
    SendSpanningChildren(size, msg, 0, startnode);
#if CMK_SMP
    /* second send msgs to my peers on this node */
    SendToPeers(size, msg);
#endif
}

/* send msg along the hypercube in broadcast. (Sameer) */
static void SendHyperCubeProc(int size, char *msg) {
    int startpe = CMI_BROADCAST_ROOT(msg)-1;
    int startnode = CmiNodeOf(startpe);
#if CMK_SMP
    if (startpe > CmiNumPes()) startnode = startpe - CmiNumPes();
#endif
    SendHyperCube(size, msg, 0, startnode);
#if CMK_SMP
    /* second send msgs to my peers on this node */
    SendToPeers(size, msg);
#endif
}

#if CMK_NODE_QUEUE_AVAILABLE
static void SendSpanningChildrenNode(int size, char *msg) {
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    SendSpanningChildren(size, msg, DGRAM_NODEMESSAGE, startnode);
}
static void SendHyperCubeNode(int size, char *msg) {
    int startnode = -CMI_BROADCAST_ROOT(msg)-1;
    SendHyperCube(size, msg, DGRAM_NODEMESSAGE, startnode);
}
#endif

#if USE_COMMON_SYNC_BCAST
/* Functions regarding broadcat op that sends to every one else except me */
void CmiSyncBroadcastFn1(int size, char *msg) {
    int i, mype;

    CQdCreate(CpvAccess(cQdState), CmiNumPes()-1);
    /*record the rank to avoid re-sending the msg in  spanning tree or hypercube*/
    CMI_DEST_RANK(msg) = CmiMyRank();

#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT(msg, CmiMyNode()+1);
    SendSpanningChildrenProc(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    CMI_SET_BROADCAST_ROOT(msg, CmiMyNode()+1);
    SendHyperCubeProc(size, msg);
#else
    mype = CmiMyPe();
    #if CMK_SMP
    /* In SMP, this function may be called from comm thread with a larger pe */
    if(mype >= _Cmi_numpes){
	for(i=0; i<_Cmi_numpes; i++)
		CmiSyncSendFn(i, size, msg);
	return;
    }
    #endif
	
    for ( i=mype+1; i<_Cmi_numpes; i++ )
        CmiSyncSendFn(i, size, msg) ;
	
    for ( i=0; i<mype; i++ )
        CmiSyncSendFn(i, size, msg) ;
#endif

    /*CmiPrintf("In  SyncBroadcast broadcast\n");*/
}

void CmiSyncBroadcastFn(int size, char *msg) {
    void *newmsg = msg;
#if CMK_BROADCAST_SPANNING_TREE && CMK_BROADCAST_USE_CMIREFERENCE
      /* need to copy the msg in case the msg is on the stack */
      /* and we only need to copy when sending out network */
    if (CmiNumNodes()>1) newmsg = CopyMsg(msg, size);
#endif
    CmiSyncBroadcastFn1(size, newmsg);
#if CMK_BROADCAST_SPANNING_TREE && CMK_BROADCAST_USE_CMIREFERENCE
    if (newmsg != msg) CmiFree(newmsg);
#endif
}

void CmiFreeBroadcastFn(int size, char *msg) {
    CmiSyncBroadcastFn1(size,msg);
    CmiFree(msg);
}
#else
#define  CmiSyncBroadcastFn1(s,m)      CmiSyncBroadcastFn(s,m)
#endif

#if USE_COMMON_ASYNC_BCAST
/* FIXME: should use spanning or hypercube, but luckily async is never used */
CmiCommHandle CmiAsyncBroadcastFn(int size, char *msg) {
    /*CmiPrintf("In  AsyncBroadcast broadcast\n");*/
    CmiAbort("CmiAsyncBroadcastFn should never be called");
    return 0;
}
#endif

/* Functions regarding broadcat op that sends to every one */
void CmiSyncBroadcastAllFn(int size, char *msg) {
    void *newmsg = msg;
#if CMK_BROADCAST_SPANNING_TREE && CMK_BROADCAST_USE_CMIREFERENCE
      /* need to copy the msg in case the msg is on the stack */
      /* and we only need to copy when sending out network */
    if (CmiNumNodes()>1) newmsg = CopyMsg(msg, size);
#endif
    CmiSyncSendFn(CmiMyPe(), size, newmsg) ;
    CmiSyncBroadcastFn1(size, newmsg);
#if CMK_BROADCAST_SPANNING_TREE && CMK_BROADCAST_USE_CMIREFERENCE
    if (newmsg != msg) CmiFree(newmsg);
#endif
}

void CmiFreeBroadcastAllFn(int size, char *msg) {
    CmiSyncBroadcastFn1(size, msg);
#if CMK_BROADCAST_SPANNING_TREE && CMK_BROADCAST_USE_CMIREFERENCE
      /* need to copy the msg in case the msg is on the stack */
      /* and we only need to copy when sending out network */
    if (CmiNumNodes()>1 && CmiGetReference(msg)>1) {
      void *newmsg = CopyMsg(msg, size);
      CmiFree(msg);
      msg = newmsg;
    }
#endif
    CmiSendSelf(msg);
}

CmiCommHandle CmiAsyncBroadcastAllFn(int size, char *msg) {
    CmiSendSelf(CopyMsg(msg, size));
    return CmiAsyncBroadcastFn(size, msg);
}

#if CMK_NODE_QUEUE_AVAILABLE
#if USE_COMMON_SYNC_BCAST
void CmiSyncNodeBroadcastFn(int size, char *msg) {
    int mynode = CmiMyNode();
    int i;
    CQdCreate(CpvAccess(cQdState), CmiNumNodes()-1);
#if CMK_BROADCAST_SPANNING_TREE
    CMI_SET_BROADCAST_ROOT(msg, -CmiMyNode()-1);
    SendSpanningChildrenNode(size, msg);
#elif CMK_BROADCAST_HYPERCUBE
    CMI_SET_BROADCAST_ROOT(msg, -CmiMyNode()-1);
    SendHyperCubeNode(size, msg);
#else
    for (i=mynode+1; i<CmiNumNodes(); i++)
        CmiSyncNodeSendFn(i, size, msg);
    for (i=0; i<mynode; i++)
        CmiSyncNodeSendFn(i, size, msg);
#endif
}

void CmiFreeNodeBroadcastFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    CmiFree(msg);
}
#endif

#if USE_COMMON_ASYNC_BCAST
CmiCommHandle CmiAsyncNodeBroadcastFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    return 0;
}
#endif

void CmiSyncNodeBroadcastAllFn(int size, char *msg) {
    CmiSyncNodeSendFn(CmiMyNode(), size, msg);
    CmiSyncNodeBroadcastFn(size, msg);
}

CmiCommHandle CmiAsyncNodeBroadcastAllFn(int size, char *msg) {
    CmiSendNodeSelf(CopyMsg(msg, size));
    return CmiAsyncNodeBroadcastFn(size, msg);
}

void CmiFreeNodeBroadcastAllFn(int size, char *msg) {
    CmiSyncNodeBroadcastFn(size, msg);
    /* Since it's a node-level msg, the msg could be executed on any other
     * procs on the same node. This means, the push of this msg to the
     * node-level queue could be immediately followed a pop of this msg on
     * other cores on the same node even when this msg has not been sent to
     * other nodes. This is the reason CmiSendNodeSelf must be called after
     * CmiSyncNodeBroadcastFn 
     */
    CmiSendNodeSelf(msg);
}
#endif
/* ##### End of Functions Related with Message Sending OPs ##### */

#if ! CMK_MULTICAST_LIST_USE_COMMON_CODE

void CmiSyncListSendFn(int npes, int *pes, int len, char *msg)
{
    LrtsSyncListSendFn(npes, pes, len, msg);
}

CmiCommHandle CmiAsyncListSendFn(int npes, int *pes, int len, char *msg)
{
    return LrtsAsyncListSendFn(npes, pes, len, msg);
}

void CmiFreeListSendFn(int npes, int *pes, int len, char *msg)
{
    LrtsFreeListSendFn(npes, pes, len, msg);
}

#endif
