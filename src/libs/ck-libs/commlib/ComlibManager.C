#include "ComlibManager.h"

CpvDeclare(int, RecvmsgHandle);
CpvDeclare(int, RecvdummyHandle);

#if CHARM_MPI
MPI_Comm groupComm;
MPI_Group group, groupWorld;
#endif

int PERIOD = 10;

void recv_msg(void *msg){

    if(msg == NULL)
        return;

    ComlibPrintf("%d:In recv_msg\n", CkMyPe());

    register envelope* env = (envelope *)msg;
    CProxyElement_ArrayBase ap(env->array_mgr(), env->array_index());
    ComlibPrintf("%d:Array Base created\n", CkMyPe());
    ap.ckSend((CkArrayMessage *)EnvToUsr(env), env->array_ep());
    
    ComlibPrintf("%d:Out of recv_msg\n", CkMyPe());
    return;
}

void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy %d\n", CkMyPe());    
    CmiFree(msg);
}

ComlibManager::ComlibManager(int s){
    init(s, 0);
}

//s = Strategy (0 = tree, 1 = tree, 2 = mesh, 3 = hypercube) 
ComlibManager::ComlibManager(int s, int n){
    init(s, n);
}

void call_endIteration(void *arg){
    ((ComlibManager *)arg)->endIteration();
    return;
}

void ComlibManager::init(int s, int n){
    
    //comm_debug = 1;

    npes = CkNumPes();
    pelist = NULL;

    cmgrID = thisgroup;

    ComlibInit();
    
    strategy = s;
    
    //currently pased by the user. Should discover it.
    if(strategy == USE_STREAMING) {
        nelements = 1;
        if(n > 0)
            PERIOD = n;
    }
    else 
        nelements = n;  //number of elements on that processor, 

    elementCount = 0;
    //    elementRecvCount = 0;

    ComlibPrintf("Strategy %d %d %d\n", strategy, nelements, messagesBeforeFlush);
    
    CpvInitialize(int, RecvmsgHandle);
    CpvAccess(RecvmsgHandle) = CmiRegisterHandler((CmiHandler)recv_msg);

    CpvInitialize(int, RecvdummyHandle);
    CpvAccess(RecvdummyHandle) = CmiRegisterHandler((CmiHandler)recv_dummy);
    
    /*
    if(CkMyPe() == 0) {
    //printf("before createinstance\n");
    comid = ComlibInstance(strategy, CkNumPes());
    //printf("after createInstance\n");
    //        idSet = 1;
    }
    */
    
    //else
    idSet = 0;
    
    iterationFinished = 0;

    messageBuf = NULL;
    messageCount = 0;

    streamingMsgBuf = new CharmMessageHolder*[CkNumPes()];
    streamingMsgCount = new int[CkNumPes()];

    for(int count = 0; count < CkNumPes(); count ++){
        streamingMsgBuf[count] = NULL;
        streamingMsgCount[count] = 0;
    }

    procMap = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count++)
        procMap[count] = count;
    
    CProxy_ComlibManager cgproxy(cmgrID);
    cgproxy[0].done();

    createDone = 0;
    doneReceived = 0;

    totalMsgCount = 0;
    totalBytes = 0;
    nIterations = 0;
}

void ComlibManager::done(){
    static int nrecvd = 0;

    nrecvd ++;

    if(nrecvd == CkNumPes()) {
      if(createDone){
        CProxy_ComlibManager cgproxy(cmgrID);

        if(npes != CkNumPes()) {
	    ComlibPrintf("Calling receive id\n");
	    cgproxy.receiveID(npes, pelist, comid);
            
            delete [] this->pelist;
	}
        else
            cgproxy.receiveID(comid);
      }
      doneReceived = 1;
    }
}

void ComlibManager::createId(){
    if(strategy < USE_GRID) {
	comid = ComlibInstance(strategy, CkNumPes());
    }
    
    if(doneReceived){
      
	CProxy_ComlibManager cgproxy(cmgrID);
	cgproxy.receiveID(comid);
    }
    createDone = 1;
}

void ComlibManager::createId(int *pelist, int npes){
    if(strategy < USE_GRID) {
	comid = ComlibInstance(strategy, npes);
	comid = ComlibEstablishGroup(comid, npes, pelist);
    }

    this->pelist = new int[npes];
    this->npes = npes;

    memcpy(this->pelist, pelist, npes * sizeof(int));

    ComlibPrintf("[%d]Creating comid with %d processors\n", CkMyPe(), npes);
    
    if(doneReceived){
	ComlibPrintf("Calling receive id\n");
	CProxy_ComlibManager cgproxy(cmgrID);
	
	cgproxy.receiveID(npes, pelist, comid);
        delete [] this->pelist;
    }
    createDone = 1;
}

void ComlibManager::setReverseMap(int *pelist, int npes){
    
    for(int pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;
    
    for(int pcount = 0; pcount < npes; pcount++) 
        procMap[pelist[pcount]] = pcount;

    //for(int pcount = 0; pcount < CkNumPes(); pcount++) 
    //if(procMap[pcount] != -1)
    //    ComlibPrintf("(%d, %d), ", pcount, procMap[pcount]);
    //ComlibPrintf("\n");
}

void ComlibManager::localElement(){
    //    ComlibPrintf("In Local Element\n");
    nelements ++;
}

//Called when the array element starts sending messages
//should rather be a function call ?
void ComlibManager::beginIteration(){
    //right now does not do anything might need later
    
    ComlibPrintf("[%d]:In Begin Iteration %d\n", CkMyPe(), elementCount);
    iterationFinished = 0;
}

//called when the array elements has finished sending messages
void ComlibManager::endIteration(){
  
    elementCount ++;
    int count = 0;
    if(elementCount == nelements) {
      
        ComlibPrintf("[%d]:In End Iteration %d\n", CkMyPe(), elementCount);
        iterationFinished = 1;
        
        totalMsgCount += messageCount;
        nIterations ++;

        if(nIterations == LEARNING_PERIOD) {
            //CkPrintf("Sending %d, %d\n", totalMsgCount, totalBytes);
            CProxy_ComlibManager cgproxy(cmgrID);
            cgproxy[0].learnPattern(totalMsgCount, totalBytes);
        }

        if((messageCount == 0) && (CkNumPes() > 0)) {
            DummyMsg * dummymsg = new DummyMsg;
            
            ComlibPrintf("Creating a dummy message\n");
            
            CmiSetHandler(UsrToEnv(dummymsg), 
                          CpvAccess(RecvdummyHandle));
            
            messageBuf = new CharmMessageHolder((char *)dummymsg, CkMyPe());
            messageCount ++;
        }
        
        if(idSet) {	    
            if(strategy == NAMD_STRAT) {
		/*
		// In the Namd Strategy half the messages are sent to a remote 
                //processor and the remaining half are sent directly.
                
                CharmMessageHolder *cmsg = messageBuf;
                envelope *env = (envelope *)cmsg->data;
                for(int count = 0; count < messageCount/2; count ++) {
                    CkSendMsgBranch(env->getEpIdx(), cmsg->getCharmMessage(), cmsg->dest_proc, 
                                    env->getGroupNum());
                    cmsg = cmsg->next;
                }
                
                CharmMessageHolder *tmpbuf = cmsg;
                int tot_size = 0;
                for(int count = messageCount/2; count < messageCount; count ++) {
                    tot_size += cmsg->getSize();
                    cmsg =  cmsg->next;
                }
                
                ComlibMsg *newmsg = new(&tot_size, 0) ComlibMsg;

                cmsg = tmpbuf;
                for(int count = messageCount/2; count < messageCount; count ++) {
                    newmsg->insert(cmsg);
                    delete tmpbuf;
                    cmsg = cmsg->next;
                    tmpbuf = cmsg;
                }
                
                CProxy_ComlibManager cmgr(thisgroup);
                cmgr[0].receiveNamdMessage(newmsg); // yet to decide!
		*/
	    }
	    else if(strategy == USE_MPI){
#if CHARM_MPI
		ComlibPrintf("[%d] In MPI strategy\n", CkMyPe());
		
		CharmMessageHolder *cmsg = messageBuf;
		char *buf_ptr = mpi_sndbuf;

		//if(npes == 0)
		//  npes = CkNumPes();

		for(count = 0; count < npes; count ++) {
		    ((int *)buf_ptr)[0] = 0;
		    buf_ptr += MPI_MAX_MSG_SIZE;
		}

		buf_ptr = mpi_sndbuf;
                for(count = 0; count < messageCount; count ++) {
		    if(npes < CkNumPes()) {
			ComlibPrintf("[%d] Copying data to %d and rank %d\n", cmsg->dest_proc, 
				     procMap[cmsg->dest_proc]);
			buf_ptr = mpi_sndbuf + MPI_MAX_MSG_SIZE * procMap[cmsg->dest_proc];  
		    }
		    else
			buf_ptr = mpi_sndbuf + MPI_MAX_MSG_SIZE * cmsg->dest_proc; 
		    
		    char * msg = cmsg->getCharmMessage();
		    envelope * env = UsrToEnv(msg);
		    
		    ((int *)buf_ptr)[0] = env->getTotalsize();

		    ComlibPrintf("[%d] Copying message\n", CkMyPe());
		    memcpy(buf_ptr + sizeof(int), (char *)env, env->getTotalsize());

		    ComlibPrintf("[%d] Deleting message\n", CkMyPe());
		    CmiFree((char *) env);
		    CharmMessageHolder *prev = cmsg;
		    cmsg = cmsg->next;
		    delete prev;
		}

		//ComlibPrintf("[%d] Calling Barrier\n", CkMyPe());
		//PMPI_Barrier(groupComm);
		
		ComlibPrintf("[%d] Calling All to all\n", CkMyPe());
		PMPI_Alltoall(mpi_sndbuf, MPI_MAX_MSG_SIZE, MPI_CHAR, mpi_recvbuf, 
			     MPI_MAX_MSG_SIZE, MPI_CHAR, groupComm);
		
		ComlibPrintf("[%d] All to all finished\n", CkMyPe());
		buf_ptr = mpi_recvbuf;
		for(count = 0; count < npes; count ++) {
		    int recv_msg_size = ((int *)buf_ptr)[0];
		    char * recv_msg = buf_ptr + sizeof(int);
		    
		    if((recv_msg_size > 0) && recv_msg_size < MPI_MAX_MSG_SIZE) {
			ComlibPrintf("[%d] Receiving message of size %d\n", CkMyPe(), recv_msg_size);
			CmiSyncSend(CmiMyPe(), recv_msg_size, recv_msg);
		    }
		    buf_ptr += MPI_MAX_MSG_SIZE;
		}
#endif
	    }
	    else if(strategy == USE_STREAMING){
		ComlibPrintf("[%d] In Streaming strategy\n", CkMyPe());
		
		CharmMessageHolder *cmsg = messageBuf;

                int buf_size = 0;
                for(count = 0; count < CkNumPes(); count ++) {
                    if(streamingMsgCount[count] == 0)
                        continue;

                    cmsg = streamingMsgBuf[count];
                    char ** msgComps = new char*[streamingMsgCount[count]];
                    int *sizes = new int[streamingMsgCount[count]];
                    
                    int msg_count = 0;
                    while (cmsg != NULL) {
                        char * msg = cmsg->getCharmMessage();
                        envelope * env = UsrToEnv(msg);
                        sizes[msg_count] = env->getTotalsize();
                        msgComps[msg_count] = (char *)env;

                        cmsg = cmsg->next;
                        msg_count ++;
                    }
                    
                    CmiMultipleSend(count, streamingMsgCount[count], sizes, msgComps);
                    delete [] msgComps;
                    delete [] sizes;
                }

                for(count = 0; count < CkNumPes(); count ++){
                    streamingMsgBuf[count] = NULL;
                    streamingMsgCount[count] = 0;
                }

                beginIteration();
                CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
	    }
	    else{
                ComlibPrintf("%d:Setting Num Deposit to %d\n", CkMyPe(), messageCount);
                NumDeposits(comid, messageCount);
                
                CharmMessageHolder *cmsg = messageBuf;
                for(count = 0; count < messageCount; count ++) {
                    if(strategy != USE_DIRECT) {
                        char * msg = cmsg->getCharmMessage();
                        ComlibPrintf("Calling EachToMany %d %d %d procMap=%d\n", 
                                     UsrToEnv(msg)->getTotalsize(), CkMyPe(), 
				     cmsg->dest_proc, procMap[cmsg->dest_proc]);
                        EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
					    UsrToEnv(msg), 1, 
					    &procMap[cmsg->dest_proc]);
                    }
                    CharmMessageHolder *prev = cmsg;
                    cmsg = cmsg->next;
                    if(prev != NULL)
                        delete prev;                //foobar getrid of the delete
                    
                }
            }
            messageCount = 0;
        }
        elementCount = 0;
    }
}


void ComlibManager::receiveID(comID id){
    receiveID(CkNumPes(), NULL, id);
}

void ComlibManager::receiveID(int npes, int *pelist, comID id){

  if(npes != CkNumPes()) {
    this->npes = npes;
    this->pelist = new int[npes];
    
    memcpy(this->pelist, pelist, sizeof(int) * npes);
  }

#if CHARM_MPI
    if(npes < CkNumPes()){
        PMPI_Comm_group(MPI_COMM_WORLD, &groupWorld);
	PMPI_Group_incl(groupWorld, npes, pelist, &group);
	PMPI_Comm_create(MPI_COMM_WORLD, group, &groupComm);
    }
    else groupComm = MPI_COMM_WORLD;
#endif

    ComlibPrintf("received id in %d, npes = %d\n", CkMyPe(), npes);
    int count = 0;

    if(idSet)
        return;

    if(npes != CkNumPes())
	setReverseMap(pelist, npes);

    comid = id; 
    idSet = 1;

    if(strategy == USE_STREAMING) {
        CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
        return;
    }   

    if(iterationFinished){

        if(strategy == NAMD_STRAT) {
	  // In the Namd Strategy half the messages are sent to a remote 
	  //processor and the remaining half are sent directly.

            CharmMessageHolder *cmsg = messageBuf;
            envelope *env = (envelope *)cmsg->data;
            for(count = 0; count < messageCount/2; count ++) {
                CkSendMsgBranch(env->getEpIdx(), cmsg->getCharmMessage(), 
				cmsg->dest_proc, env->getGroupNum());
                cmsg = cmsg->next;
            }

            CharmMessageHolder *tmpbuf = cmsg;
            int tot_size = 0;
            for(count = messageCount/2; count < messageCount; count ++) {
                tot_size += cmsg->getSize();
                cmsg =  cmsg->next;
            }

            ComlibMsg *newmsg = new(&tot_size, 0) ComlibMsg;
            
            cmsg = tmpbuf;
            for(count = messageCount/2; count < messageCount; count ++) {
                newmsg->insert(cmsg);
                delete tmpbuf;
                cmsg = cmsg->next;
                tmpbuf = cmsg;
            }

            CProxy_ComlibManager cmgr(thisgroup);
            cmgr[/*yet to decide*/0].receiveNamdMessage(newmsg);
	}
	else if(strategy == USE_MPI) {
#if CHARM_MPI
	    CharmMessageHolder *cmsg = messageBuf;
	    char *buf_ptr = mpi_sndbuf;
	    
	    //if(npes == 0)
	    //npes = CkNumPes();
	    
	    for(count = 0; count < npes; count ++) {
		((int *)buf_ptr)[0] = 0;
		buf_ptr += MPI_MAX_MSG_SIZE;
	    }
	    
	    buf_ptr = mpi_sndbuf;
	    for(count = 0; count < messageCount; count ++) {
		if(npes < CkNumPes())
		    buf_ptr = mpi_sndbuf + MPI_MAX_MSG_SIZE * procMap[cmsg->dest_proc];
		else
		    buf_ptr = mpi_sndbuf + MPI_MAX_MSG_SIZE * cmsg->dest_proc; 

		char * msg = cmsg->getCharmMessage();
		envelope * env = UsrToEnv(msg);
		
		((int *)buf_ptr)[0] = env->getTotalsize();
		memcpy(buf_ptr + sizeof(int), (char *)env, env->getTotalsize());
		CmiFree((char *) env);
		CharmMessageHolder *prev = cmsg;
		cmsg = cmsg->next;
		delete prev;
	    }
	    
	    PMPI_Alltoall(mpi_sndbuf, MPI_MAX_MSG_SIZE, MPI_CHAR, mpi_recvbuf, 
			  MPI_MAX_MSG_SIZE, MPI_CHAR, groupComm);
	    
	    ComlibPrintf("[%d] All to all finished\n", CkMyPe());
	    buf_ptr = mpi_recvbuf;
	    for(count = 0; count < npes; count ++) {
		int recv_msg_size = ((int *)buf_ptr)[0];
		char * recv_msg = buf_ptr + sizeof(int);
		
		if((recv_msg_size > 0) && recv_msg_size < MPI_MAX_MSG_SIZE) {
		    ComlibPrintf("[%d] Receiving message of size %d\n", CkMyPe(), recv_msg_size);
		    CmiSyncSend(CmiMyPe(), recv_msg_size, recv_msg);
		}
		buf_ptr += MPI_MAX_MSG_SIZE;
	    }
#endif
        }
        /*
        else if(strategy == USE_STREAMING){
		ComlibPrintf("[%d] In Streaming strategy\n", CkMyPe());
		
		CharmMessageHolder *cmsg = messageBuf;

                int buf_size = 0;
                for(count = 0; count < CkNumPes(); count ++) {
                    if(streamingMsgCount[count] == 0)
                        continue;

                    cmsg = streamingMsgBuf[count];
                    char ** msgComps = new char*[streamingMsgCount[count]];
                    int *sizes = new int[streamingMsgCount[count]];
                    
                    int msg_count = 0;
                    while (cmsg != NULL) {
                        char * msg = cmsg->getCharmMessage();
                        envelope * env = UsrToEnv(msg);
                        sizes[msg_count] = env->getTotalsize();
                        msgComps[msg_count] = (char *)env;

                        cmsg = cmsg->next;
                        msg_count ++;
                    }
                    
                    CmiMultipleSend(count, streamingMsgCount[count], sizes, msgComps);
                    beginIteration();

                    CcdCallFnAfter(call_endIteration, (void *)this, PERIOD);
                }
        }
        */
	else {
	    
	    NumDeposits(comid, messageCount);
	    ComlibPrintf("Setting Num Deposit to %d\n", messageCount);
	    
            CharmMessageHolder *cmsg = messageBuf;
            for(count = 0; count < messageCount; count ++) {
                if(strategy != USE_DIRECT) {
                    char * msg = cmsg->getCharmMessage();
		    
                    ComlibPrintf("Calling EachToMany %d %d %d procMap=%d\n", 
                                 UsrToEnv(msg)->getTotalsize(), CkMyPe(), 
				 cmsg->dest_proc,
                                 procMap[cmsg->dest_proc]);
		    
                    EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
					UsrToEnv(msg), 1, &procMap[cmsg->dest_proc]);
                }
                CharmMessageHolder *prev = cmsg;
                cmsg = cmsg->next;
                if(prev != NULL)
                    delete prev;                //foobar getrid of the delete
            }
        }
	
        messageCount = 0;
        messageBuf = NULL;
    }
}

extern int _charmHandlerIdx;
void ComlibManager::ArraySend(int ep, void *msg, 
                              const CkArrayIndexMax &idx, CkArrayID a){
    
    CkArrayIndexMax myidx = idx;
    int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(myidx);
    
    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, 
		 UsrToEnv(msg)->getTotalsize());

    if(dest_proc == CkMyPe()){
        //CkArrayID::CkLocalBranch(a)->deliverViaQueue((CkArrayMessage *) msg);
        CProxyElement_ArrayBase ap(a,idx);
        ap.ckSend((CkArrayMessage *)msg, ep);
        return;
    }

    register envelope * env = UsrToEnv(msg);
    
    env->array_mgr()=a;
    env->array_srcPe()=CkMyPe();
    env->array_ep()=ep;
    env->array_hops()=0;
    env->array_index()=idx;
    CkPackMessage(&env);
    //    CmiSetHandler(env, _charmHandlerIdx);
    CmiSetHandler(env, CpvAccess(RecvmsgHandle));
    
    if(strategy == USE_DIRECT) {
        ComlibPrintf("Sending Directly\n");
        CmiSyncSendAndFree(dest_proc, UsrToEnv(msg)->getTotalsize(), (char *)UsrToEnv(msg));
        return;
    }

    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.
    
    /*
    cmsg->aid = aid;
    cmsg->idx = idx;
    cmsg->ep = ep;
    */
    messageCount ++;

    totalBytes += UsrToEnv(msg)->getTotalsize();

    if(strategy != USE_STREAMING) {
        cmsg->next = messageBuf;
        messageBuf = cmsg;    
    }
    else {
        cmsg->next = streamingMsgBuf[cmsg->dest_proc];
        streamingMsgBuf[cmsg->dest_proc] = cmsg;
        streamingMsgCount[cmsg->dest_proc] ++;
    }
}

#include "qd.h"
//CpvExtern(QdState*, _qd);

void ComlibManager::GroupSend(int ep, void *msg, int onPE, CkGroupID gid){
    
    int dest_proc = onPE;
    ComlibPrintf("Send Data %d %d %d\n", CkMyPe(), dest_proc, UsrToEnv(msg)->getTotalsize());

    register envelope * env = UsrToEnv(msg);
    if(dest_proc == CkMyPe()){
        _SET_USED(env, 0);
        CkSendMsgBranch(ep, msg, dest_proc, gid);
        return;
    }
    
    CpvAccess(_qd)->create(1);

    env->setMsgtype(ForBocMsg);
    env->setEpIdx(ep);
    env->setGroupNum(gid);
    env->setSrcPe(CkMyPe());
    
    CkPackMessage(&env);
    CmiSetHandler(env, _charmHandlerIdx);

    if(strategy == USE_DIRECT) {
        ComlibPrintf("Sending Directly\n");
        CmiSyncSendAndFree(dest_proc, UsrToEnv(msg)->getTotalsize(), (char *)UsrToEnv(msg));
        return;
    }
    
    CharmMessageHolder *cmsg = new CharmMessageHolder((char *)msg, dest_proc); 
    //get rid of the new.
    
    /*
    cmsg->gid = gid;
    cmsg->ep = ep;
    */
    messageCount ++;
    
    if(strategy != USE_STREAMING) {
        cmsg->next = messageBuf;
        messageBuf = cmsg;    
    }
    else {
        cmsg->next = streamingMsgBuf[cmsg->dest_proc];
        streamingMsgBuf[cmsg->dest_proc] = cmsg;
        streamingMsgCount[cmsg->dest_proc] ++;
    }
}

void ComlibManager::receiveNamdMessage(ComlibMsg * msg){
    CharmMessageHolder *cmsg = msg->next();
    char *charm_msg = cmsg->getCharmMessage();

    CkPrintf("In receive namd message\n");

    while(charm_msg != NULL){
        register envelope * env = UsrToEnv(charm_msg);
        CkSendMsgBranch(env->getEpIdx(), charm_msg, cmsg->dest_proc, env->getGroupNum());

        cmsg = msg->next();
        charm_msg = cmsg->getCharmMessage();
    }
    
    CkPrintf("After receive namd message\n");

    delete msg;
}

void ComlibManager::learnPattern(int total_msg_count, int total_bytes) {
    static int nrecvd = 0;
    static double avg_message_count = 0;
    static double avg_message_bytes = 0;

    avg_message_count += ((double) total_msg_count) / LEARNING_PERIOD;
    avg_message_bytes += ((double) total_bytes) /  LEARNING_PERIOD;

    nrecvd ++;
    
    if(nrecvd == CmiNumPes()) {
        //Number of messages and bytes a processor sends in each iteration
        avg_message_count /= CmiNumPes();
        avg_message_bytes /= CmiNumPes();
        
        CkPrintf("STATS = %5.3lf, %5.3lf", avg_message_count, avg_message_bytes);

        //Learning, ignoring contention for now! 
        double cost_dir, cost_mesh, cost_grid, cost_hyp;
        cost_dir = ALPHA * avg_message_count + BETA * avg_message_bytes;
        cost_mesh = ALPHA * 2 * sqrt(CmiNumPes()) + BETA * avg_message_bytes * 2;
        cost_grid = ALPHA * 3 * cbrt(CmiNumPes()) + BETA * avg_message_bytes * 3;
        cost_hyp =  (log(CmiNumPes())/log(2.0))*(ALPHA  + BETA * avg_message_bytes/2.0);
        
        // Find the one with the minimum cost!
        int min_strat = USE_MESH; 
        double min_cost = cost_mesh;
        if(min_cost > cost_hyp)
            min_strat = USE_HYPERCUBE;
        if(min_cost > cost_grid)
            min_strat = USE_GRID;

        if(min_cost > cost_dir)
            min_strat = USE_DIRECT;

        switchStrategy(min_strat);        
    }
}

void ComlibManager::switchStrategy(int strat){
    CkPrintf("Switching to %d\n", strat);
}

CharmMessageHolder::CharmMessageHolder(char * msg, int proc) {
    data = (char *)UsrToEnv(msg);
    dest_proc = proc;
}

void CharmMessageHolder::init(char * root_msg) {

    data = (char *)this + sizeof(CharmMessageHolder);
    
    register envelope * env = (envelope *)data;
    CkUnpackMessage(&env);

    setRefcount(root_msg);
}

char * CharmMessageHolder::getCharmMessage(){
    return (char *)EnvToUsr((envelope *) data);
}

void CharmMessageHolder::copy(char *buf){
    *((CharmMessageHolder *)buf) = *this;
    memcpy(buf + sizeof(CharmMessageHolder), data, ((envelope *)data)->getTotalsize());
}

int CharmMessageHolder::getSize(){
    return sizeof(CharmMessageHolder) + ((envelope *)data)->getTotalsize();
}

#define REFFIELD(m) ((int *)((char *)(m)-sizeof(int)))[0]
void CharmMessageHolder::setRefcount(char *msg){
    char *env = (char *)UsrToEnv(msg);
    int pref = REFFIELD(env);
    int ref =  REFFIELD(data);
    
    while (pref < 0) {
        env = env + pref;
        pref =  REFFIELD(env);
    }
    
    pref ++;
    ref = (unsigned int)((char *) env - (char *) data);
    
    REFFIELD(data) = ref;
    REFFIELD(env) = pref;    
}

void ComlibMsg::insert(CharmMessageHolder *msg){
    nmessages ++;
    
    msg->copy(data + curSize);
    curSize += msg->getSize();
    //CkPrintf("CurSize = %d\n", curSize);
}

CharmMessageHolder * ComlibMsg::next(){
    
    if(nmessages == 0){
        CkPrintf("ComlibMsg::next OVEFLOW\n");
        return NULL;
    }

    nmessages --;
    CharmMessageHolder *cmsg = (CharmMessageHolder *)(data + curSize);
    
    cmsg->init((char *) this);
    curSize += cmsg->getSize();
    return cmsg;
}

#include "ComlibModule.def.h"
