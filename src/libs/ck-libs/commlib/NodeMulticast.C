#include "NodeMulticast.h"
#include "converse.h"

NodeMulticast *nm_mgr;

void* NodeMulticastHandler(void *msg){
  ComlibPrintf("In Node MulticastHandler\n");
  nm_mgr->recvHandler(msg);
  return NULL;
}

//Handles multicast by sending only one message to a nodes and making 
//them multicast locally
void NodeMulticast::setDestinationArray(CkArrayID a, int nelem, 
					CkArrayIndexMax **idx, int ep){
  
    messageBuf = NULL;
    pes_per_node = 4;
    //if(getenv("RMS_NODES") != NULL)
    //pes_per_node = CkNumPes()/atoi(getenv("RMS_NODES"));

    mAid = a;
    nelements = nelem;
    entryPoint = ep;
  
    numNodes = CkNumPes()/pes_per_node;
    myRank = 0;
    nodeMap = new int[numNodes];
  
    ComlibPrintf("In SetDestinationArray %d, %d, %d, %d\n", numNodes, 
                 pes_per_node, nelements, ep);
  
    indexVec = new CkVec<CkArrayIndexMax> [CkNumPes()];
    
    for(int count = 0; count < nelements; count++) {
        ComlibPrintf("Before lastKnown %d\n", count);
        int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(*idx[count]);
        ComlibPrintf("After lastKnown %d\n", dest_proc);
        nodeMap[dest_proc/pes_per_node] = 1;
        
        indexVec[dest_proc].insertAtEnd(*idx[count]);
    }
    

    ComlibPrintf("After SetDestinationArray\n");
}

void NodeMulticast::recvHandler(void *msg) {
    register envelope* env = (envelope *)msg;
    void *charm_msg = (void *)EnvToUsr(env);

    env->setUsed(0);
    env->array_mgr()=mAid;
    //  env->array_srcPe()=CkMyPe();
    env->array_ep()=entryPoint;
    env->array_hops()=0;

    //CkUnpackMessage(&env);
    
    ComlibPrintf("In receive Handler\n");
    
    for(int count = 0; count < pes_per_node; count ++){
        int dest_pe = (CkMyPe()/pes_per_node) * pes_per_node + count;
        int size = indexVec[dest_pe].size();
        
        ComlibPrintf("[%d], %d elements to send to %d of size %d\n", CkMyPe(), size, dest_pe, env->getTotalsize());
        
        CkArrayIndexMax * idx_arr = indexVec[dest_pe].getVec();
        for(int itr = 0; itr < size; itr ++) {
            void *newcharmmsg = CkCopyMsg(&charm_msg); //(char *)CmiAlloc(env->getTotalsize());
            //memcpy(newmsg, msg, env->getTotalsize());
            envelope* newenv = UsrToEnv(newcharmmsg);
            
            //CkArrayIndex1D idx(dest_pe);
            //idx_arr[itr].print();
            CProxyElement_ArrayBase ap(mAid, idx_arr[itr]);
            ComlibPrintf("%d:Array Base created %x\n", CkMyPe(), (char *)newenv -  2*sizeof(int));
            
            newenv->array_index()=idx_arr[itr];
            //CmiSetHandler(env, 0);
            ap.ckSend((CkArrayMessage *)newcharmmsg, entryPoint);
        }
    }
    ComlibPrintf("[%d] CmiFree (Code) (%x)\n", CkMyPe(), (long) msg - 2*sizeof(int));
    CmiFree(msg);
}

void NodeMulticast::insertMessage(CharmMessageHolder *cmsg){

    ComlibPrintf("In insertMessage \n");
    envelope *env = UsrToEnv(cmsg->getCharmMessage());

    CmiSetHandler(env, NodeMulticastHandlerId);
    messageBuf->enq(cmsg);
}

void NodeMulticast::doneInserting(){
    CharmMessageHolder *cmsg;
    char *msg;
    register envelope *env;
    
    ComlibPrintf("NodeMulticast :: doneInserting\n");
    
    if(messageBuf->length() > 1) {
        /*
        char **msgComps;
        int *sizes, msg_count;
    
        msgComps = new char*[messageBuf->length()];
        sizes = new int[messageBuf->length()];
        msg_count = 0;
        while (!messageBuf->isEmpty()) {
            cmsg = messageBuf->deq();
            msg = cmsg->getCharmMessage();
            env = UsrToEnv(msg);
            sizes[msg_count] = env->getTotalsize();
            msgComps[msg_count] = (char *)env;
            msg_count++;
            
            delete cmsg;
        }
        
        for(int count = 0; count < numNodes; count++)
            if(nodeMap[count])
                CmiMultipleSend(count * pes_per_node + myRank, msg_count, 
                                sizes, msgComps);
        
        delete [] msgComps;
        delete [] sizes;
        */
    }
    else if (messageBuf->length() == 1){
        ComlibPrintf("Sending Node Multicast\n");
        cmsg = messageBuf->deq();
        msg = cmsg->getCharmMessage();
        env = UsrToEnv(msg);
        env->array_srcPe()=CkMyPe();

        CmiSetHandler(env, NodeMulticastHandlerId);
        ComlibPrintf("After set handler\n");
        
        for(int count = 0; count < numNodes; count++) 
            if(nodeMap[count]) {
                char *newmsg = (char *)CmiAlloc(env->getTotalsize());
                memcpy(newmsg, (char *)env, env->getTotalsize());

                ComlibPrintf("In cmisyncsend to %d\n", count * pes_per_node + myRank);
                CmiSyncSendAndFree(count * pes_per_node + myRank, env->getTotalsize(), newmsg);
            }
        
        ComlibPrintf("[%d] CmiFree (Code) (%x)\n", CkMyPe(), (char *)env - 2*sizeof(int));
        CmiFree(env);
        delete cmsg;
    }
}

void NodeMulticast::pup(PUP::er &p){
    
    Strategy::pup(p);

    p | pes_per_node;
    p | numNodes;
    p | nelements;
    p | entryPoint;
    
    p | mAid;
    
    if(p.isUnpacking()) {
        nodeMap = new int[numNodes];
        indexVec = new CkVec<CkArrayIndexMax> [CkNumPes()];
    }
    
    p(nodeMap, numNodes);
    
    for(int count = 0; count < CkNumPes(); count++)
        p | indexVec[count];
    
    if(p.isUnpacking()) {
        messageBuf = new CkQ <CharmMessageHolder *>;
        myRank = CkMyPe() % pes_per_node;
        
        NodeMulticastHandlerId = CmiRegisterHandler((CmiHandler)NodeMulticastHandler);
        nm_mgr = this;
    }
}

//PUPable_def(NodeMulticast);
