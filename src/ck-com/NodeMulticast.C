// #ifdef filippo

// #include "NodeMulticast.h"
// #include "converse.h"

// #define MAX_BUF_SIZE 165000
// #define MAX_SENDS_PER_BATCH 16
// #define MULTICAST_DELAY 5

// static NodeMulticast *nm_mgr;

// static void call_doneInserting(void *ptr,double curWallTime){
//     NodeMulticast *mgr = (NodeMulticast *)ptr;
//     mgr->doneInserting();
// }

// static void* NodeMulticastHandler(void *msg){
//     ComlibPrintf("In Node MulticastHandler\n");
//     nm_mgr->recvHandler(msg);
//     return NULL;
// }

// static void* NodeMulticastCallbackHandler(void *msg){
//     ComlibPrintf("[%d]:In Node MulticastCallbackHandler\n", CkMyPe());
//     register envelope *env = (envelope *)msg;
//     CkUnpackMessage(&env);
//     //nm_mgr->getCallback().send(EnvToUsr(env));

//     //nm_mgr->getHandler()(env);
//     return NULL;
// }

// //Handles multicast by sending only one message to a nodes and making 
// //them multicast locally
// void NodeMulticast::setDestinationArray(CkArrayID a, int nelem, 
// 					CkArrayIndex **idx, int ep){

//     mode = ARRAY_MODE;
//     messageBuf = NULL;
//     pes_per_node = 4;
//     if(getenv("RMS_NODES") != NULL)
//         pes_per_node = CkNumPes()/atoi(getenv("RMS_NODES"));

//     mAid = a;
//     nelements = nelem;
//     entryPoint = ep;
  
//     numNodes = CkNumPes()/pes_per_node;
//     numCurDestPes = CkNumPes();
//     myRank = 0;
//     nodeMap = new int[numNodes];
  
//     ComlibPrintf("In SetDestinationArray %d, %d, %d, %d\n", numNodes, 
//                  pes_per_node, nelements, ep);
  
//     indexVec = new CkVec<CkArrayIndex> [CkNumPes()];
    
//     for(int count = 0; count < nelements; count++) {
//         ComlibPrintf("Before lastKnown %d\n", count);
//         int dest_proc = CkArrayID::CkLocalBranch(a)->lastKnown(*idx[count]);
//         ComlibPrintf("After lastKnown %d\n", dest_proc);
//         nodeMap[dest_proc/pes_per_node] = 1;
        
//         indexVec[dest_proc].insertAtEnd(*idx[count]);
//     }    

//     ComlibPrintf("After SetDestinationArray\n");
// }
// /*
// void NodeMulticast::setPeList(int npes, int *pelist, ComlibMulticastHandler handler){
//     mode = PROCESSOR_MODE;
//     messageBuf = NULL;
//     pes_per_node = 4;
//     //if(getenv("RMS_NODES") != NULL)
//     //pes_per_node = CkNumPes()/atoi(getenv("RMS_NODES"));

//     //cb = callback;
//     this->handler = (long)handler;
  
//     numNodes = CkNumPes()/pes_per_node;
//     numCurDestPes = npes;
    
//     myRank = 0;
//     nodeMap = new int[numNodes];
  
//     this->npes = npes;
//     this->pelist = new int[npes];
//     memcpy(this->pelist, pelist, npes * sizeof(int));

//     ComlibPrintf("In setPeList %d, %d, %d\n", numNodes, 
//                  pes_per_node, npes);
    
//     for(int count = 0; count < npes; count++)
//         nodeMap[pelist[count]/pes_per_node] = 1;        
    
//     ComlibPrintf("After setPeList\n");
// }
// */

// void NodeMulticast::recvHandler(void *msg) {
//     register envelope* env = (envelope *)msg;
//     void *charm_msg = (void *)EnvToUsr(env);

//     env->setUsed(0);
//     ComlibPrintf("In receive Handler\n");
//     if(mode == ARRAY_MODE) {
//         env->setArrayMgr(mAid);
// 	env->getsetArrayEp()=entryPoint;
// 	env->getsetArrayHops()=0;	
// 	CkUnpackMessage(&env);

// 	for(int count = 0; count < pes_per_node; count ++){
// 	    int dest_pe = (CkMyPe()/pes_per_node) * pes_per_node + count;
// 	    int size = indexVec[dest_pe].size();
	    
// 	    ComlibPrintf("[%d], %d elements to send to %d of size %d\n", CkMyPe(), size, dest_pe, env->getTotalsize());
	    
// 	    CkArrayIndex * idx_arr = indexVec[dest_pe].getVec();
// 	    for(int itr = 0; itr < size; itr ++) {
// 		void *newcharmmsg = CkCopyMsg(&charm_msg); 
// 		envelope* newenv = UsrToEnv(newcharmmsg);
// 		CProxyElement_ArrayBase ap(mAid, idx_arr[itr]);		
// 		newenv->getsetArrayIndex()=idx_arr[itr];
// 		ap.ckSend((CkArrayMessage *)newcharmmsg, entryPoint);
// 	    }
// 	}
//     }
//     else {
//       CkUnpackMessage(&env);
//       for(int count = 0; count < pes_per_node; count++) 
// 	if(validRank[count]){
//             void *newcharmmsg;
//             envelope* newenv;
	  
//             if(count <  pes_per_node - 1) {
//                 newcharmmsg = CkCopyMsg(&charm_msg); 
//                 newenv = UsrToEnv(newcharmmsg);
//             }
//             else {
//                 newcharmmsg = charm_msg;
//                 newenv = UsrToEnv(newcharmmsg);
//             }

//             CmiSetHandler(newenv, NodeMulticastCallbackHandlerId);
//             ComlibPrintf("[%d] In receive Handler (proc mode), sending message to %d at handler %d\n", 
//                          CkMyPe(), (CkMyPe()/pes_per_node) * pes_per_node 
//                          + count, NodeMulticastCallbackHandlerId);
            
//             CkPackMessage(&newenv);
//             CmiSyncSendAndFree((CkMyPe()/pes_per_node) *pes_per_node + count, 
//                                newenv->getTotalsize(), (char *)newenv);
// 	}
//     }
//     ComlibPrintf("[%d] CmiFree (Code) (%x)\n", CkMyPe(), 
//                  (long) msg - 2*sizeof(int));
//     //CmiFree(msg);
// }

// void NodeMulticast::insertMessage(CharmMessageHolder *cmsg){

//     ComlibPrintf("In insertMessage \n");
//     envelope *env = UsrToEnv(cmsg->getCharmMessage());

//     CmiSetHandler(env, NodeMulticastHandlerId);
//     messageBuf->enq(cmsg);
// }

// void NodeMulticast::doneInserting(){
//     CharmMessageHolder *cmsg;
//     char *msg;
//     register envelope *env;
    
//     ComlibPrintf("NodeMulticast :: doneInserting\n");
    
//     if(messageBuf->length() > 1) {
//         //CkPrintf("NodeMulticast :: doneInserting length > 1\n");
//         /*
//         char **msgComps;
//         int *sizes, msg_count;
    
//         msgComps = new char*[messageBuf->length()];
//         sizes = new int[messageBuf->length()];
//         msg_count = 0;
//         while (!messageBuf->isEmpty()) {
//             cmsg = messageBuf->deq();
//             msg = cmsg->getCharmMessage();
//             env = UsrToEnv(msg);
//             sizes[msg_count] = env->getTotalsize();
//             msgComps[msg_count] = (char *)env;
//             msg_count++;
            
//             delete cmsg;
//         }
        
//         for(int count = 0; count < numNodes; count++)
//             if(nodeMap[count])
//                 CmiMultipleSend(count * pes_per_node + myRank, msg_count, 
//                                 sizes, msgComps);
        
//         delete [] msgComps;
//         delete [] sizes;
//         */
//     }
//     else if (messageBuf->length() == 1){
//         static int prevCount = 0;
//         int count = 0;
//         ComlibPrintf("Sending Node Multicast\n");
//         cmsg = messageBuf->deq();
//         msg = cmsg->getCharmMessage();
//         env = UsrToEnv(msg);
	
// 	if(mode == ARRAY_MODE)
// 	    env->getsetArraySrcPe()=CkMyPe();
// 	CkPackMessage(&env);

//         CmiSetHandler(env, NodeMulticastHandlerId);
//         ComlibPrintf("After set handler\n");

//         //CmiPrintf("cursedtpes = %d, %d\n", cmsg->npes, numCurDestPes);
        
//         if((mode != ARRAY_MODE) && cmsg->npes < numCurDestPes) {
//             numCurDestPes = cmsg->npes;
//             for(count = 0; count < numNodes; count++) 
//                 nodeMap[count] = 0;        
            
//             for(count = 0; count < cmsg->npes; count++) 
//                 nodeMap[(cmsg->pelist[count])/pes_per_node] = 1;        
//         }
        
//         for(count = prevCount; count < numNodes; count++) {
// 	    //int dest_node = count;
// 	    int dest_node = (count + (CkMyPe()/pes_per_node))%numNodes;
// 	    if(nodeMap[dest_node]) {
//                 void *newcharmmsg;
//                 envelope* newenv;
                
//                 if(count < numNodes - 1) {
//                     newcharmmsg = CkCopyMsg((void **)&msg); 
//                     newenv = UsrToEnv(newcharmmsg);
//                 }
//                 else {
//                     newcharmmsg = msg;
//                     newenv = UsrToEnv(newcharmmsg);
//                 }
		
// 		ComlibPrintf("[%d]In cmisyncsend to %d\n", CkMyPe(), 
// 			     dest_node * pes_per_node + myRank);
// #if CMK_PERSISTENT_COMM
// 		if(env->getTotalsize() < MAX_BUF_SIZE)
//                     CmiUsePersistentHandle(&persistentHandlerArray[dest_node],1);
// #endif
// 		CkPackMessage(&newenv);
// 		CmiSyncSendAndFree(dest_node * pes_per_node + myRank, 
// 				   newenv->getTotalsize(), (char *)newenv);
// #if CMK_PERSISTENT_COMM
// 		if(env->getTotalsize() < MAX_BUF_SIZE)
//                     CmiUsePersistentHandle(NULL, 0);
// #endif          
//             }
//             prevCount ++;
//             if((prevCount % MAX_SENDS_PER_BATCH == 0) &&
//                (prevCount != numNodes)) {
//                 CcdCallFnAfterOnPE((CcdVoidFn)call_doneInserting, (void *)this, 
//                                MULTICAST_DELAY, CkMyPe());
//                 return;
//             }
//             prevCount = 0;
// 	}

//         ComlibPrintf("[%d] CmiFree (Code) (%x)\n", CkMyPe(), (char *)env - 2*sizeof(int));
//         //CmiFree(env);
//         delete cmsg;
//     }
// }

// void NodeMulticast::pup(PUP::er &p){
    
//     CharmStrategy::pup(p);

//     p | pes_per_node;
//     p | numNodes;
//     p | nelements;
//     p | entryPoint;
//     p | npes;
//     p | mode;
//     p | numCurDestPes;
//     p | mAid;
    
//     if(p.isUnpacking()) {
//         nodeMap = new int[numNodes];
	
// 	if(mode == ARRAY_MODE) {
// 	    typedef CkVec<CkArrayIndex> CkVecArrayIndex;
// 	    CkVecArrayIndex *vec = new CkVecArrayIndex[CkNumPes()];
// 	    indexVec = vec;
// 	}

// 	if(mode == PROCESSOR_MODE)
// 	    pelist = new int[npes];
//     }

//     p | cb;
//     p | handler;
//     p(nodeMap, numNodes);

//     if(mode == PROCESSOR_MODE)
//       p(pelist, npes);

//     if(mode == ARRAY_MODE)
//       for(int count = 0; count < CkNumPes(); count++)
//         p | indexVec[count];
    
//     if(p.isUnpacking()) {
//         messageBuf = new CkQ <CharmMessageHolder *>;
//         myRank = CkMyPe() % pes_per_node;
        
//         NodeMulticastHandlerId = CkRegisterHandler((CmiHandler)NodeMulticastHandler);
// 	NodeMulticastCallbackHandlerId = CkRegisterHandler
// 	    ((CmiHandler)NodeMulticastCallbackHandler);
	
//         nm_mgr = this;

// 	//validRank[0] =  validRank[1] = validRank[2] = validRank[3] = 0;
//         memset(validRank, 0, MAX_PES_PER_NODE * sizeof(int));
// 	for(int count = 0; count < npes; count ++){
// 	    if(CkMyPe()/pes_per_node == pelist[count] / pes_per_node)
// 		validRank[pelist[count] % pes_per_node] = 1;
// 	}

// #if CMK_PERSISTENT_COMM
// 	persistentHandlerArray = new PersistentHandle[numNodes];
// 	for(int count = 0; count < numNodes; count ++)
//             //if(nodeMap[count])
//             persistentHandlerArray[count] = CmiCreatePersistent
//                 (count * pes_per_node + myRank, MAX_BUF_SIZE);
// #endif
//     }
// }

// //PUPable_def(NodeMulticast);

// #endif
