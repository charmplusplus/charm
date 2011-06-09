// #ifdef filippo

// #include "KDirectMulticastStrategy.h"

// //Group Constructor
// KDirectMulticastStrategy::KDirectMulticastStrategy(int kf, 
//                                                    int ndest, int *pelist) 
//     : DirectMulticastStrategy(ndest, pelist), kfactor(kf) {
//     //FIXME: verify the list is sorted
//     commonKDirectInit();
// }

// //Array Constructor
// KDirectMulticastStrategy::KDirectMulticastStrategy(int kf, 
//                                                    CkArrayID dest_aid)
//     : DirectMulticastStrategy(dest_aid), kfactor(kf){
//     commonKDirectInit();    
// }

// void KDirectMulticastStrategy::commonKDirectInit(){
//     //sort list and create a reverse map
// }

// extern int _charmHandlerIdx;
// void KDirectMulticastStrategy::doneInserting(){
//     ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
//     if(messageBuf->length() == 0) {
//         return;
//     }
    
//     while(!messageBuf->isEmpty()) {
// 	CharmMessageHolder *cmsg = messageBuf->deq();
//         char *msg = cmsg->getCharmMessage();
//         register envelope* env = UsrToEnv(msg);

//         ComlibPrintf("[%d] Calling KDirect %d %d %d\n", CkMyPe(),
//                      env->getTotalsize(), ndestpes, cmsg->dest_proc);
        	
//         if(cmsg->dest_proc == IS_MULTICAST) {      
//             CmiSetHandler(env, handlerId);
            
//             int *cur_pelist = NULL;
//             int cur_npes = 0;
            
//             if(cmsg->sec_id == NULL) {
//                 cur_pelist = kdestpelist;
//                 cur_npes = kfactor;
//             }
//             else {                
//                 cur_npes = (kfactor <= cmsg->sid.npes)?kfactor : 
//                     cmsg->sid.npes;
//                 cur_pelist = cmsg->sid.pe_list;
//             }
            
//             ComlibPrintf("[%d] Sending Message to %d\n", CkMyPe(), cur_npes);
//             CmiSyncListSendAndFree(cur_npes, cur_pelist, 
//                                    UsrToEnv(msg)->getTotalsize(), 
//                                    UsrToEnv(msg));
//         }
//         else {
//             CmiSyncSendAndFree(cmsg->dest_proc, 
//                                UsrToEnv(msg)->getTotalsize(), 
//                                (char *)UsrToEnv(msg));
//         }        
        
//         delete cmsg; 
//     }
// }

// void KDirectMulticastStrategy::pup(PUP::er &p){
//     DirectMulticastStrategy::pup(p);

//     p | kfactor;
// }

// void KDirectMulticastStrategy::beginProcessing(int  nelements){

//     DirectMulticastStrategy::beginProcessing(nelements);

//     kndestpelist = new int[kfactor]; 

//     int next_pe = 0, count = 0;
//     //Assuming the destination pe list is sorted.
//     for(count = 0; count < ndestpes; count++)        
//         if(destpelist[count] > CkMyPe()) {
//             next_pe = count;
//             break;
//         }

//     int kpos = 0;
//     for(count = next_pe; count < next_pe + kfactor; count++){
//         int pe = destpelist[count % ndestpes];
//         kdestpelist[kpos ++] = pe;
//     }
// }

// void KDirectMulticastStrategy::handleMulticastMessage(void *msg){
//     register envelope *env = (envelope *)msg;
    
//     CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);
//     int src_pe = cbmsg->_cookie.pe;
//     if(isDestinationGroup){               
//         CmiSetHandler(env, _charmHandlerIdx);
//         CmiSyncSend(CkMyPe(), env->getTotalsize(), (char *)env);
        
//         int nmsgs = getNumMessagesToSend(src_pe, CkMyPe, CkNumPes());
//         if(nmsgs > 0){            
//             CmiSetHandler(env, handlerId);            
//             CmiSyncListSendAndFree(nmsgs, kdestpelist, 
//                                    env->getTotalsize(), env);
//         }        
//         return;
//     }

//     int status = cbmsg->_cookie.sInfo.cInfo.status;
//     ComlibPrintf("[%d] In handle multicast message %d\n", CkMyPe(), status);

//     if(status == COMLIB_MULTICAST_ALL) {                        
//         int nmsgs = getNumMessagesToSend(src_pe. CkMyPe(), CkNumPes());
//         if(nmsgs > 0){ //Have to forward the messages           
//             void *msg = EnvToUsr(env);
//             void *newmsg = CkCopyMsg(&msg);
//             envelope *newenv = UsrToEnv(newmsg);        
//             CmiSyncListSendAndFree(nmsgs, kdestpelist, 
//                                    newenv->getTotalsize(), newenv);
//         }

//         //Multicast to all destination elements on current processor        
//         ComlibPrintf("[%d] Local multicast sending all %d\n", CkMyPe(), 
//                      localDestIndices.size());
        
//         localMulticast(&localDestIndices, env);
//     }   
//     else if(status == COMLIB_MULTICAST_NEW_SECTION){        
//         CkUnpackMessage(&env);
//         ComlibPrintf("[%d] Received message for new section src=%d\n", 
//                      CkMyPe(), cbmsg->_cookie.pe);

//         ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)cbmsg;
        
//         KDirectHashObject *kobj = 
//             createHashObject(ccmsg->nIndices, ccmsg->indices);
        
//         envelope *usrenv = (envelope *) ccmsg->usrMsg;
        
//         envelope *newenv = (envelope *)CmiAlloc(usrenv->getTotalsize());
//         memcpy(newenv, usrenv, usrenv->getTotalsize());

//         localMulticast(&kobj->indices, newenv);

//         ComlibSectionHashKey key(cbmsg->_cookie.pe, 
//                                  cbmsg->_cookie.sInfo.cInfo.id);

//         KDirectHashObject *old_kobj = 
//             (KDirectHashObject*)sec_ht.get(key);
//         if(old_kobj != NULL)
//             delete old_kobj;
        
//         sec_ht.put(key) = kobj;

//         if(kobj->npes > 0) {
//             ComlibPrintf("[%d] Forwarding Message of %d to %d pes\n", 
//                          CkMyPe(), cbmsg->_cookie.pe, kobj->npes);
//             CkPackMessage(&env);
//             CmiSyncListSendAndFree(kpbj->npes, kobj->pelist, 
//                                    env->getTotalsize(), env);
//         }
//         else
//             CmiFree(env);       
//     }
//     else {
//         //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
//         ComlibSectionHashKey key(cbmsg->_cookie.pe, 
//                                  cbmsg->_cookie.sInfo.cInfo.id);    
//         KDirectHashObject *kobj = (KDirectHashObject *)sec_ht.get(key);
        
//         if(kobj == NULL)
//             CkAbort("Destination indices is NULL\n");
        
//         if(kobj->npes > 0){
//             void *msg = EnvToUsr(env);
//             void *newmsg = CkCopyMsg(&msg);
//             envelope *newenv = UsrToEnv(newmsg);        
//             CmiSyncListSendAndFree(kpbj->npes, kobj->pelist, 
//                                    newenv->getTotalsize(), newenv);

//         }
        
//         localMulticast(&kobj->indices, env);
//     }
// }

// void KDirectMulticastStrategy::initSectionID(CkSectionID *sid){

//     ComlibPrintf("KDirect Init section ID\n");
//     sid->pelist = NULL;
//     sid->npes = 0;

//     int *pelist = new int[kfactor];
//     int npes;
//     getPeList(sid->_nElem,  sid->_elems, pelist, npes);
    
//     sid->destpelist = pelist;
//     sid->ndestpes = npes;    
// }

// KDirectHashObject *KDirectMulticastStrategy::createHashObject(int nelements, CkArrayIndex *elements){

//     KDirectHashObject *kobj = new KDirectHashObject;
//     kobj->pelist = new int[kfactor];
//     getPeList(nelements,  elements, kobj->pelist, kobj->npes);

//     return kobj;
// }


// void KDirectMulticastStrategy::getPeList(int nelements, 
//                                          CkArrayIndex *elements, 
//                                          int *pelist, int &npes, 
//                                          int src_pe){
    
//     npes = 0;
    
//     int *tmp_pelist = new int[CkNumPes()];
//     int num_pes;
    
//     //make this a reusable function call later.
//     int count = 0, acount = 0;
//     for(acount = 0; acount < nelements; acount++){
//         int p = CkArrayID::CkLocalBranch(destArrayID)->
//             lastKnown(elements[acount]);
        
//         for(count = 0; count < num_pes; count ++)
//             if(tmp_pelist[count] == p)
//                 break;
        
//         if(count == num_pes) {
//             tmp_pelist[num_pes ++] = p;
//         }
//     }

//     if(num_pes == 0) {
//         delete [] tmp_pelist;
//         return;
//     }

//     qsort(tmp_pelist, num_pes, sizeof(int), intCompare);
    
//     int pdiff = 0;
//     int my_pos = 0;
//     int src_pos = 0;

//     int count;
//     for(count = 0; count < num_pes; count ++) {
//         if(tmp_pelist[count] == CkMyPe()){
//             my_pos = count;
//         }

//         if(tmp_pelist[count] == src_pos){
//             src_pos = count;
//         }        
//     }            

//     int n_tosend = getNumMessagesToSend(src_pos, my_pos, num_pes);
//     for(count = 0; count < n_tosend; count ++) {
//         pelist[npes ++] = tmp_pelist[(src_pos + count)%num_pes];
//     }    

//     delete [] tmp_pelist;    
// }

// int KDirectMulticastStrategy::getNumMessagesToSend(int src_pe, int my_pe, 
//                                                    int num_pes){
    
//     if(src_pe == my_pe) {
//         retutn 0;
//     }

//     int nToSend = 0;

//     int pdiff = my_pe - src_pe;
    
//     if(pdiff < 0)
//         pdiff += num_pes;
    
//     if(pdiff % kfactor != 0)
//         return 0;
    
//     return (num_pes - pdiff > kfactor)? kfactor : num_pes - pdiff;
// }

// #endif
