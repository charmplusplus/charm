#include "RingMulticastStrategy.h"

//Group Constructor
RingMulticastStrategy::RingMulticastStrategy(int ndest, int *pelist) 
    : DirectMulticastStrategy(ndest, pelist) {
    commonRingInit();
}

//Array Constructor
RingMulticastStrategy::RingMulticastStrategy(CkArrayID dest_aid)
    : DirectMulticastStrategy(dest_aid){
    commonRingInit();    
}

void RingMulticastStrategy::commonRingInit(){
    //Sort destpelist
}

extern int _charmHandlerIdx;
void RingMulticastStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
    if(messageBuf->length() == 0) {
        return;
    }

    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char *msg = cmsg->getCharmMessage();
        register envelope* env = UsrToEnv(msg);

        ComlibPrintf("[%d] Calling Ring %d %d %d\n", CkMyPe(),
                     env->getTotalsize(), ndestpes, cmsg->dest_proc);
        	
        if(cmsg->dest_proc == IS_MULTICAST) {      
            CmiSetHandler(env, handlerId);
            
            int dest_pe = -1;
            RingMulticastHashObject *robj;
            if(cmsg->sec_id == NULL)
                dest_pe = nextPE;
            else {
                robj = getHashObject(CkMyPe(), 
                                     cmsg->sec_id->_cookie.sInfo.cInfo.id);
                dest_pe = robj->nextPE;
            }
            
            ComlibPrintf("[%d] Sending Message to %d\n", CkMyPe(), dest_pe);

            if(dest_pe != -1)
                CmiSyncSend(dest_pe, env->getTotalsize(), (char *)env); 
            
            if(isDestinationArray) {
                if(robj != NULL)
                    localMulticast(&robj->indices, env);
                else
                    localMulticast(&localDestIndices, env);
            }
            else {
                CmiSetHandler(env, _charmHandlerIdx);
                CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);
            }
        }
        else {
            CmiSyncSendAndFree(cmsg->dest_proc, UsrToEnv(msg)->getTotalsize(), 
                               (char *)UsrToEnv(msg));
        }        
        
	delete cmsg; 
    }
}

void RingMulticastStrategy::pup(PUP::er &p){

    DirectMulticastStrategy::pup(p);
}

void RingMulticastStrategy::beginProcessing(int  nelements){

    DirectMulticastStrategy::beginProcessing(nelements);

    nextPE = -1;
    if(ndestpes == 1)
        return;

    for(int count = 0; count < ndestpes; count++)
        if(destpelist[count] > CkMyPe()) {
            nextPE = destpelist[count];
            break;
        }
    if(nextPE == -1)
        nextPE = destpelist[0];
}

void RingMulticastStrategy::handleMulticastMessage(void *msg){
    register envelope *env = (envelope *)msg;
       
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)EnvToUsr(env);
    int src_pe = cbmsg->_cookie.pe;
    if(isDestinationGroup){               

        if(!isEndOfRing(nextPE, src_pe)) {
            ComlibPrintf("[%d] Forwarding Message to %d\n", CkMyPe(), nextPE);
            CmiSyncSend(nextPE, env->getTotalsize(), (char *)env);        
        }
        CmiSetHandler(env, _charmHandlerIdx);
        CmiSyncSendAndFree(CkMyPe(), env->getTotalsize(), (char *)env);
        
        return;
    }

    int status = cbmsg->_cookie.sInfo.cInfo.status;
    ComlibPrintf("[%d] In handle multicast message %d\n", CkMyPe(), status);

    if(status == COMLIB_MULTICAST_ALL) {                        
        if(!isEndOfRing(nextPE, src_pe)) {
            ComlibPrintf("[%d] Forwarding Message to %d\n", CkMyPe(), nextPE);
            CmiSyncSend(nextPE, env->getTotalsize(), (char *)env); 
        }

        //Multicast to all destination elements on current processor        
        ComlibPrintf("[%d] Local multicast sending all %d\n", CkMyPe(), 
                     localDestIndices.size());

        localMulticast(&localDestIndices, env);
    }   
    else if(status == COMLIB_MULTICAST_NEW_SECTION){        
        CkUnpackMessage(&env);
        ComlibPrintf("[%d] Received message for new section src=%d\n", 
                     CkMyPe(), cbmsg->_cookie.pe);

        ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)cbmsg;
        
        RingMulticastHashObject *robj = 
            createHashObject(ccmsg->nIndices, ccmsg->indices);
        
        envelope *usrenv = (envelope *) ccmsg->usrMsg;
        envelope *newenv = (envelope *)CmiAlloc(usrenv->getTotalsize());
        memcpy(newenv, ccmsg->usrMsg, usrenv->getTotalsize());

        localMulticast(&robj->indices, newenv);

        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);

        RingMulticastHashObject *old_robj = 
            (RingMulticastHashObject*)sec_ht.get(key);
        if(old_robj != NULL)
            delete old_robj;
        
        sec_ht.put(key) = robj;

        if(!isEndOfRing(robj->nextPE, src_pe)) {
            ComlibPrintf("[%d] Forwarding Message of %d to %d\n", CkMyPe(), 
                         cbmsg->_cookie.pe, robj->nextPE);
            CkPackMessage(&env);
            CmiSyncSendAndFree(robj->nextPE, env->getTotalsize(), 
                               (char *)env);
        }
        else
            CmiFree(env);       
    }
    else {
        //status == COMLIB_MULTICAST_OLD_SECTION, use the cached section id
        ComlibSectionHashKey key(cbmsg->_cookie.pe, 
                                 cbmsg->_cookie.sInfo.cInfo.id);    
        RingMulticastHashObject *robj = (RingMulticastHashObject *)sec_ht.
            get(key);
        
        if(robj == NULL)
            CkAbort("Destination indices is NULL\n");
        
        if(!isEndOfRing(robj->nextPE, src_pe)) {
            CmiSyncSend(robj->nextPE, env->getTotalsize(), (char *)env);
            ComlibPrintf("[%d] Forwarding Message to %d\n", CkMyPe(), robj->nextPE);
        }

        localMulticast(&robj->indices, env);
    }
}

void RingMulticastStrategy::initSectionID(CkSectionID *sid){
    
    sid->pelist = NULL;
    sid->npes = 0;

    RingMulticastHashObject *robj = 
        createHashObject(sid->_nElems, sid->_elems);
    
    ComlibSectionHashKey key(CkMyPe(), sid->_cookie.sInfo.cInfo.id);
    sec_ht.put(key) = robj;
}

RingMulticastHashObject *RingMulticastStrategy::createHashObject
(int nelements, CkArrayIndexMax *elements){

    RingMulticastHashObject *robj = new RingMulticastHashObject;

    int next_pe = CkNumPes();
    int acount = 0;
    int min_dest = CkNumPes();
    for(acount = 0; acount < nelements; acount++){
        //elements[acount].print();
        int p = CkArrayID::CkLocalBranch(destArrayID)->
            lastKnown(elements[acount]);
        
        if(p < min_dest)
            min_dest = p;
        
        if(p > CkMyPe() && next_pe > p) 
            next_pe = p;       

        if (p == CkMyPe())
            robj->indices.insertAtEnd(elements[acount]);
    }
    
    //Recycle the destination pelist and start from the begining
    if(next_pe == CkNumPes() && min_dest != CkMyPe())        
        next_pe = min_dest;
    
    if(next_pe == CkNumPes())
        next_pe = -1;

    robj->nextPE = next_pe;

    return robj;
}


RingMulticastHashObject *RingMulticastStrategy::getHashObject(int pe, int id){
    
    ComlibSectionHashKey key(pe, id);
    RingMulticastHashObject *robj = (RingMulticastHashObject *)sec_ht.get(key);
    return robj;
}

int RingMulticastStrategy::isEndOfRing(int next_pe, int src_pe){

    if(next_pe < 0)
        return 1;

    ComlibPrintf("[%d] isEndofring %d, %d\n", CkMyPe(), next_pe, src_pe);

    if(next_pe > CkMyPe()){
        if(src_pe <= next_pe && src_pe > CkMyPe())
            return 1;

        return 0;
    }
    
    //next_pe < CkMyPe()

    if(src_pe > CkMyPe() || src_pe <= next_pe)
        return 1;
    
    return 0;
}
