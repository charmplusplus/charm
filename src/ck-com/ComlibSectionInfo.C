
#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

ComlibMulticastMsg * ComlibSectionInfo::getNewMulticastMessage
(CharmMessageHolder *cmsg){
    
    if(cmsg->sec_id == NULL || cmsg->sec_id->_nElems == 0)
        return NULL;

    void *m = cmsg->getCharmMessage();
    envelope *env = UsrToEnv(m);
    
    if(cmsg->sec_id->_cookie.sInfo.cInfo.id != 0) 
        CmiAbort("In correct section\n");

    CkPackMessage(&env);
    int sizes[2];
    sizes[0] = cmsg->sec_id->_nElems;
    sizes[1] = env->getTotalsize();                
    
    cmsg->sec_id->_cookie.sInfo.cInfo.id = MaxSectionID ++;
    
    ComlibPrintf("Creating new comlib multicast message %d, %d\n", sizes[0], sizes[1]);
    
    ComlibMulticastMsg *msg = new(sizes, 0) ComlibMulticastMsg;
    msg->nIndices = cmsg->sec_id->_nElems;
    msg->_cookie.sInfo.cInfo.instId = instanceID;
    msg->_cookie.sInfo.cInfo.id = MaxSectionID - 1;
    msg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_NEW_SECTION;
    msg->_cookie.type = COMLIB_MULTICAST_MESSAGE;
    msg->_cookie.pe = CkMyPe();
    
    memcpy(msg->indices, cmsg->sec_id->_elems, 
           sizes[0] * sizeof(CkArrayIndexMax));
    memcpy(msg->usrMsg, env, sizes[1] * sizeof(char));         
    envelope *newenv = UsrToEnv(msg);
    
    newenv->getsetArrayMgr() = env->getsetArrayMgr();
    newenv->getsetArraySrcPe() = env->getsetArraySrcPe();
    newenv->getsetArrayEp() = env->getsetArrayEp();
    newenv->getsetArrayHops() = env->getsetArrayHops();
    newenv->getsetArrayIndex() = env->getsetArrayIndex();

    // for trace projections
    newenv->setEvent(env->getEvent());
    newenv->setSrcPe(env->getSrcPe());
    
    CkPackMessage(&newenv);        
    return (ComlibMulticastMsg *)EnvToUsr(newenv);

    return NULL;
}

void ComlibSectionInfo::unpack(envelope *cb_env, 
                               CkVec<CkArrayIndexMax> *&dest_indices, 
                               envelope *&env) {
        
    dest_indices = NULL;    
    ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)EnvToUsr(cb_env);
    
    if(ccmsg->nIndices > 0)
        dest_indices = new CkVec<CkArrayIndexMax>;

    for(int count = 0; count < ccmsg->nIndices; count++){
        CkArrayIndexMax idx = ccmsg->indices[count];
        
        //This will work because. lastknown always knows if I have the
        //element of not
        int dest_proc = ComlibGetLastKnown(destArrayID, idx);
        //CkArrayID::CkLocalBranch(destArrayID)->lastKnown(idx);
        
        if(dest_proc == CkMyPe())
            dest_indices->insertAtEnd(idx);                        
    }            
    
    envelope *usrenv = (envelope *) ccmsg->usrMsg;
    env = (envelope *)CmiAlloc(usrenv->getTotalsize());
    memcpy(env, ccmsg->usrMsg, usrenv->getTotalsize());
}


void ComlibSectionInfo::processOldSectionMessage(CharmMessageHolder *cmsg) {

    ComlibPrintf("Process Old Section Message \n");

    int cur_sec_id = ComlibSectionInfo::getSectionID(*cmsg->sec_id);

    //Old section id, send the id with the message
    CkMcastBaseMsg *cbmsg = (CkMcastBaseMsg *)cmsg->getCharmMessage();
    cbmsg->_cookie.sInfo.cInfo.id = cur_sec_id;
    cbmsg->_cookie.sInfo.cInfo.status = COMLIB_MULTICAST_OLD_SECTION;
}

void ComlibSectionInfo::initSectionID(CkSectionID *sid){
    
    if(sid->npes > 0) 
        return;

    sid->pelist = new int[CkNumPes()];
    sid->npes = 0;
    
    int count = 0, acount = 0;

    for(acount = 0; acount < sid->_nElems; acount++){

        int p = ComlibGetLastKnown(destArrayID, sid->_elems[acount]);
        //CkArrayID::CkLocalBranch(destArrayID)->
        //lastKnown(sid->_elems[acount]);
        
        if(p == -1) CkAbort("Invalid Section\n");        
        for(count = 0; count < sid->npes; count ++)
            if(sid->pelist[count] == p)
                break;
        
        if(count == sid->npes) {
            sid->pelist[sid->npes ++] = p;
        }
    }   
}

void ComlibSectionInfo::localMulticast(envelope *env){
    ComlibArrayInfo::localMulticast(&localDestIndexVec, env);
}

