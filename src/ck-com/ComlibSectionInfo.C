
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

    initSectionID(cmsg->sec_id);   

    CkPackMessage(&env);
    int sizes[2];
    sizes[0] = cmsg->sec_id->_nElems;
    sizes[1] = env->getTotalsize();                
    
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
}

void ComlibSectionInfo::unpack(envelope *cb_env, 
                               CkVec<CkArrayIndexMax> &dest_indices, 
                               envelope *&env) {
        
    ComlibMulticastMsg *ccmsg = (ComlibMulticastMsg *)EnvToUsr(cb_env);
    
    for(int count = 0; count < ccmsg->nIndices; count++){
        CkArrayIndexMax idx = ccmsg->indices[count];
        
        //This will work because. lastknown always knows if I have the
        //element of not
        int dest_proc = ComlibGetLastKnown(destArrayID, idx);
        //CkArrayID::CkLocalBranch(destArrayID)->lastKnown(idx);
        
        //        if(dest_proc == CkMyPe())
        dest_indices.insertAtEnd(idx);                        
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

void ComlibSectionInfo::getPeList(int _nElems, 
                                  CkArrayIndexMax *_elems, 
                                  int &npes, int *&pelist){
    
    int length = CkNumPes();
    if(length > _nElems)    //There will not be more processors than
                            //number of elements. This is wastage of
                            //memory as there may be fewer
                            //processors. Fix later.
        length = _nElems;
    
    pelist = new int[length];
    npes = 0;
    
    int count = 0, acount = 0;
    
    for(acount = 0; acount < _nElems; acount++){
        
        int p = ComlibGetLastKnown(destArrayID, _elems[acount]);
        
        if(p == -1) CkAbort("Invalid Section\n");        
        for(count = 0; count < npes; count ++)
            if(pelist[count] == p)
                break;
        
        if(count == npes) {
            pelist[npes ++] = p;
        }
    }   

    if(npes == 0) {
        delete [] pelist;
        pelist = NULL;
    }
}


void ComlibSectionInfo::getRemotePelist(int nindices, 
                                        CkArrayIndexMax *idxlist, 
                                        int &npes, int *&pelist) {

    int count = 0, acount = 0;
    
    int length = CkNumPes();
    if(length > nindices)
        length = nindices;
    
    pelist = new int[length];
    npes = 0;

    for(acount = 0; acount < nindices; acount++){
        
        int p = ComlibGetLastKnown(destArrayID, idxlist[acount]);
        if(p == CkMyPe())
            continue;
        
        if(p == -1) CkAbort("Invalid Section\n");        
        
        //Collect remote processors
        for(count = 0; count < npes; count ++)
            if(pelist[count] == p)
                break;
        
        if(count == npes) {
            pelist[npes ++] = p;
        }
    }
    
    if(npes == 0) {
        delete [] pelist;
        pelist = NULL;
    }
}


void ComlibSectionInfo::getLocalIndices(int nindices, 
                                        CkArrayIndexMax *idxlist, 
                                        CkVec<CkArrayIndexMax> &idx_vec){    
    int count = 0, acount = 0;
    idx_vec.resize(0);
    
    for(acount = 0; acount < nindices; acount++){
        int p = ComlibGetLastKnown(destArrayID, idxlist[acount]);
        if(p == CkMyPe()) 
            idx_vec.insertAtEnd(idxlist[acount]);
    }
}


void ComlibSectionInfo::localMulticast(envelope *env){
    ComlibArrayInfo::localMulticast(&localDestIndexVec, env);
}
