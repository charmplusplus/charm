#include "ComlibManager.h"
#include "MsgPacker.h"
#include "register.h"
#include "pup_cmialloc.h"

CkpvExtern(int, RecvCombinedShortMsgHdlrIdx);

void short_envelope::pup(PUP::er &p){
    p | idx;
    p | epIdx;
    p | size;
    
    //if(p.isUnpacking()) 
    //  data = new char[size];

    p.pupCmiAllocBuf((void **)&data, size);
}

short_envelope::short_envelope(){
    epIdx = 0;
    data = NULL;
}

short_envelope::~short_envelope(){
    /*
      if(data) 
      CmiFree(data);        
      data = NULL;
    */
}


MsgPacker::MsgPacker(){
    nShortMsgs = 0;
    msgList = 0;    
}

MsgPacker::MsgPacker(CkQ<CharmMessageHolder *> &msgq, int n_msgs){

    CkAssert(n_msgs < 65536);  //16 bit field for num messages

    nShortMsgs = n_msgs;
    msgList = new short_envelope[n_msgs];    

    for(int count = 0; count < n_msgs; count ++){
        CharmMessageHolder *cmsg = msgq.deq();
        envelope *env = (envelope *)UsrToEnv(cmsg->getCharmMessage());
        CkPackMessage(&env);

        if(count == 0) {
            aid = env->getsetArrayMgr();
            if(aid.isZero()) 
                CkAbort("Array packing set and ArrayID is zero");
        }        
        
        msgList[count].epIdx = env->getsetArrayEp();
        msgList[count].size = env->getTotalsize() - sizeof(envelope);
        msgList[count].idx = env->getsetArrayIndex();
        msgList[count].data = cmsg->getCharmMessage();

        if(msgList[count].size > MAX_MESSAGE_SIZE)
            CkAbort("Can't send messges larger than 64KB\n");

        delete cmsg;
    }
}

MsgPacker::~MsgPacker(){
    if(nShortMsgs > 0 && msgList != NULL) {
        for(int count = 0; count < nShortMsgs; count ++)
            CkFreeMsg(msgList[count].data);        
        
        delete [] msgList;
    }
}

void MsgPacker::getMessage(CombinedMessage *&cmb_msg, int &total_size){
    int count;
    PUP_cmiAllocSizer sp;

    CombinedMessage cmb_hdr;
    cmb_hdr.aid = aid;
    cmb_hdr.srcPE = CkMyPe();
    cmb_hdr.nmsgs = nShortMsgs;

    sp | cmb_hdr;
    for(count = 0; count < nShortMsgs; count ++)
        sp | msgList[count];
    
    total_size = sp.size();
    ComlibPrintf("In MsgPacker with %d bytes and %d messages\n", total_size, 
                 nShortMsgs);

    cmb_msg = (CombinedMessage *)CmiAlloc(sp.size());

    PUP_toCmiAllocMem mp(cmb_msg);
    mp | cmb_hdr;

    for(count = 0; count < nShortMsgs; count ++)
        mp | msgList[count];

    CmiSetHandler(cmb_msg, CkpvAccess(RecvCombinedShortMsgHdlrIdx));
}


void MsgPacker::deliver(CombinedMessage *cmb_msg){

    CombinedMessage cmb_hdr;

    PUP_fromCmiAllocMem fp(cmb_msg);
    fp | cmb_hdr;

    int nmsgs = cmb_hdr.nmsgs;

    ComlibPrintf("In MsgPacker::deliver\n");
    CkArrayID aid = cmb_hdr.aid;
    int src_pe = cmb_hdr.srcPE;

    for(int count = 0; count < nmsgs; count ++){
        short_envelope senv;
        fp | senv;
        
        int ep = senv.epIdx;
        CkArrayIndexMax idx = senv.idx;
        int size = senv.size;

        CProxyElement_ArrayBase ap(aid, idx);
        ArrayElement *a_elem = ap.ckLocal();
        CkArray *a=(CkArray *)_localBranch(aid);

        int msgIdx = _entryTable[ep]->msgIdx;
        if(_entryTable[ep]->noKeep && a_elem != NULL) {
            //Unpack the message
            senv.data = (char *)_msgTable[msgIdx]->unpack(senv.data); 
            CkDeliverMessageReadonly(ep, senv.data, a_elem);            
            CmiFree(senv.data);
        }
        else {
            //envelope *env = (envelope *)CmiAlloc(sizeof(envelope) + size);
            envelope *env = _allocEnv(ForArrayEltMsg, 
                                      sizeof(envelope) + size);

            void *data = EnvToUsr(env);
            memcpy(data, senv.data, size);
            
            //Unpack the message
            data = (char *)_msgTable[msgIdx]->unpack(data); 
            
            env->getsetArrayMgr() = aid;
            env->getsetArrayIndex() = idx;
            env->getsetArrayEp() = ep;
            env->setPacked(0); 
            env->getsetArraySrcPe()=src_pe;  
            env->getsetArrayHops()=1;  
            env->setQueueing(CK_QUEUEING_FIFO);            
            env->setUsed(0);
            env->setMsgIdx(msgIdx);

            env->setTotalsize(sizeof(envelope) + size);

            //if(a_elem)
            //  CkDeliverMessageFree(ep, data, a_elem);                     
            //else
            //ap.ckSend((CkArrayMessage *)data, ep);
            
            a->deliver((CkArrayMessage *)data, CkDeliver_queue);

            CmiFree(senv.data);
        }        
    }      
        
    CmiFree(cmb_msg);
}





