#ifndef MESSAGE_PACKER_H
#define MESSAGE_PACKER_H

#include "charm++.h"
#include "envelope.h"
#include "ComlibManager.h"
#include "register.h"
#include "pup_cmialloc.h"

#define MAX_MESSAGE_SIZE 32768

class short_envelope {
 public:
    UShort epIdx;
    UShort size;  //Can only send messages up to 64KB :)    
    
    CkArrayIndexMax idx;
    char *data;

    short_envelope();
    ~short_envelope();
    inline short_envelope(CkMigrateMessage *){}
    
    void pup(PUP::er &p);
};

inline short_envelope::short_envelope(){
    epIdx = 0;
    data = NULL;
}

inline short_envelope::~short_envelope(){
    /*
      if(data) 
      CmiFree(data);        
      data = NULL;
    */
}

inline void short_envelope::pup(PUP::er &p){    

    p | epIdx;
    p | size;        
    //p | idx;
    
    if(p.isUnpacking())
        idx.nInts = 0;

    p((char *)&(idx.nInts), 1);
    p((int *)(idx.data()), idx.nInts);

    p.pupCmiAllocBuf((void **)&data, size);
}

struct CombinedMessage{

    char header[CmiReservedHeaderSize];
    CkArrayID aid;
    int srcPE;
    int nmsgs;
};

PUPbytes(CombinedMessage);

class MsgPacker {        
    CkArrayID aid;
    short_envelope * msgList;
    int nShortMsgs;   

 public:
    MsgPacker();
    ~MsgPacker();    
    
    //Makes a message out of a queue of CharmMessageHolders
    MsgPacker(CkQ<CharmMessageHolder*> &cmsg_list, int n_msgs);
    
    //Takes a queue of envelopes as char* ptrs and not charm message holders
    //Used by mesh streaming strategy
    MsgPacker::MsgPacker(CkQ<char *> &msgq, int n_msgs);
    
    void getMessage(CombinedMessage *&msg, int &size);
    static void deliver(CombinedMessage *cmb_msg);
};

inline void MsgPacker::deliver(CombinedMessage *cmb_msg){

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


#endif
