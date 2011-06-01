#ifndef MESSAGE_PACKER_H
#define MESSAGE_PACKER_H

/**
   @addtogroup CharmComlib
   *@{

   @file
   
   @brief An envelope for packing multiple messages into a single message.
*/


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
    
    CkArrayIndex idx;
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
  char nints = 0;

  p | epIdx;
  //  p | size;        
  //p | idx;
  
  //Complex pup of arrays, even want to save 3 bytes, GREEDY, GREEDY :)
  if(!p.isUnpacking()) 
    nints = idx.nInts;

  p | nints;
  idx.nInts = nints;
  p((int *)(idx.data()), idx.nInts);
  
  if(p.isUnpacking()) {
      p.pupCmiAllocBuf((void **)&data);
      size = SIZEFIELD(data);
  }
  else 
      p.pupCmiAllocBuf((void **)&data, size);
}

struct CombinedMessage{
    char header[CmiReservedHeaderSize];
    CkArrayID aid;
    unsigned short srcPE;  //Will not work on a very large bluegene machine!
    unsigned short nmsgs;
};

PUPbytes(CombinedMessage)

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
    MsgPacker(CkQ<char *> &msgq, int n_msgs);
    
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
    CkArray *a=(CkArray *)_localBranch(aid);

    ArrayElement *a_elem=NULL, *prev_elem=NULL;
    CkArrayIndex prev_idx;
    prev_idx.nInts = -1;

    for(int count = 0; count < nmsgs; count ++){
        short_envelope senv;
        fp | senv;
        
        int ep = senv.epIdx;
        int size = senv.size;

        if(senv.idx == prev_idx) {
            a_elem = prev_elem;
        }
        else {
            CProxyElement_ArrayBase ap(aid, senv.idx);
            a_elem = ap.ckLocal();
        }

        int msgIdx = _entryTable[ep]->msgIdx;
        if(_entryTable[ep]->noKeep && a_elem != NULL) {
            //Unpack the message
            senv.data = (char *)_msgTable[msgIdx]->unpack(senv.data); 
            CkDeliverMessageReadonly(ep, senv.data, a_elem);            

            prev_elem = a_elem;
            prev_idx = senv.idx;
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
            env->getsetArrayIndex() = senv.idx;
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

            prev_elem = a_elem;
            prev_idx = senv.idx;
            CmiFree(senv.data);
        }   
    }      
        
    CmiFree(cmb_msg);
}

/*@}*/
#endif

