#include "MsgPacker.h"
#include "register.h"
#include "ComlibManager.h"

CkpvExtern(int, RecvCombinedShortMsgHdlrIdx);

void short_envelope::pup(PUP::er &p){
    p | idx;
    p | epIdx;
    p | size;
    
    if(p.isUnpacking()) 
        data = new char[size];    
    p(data, size);
}

short_envelope::short_envelope(){
    epIdx = 0;
    data = NULL;
}

short_envelope::~short_envelope(){
    /*
      if(data) 
      delete [] data;        
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
            if(aid.isZero()) CkAbort("Array packing set and ArrayID is zero");
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
    PUP::sizer sp;
    for(count = 0; count < nShortMsgs; count ++)
        sp | msgList[count];

    int size = sp.size();  
    total_size = ALIGN8(sizeof(CombinedMessage)) + size;
    
    ComlibPrintf("In MsgPacker with %d bytes and %d messages\n", total_size, 
                 nShortMsgs);

    cmb_msg = (CombinedMessage *)CmiAlloc(total_size);

    PUP::toMem mp((char *)cmb_msg + ALIGN8(sizeof(CombinedMessage)));
    for(count = 0; count < nShortMsgs; count ++)
        mp | msgList[count];

    cmb_msg->aid = aid;
    cmb_msg->nmsgs = nShortMsgs;

    CmiSetHandler(cmb_msg, CkpvAccess(RecvCombinedShortMsgHdlrIdx));
}


void MsgPacker::deliver(CombinedMessage *cmb_msg){
    int nmsgs = cmb_msg->nmsgs;

    ComlibPrintf("In MsgPacker::deliver\n");

    char *from_addr = (char *)cmb_msg + ALIGN8(sizeof(CombinedMessage));
    PUP::fromMem fp(from_addr);
    CkArrayID aid = cmb_msg->aid;

    for(int count = 0; count < nmsgs; count ++){
        short_envelope senv;
        fp | senv;
        
        int ep = senv.epIdx;
        CkArrayIndexMax idx = senv.idx;
        int size = senv.size;

        CProxyElement_ArrayBase ap(aid, idx);
        ArrayElement *a_elem = ap.ckLocal();

        int msgIdx = _entryTable[ep]->msgIdx;
        if(_entryTable[ep]->noKeep && a_elem != NULL) {
            //Unpack the message
            senv.data = (char *)_msgTable[msgIdx]->unpack(senv.data); 

            CkDeliverMessageReadonly(ep, senv.data, a_elem);
            delete[] senv.data;
        }
        else {
            //envelope *env = (envelope *)CmiAlloc(sizeof(envelope) + size);
            envelope *env = _allocEnv(ForArrayEltMsg, sizeof(envelope) + size);

            void *data = EnvToUsr(env);
            memcpy(data, senv.data, size);
            
            //Unpack the message
            data = (char *)_msgTable[msgIdx]->unpack(data); 

            env->getsetArrayMgr() = aid;
            env->getsetArrayIndex() = idx;
            env->getsetArrayEp() = ep;
            env->setPacked(0); 
            env->getsetArraySrcPe()=CkMyPe();  //FOO Bar change later
            env->getsetArrayHops()=0;  //FOO BAR change later
            env->setQueueing(CK_QUEUEING_FIFO);            
            env->setUsed(0);
            env->setMsgIdx(msgIdx);

            if(a_elem)
                CkDeliverMessageFree(ep, data, a_elem);                     
            else
                ap.ckSend((CkArrayMessage *)data, ep);
            
            delete[] senv.data;
        }        
    }      
        
    CmiFree(cmb_msg);
}





