#include "MsgPacker.h"
#include "register.h"
#include "ComlibManager.h"

CpvExtern(int, RecvCombinedShortMsgHdlrIdx);

void short_envelope::pup(PUP::er &p){
    p | idx;
    p | epIdx;
    p | size;
    
    if(p.isUnpacking()) 
        data = new char[size];    
    p(data, size);
}

MsgPacker::MsgPacker(){
    nShortMsgs = 0;
    msgList = 0;    
}

MsgPacker::MsgPacker(CkQ<CharmMessageHolder *> &msgq, int n_msgs){
    nShortMsgs = n_msgs;
    msgList = new short_envelope[n_msgs];    

    for(int count = 0; count < n_msgs; count ++){
        CharmMessageHolder *cmsg = msgq.deq();
        envelope *env = (envelope *)UsrToEnv(cmsg->getCharmMessage());
        CkPackMessage(&env);

        if(count == 0) {
            aid = env->array_mgr();
            if(aid.isZero()) CkAbort("Array packing set and ArrayID is zero");
        }        
        
        msgList[count].epIdx = env->array_ep();
        msgList[count].size = env->getTotalsize() - sizeof(envelope);
        msgList[count].idx = env->array_index();
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

    CmiSetHandler(cmb_msg, CpvAccess(RecvCombinedShortMsgHdlrIdx));
}


void MsgPacker::deliver(CombinedMessage *cmb_msg){
    int nmsgs = cmb_msg->nmsgs;

    char *data = (char *)cmb_msg + ALIGN8(sizeof(CombinedMessage));
    PUP::fromMem fp(data);
    CkArrayID aid = cmb_msg->aid;

    for(int count = 0; count < nmsgs; count ++){
        short_envelope senv;
        fp | senv;
        
        int ep = senv.epIdx;
        CkArrayIndexMax idx = senv.idx;
        int size = senv.size;
        ArrayElement *a_elem = CProxyElement_ArrayBase(aid, idx).ckLocal();

        int msgIdx = _entryTable[ep]->msgIdx;
        //Unpack the message
        senv.data = (char *)_msgTable[msgIdx]->unpack(senv.data); 
        
        if(_entryTable[ep]->noKeep) {
            CkDeliverMessageReadonly(ep, senv.data, a_elem);
            delete [] senv.data;
        }
        else {
            envelope *env = (envelope *)CmiAlloc(sizeof(envelope) + size);
            env->array_mgr() = aid;
            env->array_index() = idx;
            env->array_ep() = ep;
            env->setPacked(0); //We have already unpakced it.
            void *data = EnvToUsr(env);

            memcpy(data, senv.data, size);
            delete[] senv.data;
            
            CkDeliverMessageFree(ep, data, a_elem);           
        }        
    }      
        
    CmiFree(cmb_msg);
}





