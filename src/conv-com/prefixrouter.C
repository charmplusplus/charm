/**
   @addtogroup ConvComlibRouter
   @{
   @file 
*/



#include "prefixrouter.h"

#if 0
#define PREFIXDEB printf
#else
#define PREFIXDEB /* printf */
#endif

void PrefixRouter::EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq) {
  PREFIXDEB("[%d]Sending through prefix router: ",CmiMyPe());
  if(!msgq.isEmpty()){
    MessageHolder *mhdl = msgq[0];
    if(mhdl->dest_proc<0)  // broadcast or multicast
        sendMulticast(msgq);
    else
        sendPointToPoint(msgq);
  }
  Done(id);
}

void PrefixRouter::sendMulticast(CkQ<MessageHolder *> &msgq) {
    int count;
    PREFIXDEB("with a multicast\n");
    while(!msgq.isEmpty()){
        MessageHolder *mhdl = msgq.deq();
        
        if(mhdl->dest_proc == IS_BROADCAST) {
            for(count = 0; count < npes; count ++) {
                int curDest = gpes[MyPe ^ count];
                char *msg = mhdl->getMessage();
                CmiSyncSend(curDest, mhdl->size, msg);
            }
        }
        else {
            CmiAbort("Implement later");
        }
    }
}

void PrefixRouter::sendPointToPoint(CkQ<MessageHolder *> &msgq) {
    int count, i;
    PREFIXDEB("with a point-to-point\n");
    int len = msgq.length();
    for(count = 0; count < npes; count ++) {
        int curDest = gpes[MyPe ^ count];
        
        for(i = 0; i < len; i++) {
            MessageHolder *mhdl = msgq[i];
            
            CmiAssert(mhdl->dest_proc >= 0);
            if(mhdl->dest_proc == curDest) {
                char *msg = mhdl->getMessage();
                CmiSyncSendAndFree(curDest, mhdl->size, msg);
            }
        }
    }
    
    for(i = 0; i < len; i++) {
        MessageHolder *mhdl = msgq.deq();
        delete mhdl;
    }
}
/*@}*/
