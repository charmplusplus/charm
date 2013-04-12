// #ifdef filippo

// #include "MsgPacker.h"

// CkpvExtern(int, RecvCombinedShortMsgHdlrIdx);

// MsgPacker::MsgPacker(){
//     nShortMsgs = 0;
//     msgList = 0;    
// }

// MsgPacker::MsgPacker(CkQ<CharmMessageHolder *> &msgq, int n_msgs){

//     CkAssert(n_msgs < 65536);  //16 bit field for num messages

//     nShortMsgs = n_msgs;
//     msgList = new short_envelope[n_msgs];    

//     for(int count = 0; count < n_msgs; count ++){
//         CharmMessageHolder *cmsg = msgq.deq();
//         char *msg = cmsg->getCharmMessage();
//         envelope *env = (envelope *)UsrToEnv(msg);
//         CkPackMessage(&env);

//         if(count == 0) {
//             aid = env->getArrayMgr();
//             if(aid.isZero()) 
//                 CkAbort("Array packing set and ArrayID is zero");
//         }        
        
//         msgList[count].epIdx = env->getsetArrayEp();
//         msgList[count].size = env->getTotalsize() - sizeof(envelope);
//         msgList[count].idx = env->getsetArrayIndex();
//         msgList[count].data = msg;

//         CkAssert(msgList[count].size < MAX_MESSAGE_SIZE);
//         delete cmsg;
//     }
// }

// //Takes a queue of envelopes as char* ptrs and not charm message holders
// //Used by mesh streaming strategy
// MsgPacker::MsgPacker(CkQ<char *> &msgq, int n_msgs){
    
//     CkAssert(n_msgs < 65536);  //16 bit field for num messages
    
//     nShortMsgs = n_msgs;
//     msgList = new short_envelope[n_msgs];    
    
//     for(int count = 0; count < n_msgs; count ++){
//         envelope *env = (envelope *)msgq.deq();
//         char *msg = (char *)EnvToUsr(env);
//         CkPackMessage(&env);

//         if(count == 0) {
//             aid = env->getArrayMgr();
//             if(aid.isZero()) 
//                 CkAbort("Array packing set and ArrayID is zero");
//         }        
        
//         msgList[count].epIdx = env->getsetArrayEp();
//         msgList[count].size = env->getTotalsize() - sizeof(envelope);
//         msgList[count].idx = env->getsetArrayIndex();
//         msgList[count].data = msg;
        
//         CkAssert(msgList[count].size < MAX_MESSAGE_SIZE);
//     }
// }

// MsgPacker::~MsgPacker(){
//     if(nShortMsgs > 0 && msgList != NULL) {
//         for(int count = 0; count < nShortMsgs; count ++)
//             CkFreeMsg(msgList[count].data);        
        
//         delete [] msgList;
//     }
// }

// void MsgPacker::getMessage(CombinedMessage *&cmb_msg, int &total_size){
//     int count;
//     PUP_cmiAllocSizer sp;

//     CombinedMessage cmb_hdr;
//     cmb_hdr.aid = aid;
//     cmb_hdr.srcPE = CkMyPe();
//     cmb_hdr.nmsgs = nShortMsgs;

//     sp | cmb_hdr;
//     for(count = 0; count < nShortMsgs; count ++)
//         sp | msgList[count];
    
//     total_size = sp.size();
//     ComlibPrintf("In MsgPacker with %d bytes and %d messages\n", total_size, 
//                  nShortMsgs);

//     cmb_msg = (CombinedMessage *)CmiAlloc(sp.size());

//     PUP_toCmiAllocMem mp(cmb_msg);
//     mp | cmb_hdr;

//     for(count = 0; count < nShortMsgs; count ++)
//         mp | msgList[count];

//     CmiSetHandler(cmb_msg, CkpvAccess(RecvCombinedShortMsgHdlrIdx));
// }

// #endif
