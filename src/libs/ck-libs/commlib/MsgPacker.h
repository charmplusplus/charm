#ifndef MESSAGE_PACKER_H
#define MESSAGE_PACKER_H

#include "charm++.h"
#include "envelope.h"

#define MAX_MESSAGE_SIZE 65535

class short_envelope {
 public:
    UShort epIdx;
    UShort size;  //Can only send messages up to 64KB :)    
    CkArrayIndexMax idx;
    char *data;

    short_envelope(){}
    short_envelope(CkMigrateMessage *){}
    
    void pup(PUP::er &p);
};
PUPmarshall(short_envelope);

#if CMK_BLUEGENE_CHARM
#define CmiReservedHeaderSize   CmiBlueGeneMsgHeaderSizeBytes
#else
#define CmiReservedHeaderSize   CmiMsgHeaderSizeBytes
#endif

struct CombinedMessage{

    char header[CmiReservedHeaderSize];
    CkArrayID aid;
    int nmsgs;
};

class MsgPacker {        
    CkArrayID aid;
    short_envelope * msgList;
    int nShortMsgs;   

 public:
    MsgPacker();
    ~MsgPacker();    
    MsgPacker(CkQ<CharmMessageHolder*> &cmsg_list, int n_msgs);
    void getMessage(CombinedMessage *&msg, int &size);

    static void deliver(CombinedMessage *cmb_msg);
};

#endif
