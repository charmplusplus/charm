#ifndef STREAMING_STRATEGY
#define STREAMING_STRATEGY
#include "ComlibManager.h"

class StreamingStrategy : public Strategy {
    CkQ<CharmMessageHolder *> *streamingMsgBuf;
    int *streamingMsgCount;
    int PERIOD, bufferMax;
    CmiBool shortMsgPackingFlag;
    
 public:
    /**
     Create a streaming strategy, suitable for passing to ComlibManager.
     These are the criteria for flushing all pending messages:
       - it's been at least period (in ms) since the last flush, or
       - the processor just went idle.
     Thses criteria flush a single PE's pending messages:
       - more than bufferMax messages to buffered for one PE.
    */
    StreamingStrategy(int periodMs=10,int bufferMax=1000);
    StreamingStrategy(CkMigrateMessage *){}
    
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    /// Flush all pending messages:
    void periodicFlush();
    /// Flush all messages destined for this processor:
    void flushPE(int destPE);

    /// Register self to be flushed again after a delay.
    void registerFlush(void);
    
    virtual void beginProcessing(int ignored);

    virtual void pup(PUP::er &p);
    void enableShortArrayMessagePacking(){shortMsgPackingFlag=CmiTrue;}
    //Should be used only for array messages

    PUPable_decl(StreamingStrategy);
};
#endif
