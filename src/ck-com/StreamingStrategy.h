#ifndef STREAMING_STRATEGY
#define STREAMING_STRATEGY
#include "ComlibManager.h"

#define MAX_STREAMING_MESSAGE_SIZE 2048*2

class StreamingStrategy : public CharmStrategy {
 protected:
    CkQ<CharmMessageHolder *> *streamingMsgBuf;
    int *streamingMsgCount;
    int PERIOD, bufferMax;
    CmiBool shortMsgPackingFlag, idleFlush;

    int streaming_handler_id; //Getting rid of multiple send

    /// Flush all messages destined for this processor:
    void flushPE(int destPE);
    
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
    StreamingStrategy(CkMigrateMessage *m) : CharmStrategy(m) {}
    
    virtual void insertMessage(CharmMessageHolder *msg);
    virtual void doneInserting();
    
    virtual void beginProcessing(int ignored);

    virtual void pup(PUP::er &p);
    virtual void enableShortArrayMessagePacking()
        {shortMsgPackingFlag=CmiTrue;} //Should be used only for array
                                       //messages

    virtual void disableIdleFlush() { idleFlush = CmiFalse;}

    /// Register self to be flushed again after a delay.
    void registerFlush(void);
    /// Flush all pending messages:
    void periodicFlush();

    PUPable_decl(StreamingStrategy);
};
#endif
