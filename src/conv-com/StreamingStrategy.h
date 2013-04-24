/**
   @addtogroup ComlibConverseStrategy
   @{
   @file 
*/

#ifndef STREAMING_STRATEGY
#define STREAMING_STRATEGY
//#include "convcomlibmanager.h"
#include "convcomlibstrategy.h"

#define MAX_STREAMING_MESSAGE_SIZE 2048*2
#define MAX_NUM_STREAMING_MESSAGES 1000
#define DEFAULT_TIMEOUT 10

CpvExtern(int, streaming_handler_id);
extern void StreamingHandlerFn(void *msg);

/**
 * The header prepended to combine messages by StreamingStrategy and derived
 * classes.
 */
struct StreamingMessage {
  char header[CmiReservedHeaderSize];
  CmiUInt4 srcPE;
  CmiUInt4 nmsgs;
};

PUPbytes(StreamingMessage)

/**
 * Strategy that buffers small messages and combines them to send few bigger
 * messages, and therefore gain on sending overhead.

 These are the criteria for flushing all pending messages:
 <ul>
 <li> it's been at least period (in ms) since the last flush, or
 <li> the processor just went idle.
 </ul>

 These criteria flush a single PE's pending messages:
 <ul>
 <li> more than bufferMax messages buffered for one PE, or.
 <li> total size of buffered messages > bufSizeMax
 </ul>
 */
class StreamingStrategy : public Strategy {
 protected:
    CkQ<MessageHolder *> *streamingMsgBuf;
    int *streamingMsgCount;
    int *bufSize;
    int bufferMax;
    int msgSizeMax;
    int bufSizeMax;
    double PERIOD;
    //bool shortMsgPackingFlag;
    bool idleFlush;

    //int streaming_handler_id; //Getting rid of multiple send

    /// Flush all messages destined for this processor:
    void flushPE(int destPE);
    
 public:
    /**
     Create a streaming strategy, suitable for passing to Comlib.
     These are the criteria for flushing all pending messages:
       - it's been at least period (in ms) since the last flush, or
       - the processor just went idle.
     These criteria flush a single PE's pending messages:
       - more than bufferMax messages to buffered for one PE, or
       - max buffer size reached
     Messages above the size threshold are sent directly without using the strategy  .
    */
    StreamingStrategy(int periodMs=DEFAULT_TIMEOUT, 
		      int bufferMax=MAX_NUM_STREAMING_MESSAGES,
		      int msgSizeMax=MAX_STREAMING_MESSAGE_SIZE, 
		      int bufSizeMax=MAX_STREAMING_MESSAGE_SIZE*MAX_NUM_STREAMING_MESSAGES);
    StreamingStrategy(double periodMs=DEFAULT_TIMEOUT, 
		      int bufferMax=MAX_NUM_STREAMING_MESSAGES, 
		      int msgSizeMax=MAX_STREAMING_MESSAGE_SIZE, 
		      int bufSizeMax=MAX_STREAMING_MESSAGE_SIZE*MAX_NUM_STREAMING_MESSAGES);

    StreamingStrategy(CkMigrateMessage *m) : Strategy(m) {}
    
    virtual void insertMessage(MessageHolder *msg);
    virtual void doneInserting();
    
    virtual void handleMessage(void *msg) {
      CmiAbort("[%d] StreamingStrategy::handleMessage should never be called\n");
    }

    //virtual void beginProcessing(int ignored);

    virtual void pup(PUP::er &p);
    //virtual void enableShortArrayMessagePacking()
    //    {shortMsgPackingFlag = true;} //Should be used only for array
                                       //messages

    virtual void disableIdleFlush() { idleFlush = false;}

    /// Register self to be flushed again after a delay.
    void registerFlush(void);
    /// Flush all pending messages:
    void periodicFlush();

    PUPable_decl(StreamingStrategy);
};
#endif

/*@}*/
