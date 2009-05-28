/**
   @addtogroup ComlibCharmStrategy
   *@{
   @file 
*/

#ifndef PRIO_STREAMING
#define PRIO_STREAMING

#include "ComlibManager.h"
#include "StreamingStrategy.h"

/**
 * Class that streams messages the same way as StreamingStrategy, but adding a
 * bypass for high priority messages to be flushed immediately.

 These are the criteria for flushing all pending messages: 
 <ul>
 <li> it's been at least period (in ms) since the last flush, or 
 <li> the processor just went idle.
 </ul>

 These criteria flush a single E's pending messages: 
 <ul>
 <li> more than bufferMax messages to buffered for one PE.
 <li>Current message is a high priority message
 </ul>
 */
class PrioStreaming : public StreamingStrategy, public CharmStrategy {
 protected:
    int basePriority;
    CkVec<int> minPrioVec;
    
 public:
    /**
     Create a priority based streaming strategy, suitable for passing
     to ComlibManager.  
     These are the criteria for flushing all pending messages: 

     - it's been at least period (in ms) since the last flush, or 

     - the processor just went idle.  Thses criteria flush a single 
     PE's pending messages: 

     - more than bufferMax messages to buffered for one PE.

     - Current message is a high priority message
    */

    PrioStreaming(int periodMs=DEFAULT_TIMEOUT, 
		  int bufferMax=MAX_NUM_STREAMING_MESSAGES, 
		  int prio=0,
		  int msgSizeMax=MAX_STREAMING_MESSAGE_SIZE,
		  int bufSizeMAX=MAX_STREAMING_MESSAGE_SIZE*MAX_NUM_STREAMING_MESSAGES);
    PrioStreaming(CkMigrateMessage *m) : StreamingStrategy(m), CharmStrategy(m) {}

    void insertMessage(MessageHolder *msg) {insertMessage((CharmMessageHolder*)msg);}
    virtual void insertMessage(CharmMessageHolder *msg);

    //If new priority is greater than current priority, 
    //then flush all queues which have relatively high priority messages
    inline void setBasePriority(int p) {
        if(p > basePriority) {
            for(int count =0; count < CkNumPes(); count++)
                if(minPrioVec[count] <= p)
                    flushPE(count);
        }        
        basePriority = p;
    }

    virtual void pup(PUP::er &p);
    PUPable_decl(PrioStreaming);
};
#endif

/*@}*/
