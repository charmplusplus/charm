#ifndef PRIO_STREAMING
#define PRIO_STREAMING
#include "ComlibManager.h"
#include "StreamingStrategy.h"


class PrioStreaming : public StreamingStrategy {
 protected:
    int basePriority;
    
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

    PrioStreaming(int periodMs=10, int bufferMax=1000, int prio=0);
    PrioStreaming(CkMigrateMessage *){}
    
    virtual void insertMessage(CharmMessageHolder *msg);

    virtual void pup(PUP::er &p);
    PUPable_decl(PrioStreaming);
};
#endif
