/**
   @addtogroup ComlibCharmStrategy
   @{
   @file 
*/

#ifndef RING_MULTICAST_STRATEGY
#define RING_MULTICAST_STRATEGY

#include "MulticastStrategy.h"

/**
 * Multicast Strategy that sends a multicast in a ring: the source processor
 * send a message only to its following neighbour, which propagates it forward
 * to its neighbour, and so on. Only the processors involved in the multicast
 * are part of the ring.
 */
class RingMulticastStrategy: public MulticastStrategy {
 protected:
    
    int isEndOfRing(int next_pe, int src_pe);
    
    //Defining the two entries of the section multicast interface
    virtual void createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist);

    virtual void createObjectOnIntermediatePe(ComlibSectionHashObject *obj,  int npes, ComlibMulticastIndexCount *counts, int src_pe);
    
 public:
    //Array constructor
    RingMulticastStrategy(): MulticastStrategy() {}
    RingMulticastStrategy(CkMigrateMessage *m) : MulticastStrategy(m){}

    PUPable_decl(RingMulticastStrategy);
};

#endif

/*@}*/
