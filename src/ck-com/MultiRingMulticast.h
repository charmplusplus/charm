/**
   @addtogroup ComlibCharmStrategy
   @{
   @file 
   @brief Section multicast strategy that sends data along two rings

*/

#ifndef MULTIRING_MULTICAST_STRATEGY
#define MULTIRING_MULTICAST_STRATEGY

#include "MulticastStrategy.h"

/**
 Multicast strategy that sends the data using two rings. It divides the total
 number of processors involved in the multicast in two (after ordering them).
 Then two rings are created, one starting with the source processor and its
 half, the other starting at (CkMyPe()+CkNumPes()/2)%CkNumPes().
*/
class MultiRingMulticastStrategy: public MulticastStrategy {
    
 protected:
    
    ///Defines the two entries of the section multicast interface
  virtual void createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist);

    ///Creates the propagation across the half ring
    virtual void createObjectOnIntermediatePe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *counts, int srcpe);

    ///Specifies that the multicast messages need the processor list to be ordered
    virtual int needSorting() { return 1; }
    
 public:
    //Array constructor
    MultiRingMulticastStrategy(): MulticastStrategy() {}
    MultiRingMulticastStrategy(CkMigrateMessage *m): MulticastStrategy(m) {}

   
    PUPable_decl(MultiRingMulticastStrategy);
};

#endif

/*@}*/
