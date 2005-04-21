#ifndef MULTIRING_MULTICAST_STRATEGY
#define MULTIRING_MULTICAST_STRATEGY

#include "DirectMulticastStrategy.h"

/***************************************************************
              Section multicast strategy that sends data along a ring 
              with multiple start points on the ring

      Sameer - 04/19/05
*************************************************************/


class MultiRingMulticast: public DirectMulticastStrategy {
    
 protected:
    
    int isEndOfRing(int next_pe, int src_pe);
    
    //Defining the two entries of the section multicast interface
    virtual ComlibSectionHashObject *createObjectOnSrcPe(int nelements, 
                                                         CkArrayIndexMax *elements);
    
    virtual ComlibSectionHashObject *createObjectOnIntermediatePe
        (int nelements, CkArrayIndexMax *elements, int src_pe);
    
 public:
    
    //Array constructor
    MultiRingMulticast(CkArrayID dest_id, int flag = 0);    
    MultiRingMulticast(CkMigrateMessage *m) : DirectMulticastStrategy(m){}

    //Destructor
    ~MultiRingMulticast() {}
    
    void pup(PUP::er &p);    
    void beginProcessing(int nelements);
    
    PUPable_decl(MultiRingMulticast);
};

#endif

