#ifndef RING_MULTICAST_STRATEGY
#define RING_MULTICAST_STRATEGY

#include "DirectMulticastStrategy.h"


class RingMulticastStrategy: public DirectMulticastStrategy {
    
 protected:
    
    int isEndOfRing(int next_pe, int src_pe);
    
    //Defining the two entries of the section multicast interface
    virtual ComlibSectionHashObject *createObjectOnSrcPe(int nelements, 
                                                         CkArrayIndexMax *elements);

    virtual ComlibSectionHashObject *createObjectOnIntermediatePe
        (int nelements, CkArrayIndexMax *elements, int src_pe);
    
 public:
    
    //Array constructor
    RingMulticastStrategy(CkArrayID dest_id, int flag = 0);    
    RingMulticastStrategy(CkMigrateMessage *m) : DirectMulticastStrategy(m){}

    //Destructor
    ~RingMulticastStrategy() {}
    
    void pup(PUP::er &p);    
    void beginProcessing(int nelements);
    
    PUPable_decl(RingMulticastStrategy);
};

#endif

