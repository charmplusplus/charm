#ifndef RING_MULTICAST_STRATEGY
#define RING_MULTICAST_STRATEGY

#include "DirectMulticastStrategy.h"

class RingMulticastHashObject{
 public:
    CkVec<CkArrayIndexMax> indices;
    int nextPE;
};


class RingMulticastStrategy: public DirectMulticastStrategy {
    
    int nextPE;
    
    void commonRingInit();
    int isEndOfRing(int next_pe, int src_pe);
    RingMulticastHashObject *getHashObject(int pe, int id);
    RingMulticastHashObject *createHashObject(int nelements, 
                                              CkArrayIndexMax *elements);
    void initSectionID(CkSectionID *sid);

 public:
    
    //Group constructor
    RingMulticastStrategy(int ndestpes, int *destpelist);
    
    //Array constructor
    RingMulticastStrategy(CkArrayID dest_id);    
    RingMulticastStrategy(CkMigrateMessage *m) {}
    
    //void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    void handleMulticastMessage(void *msg);
    
    void pup(PUP::er &p);    
    void beginProcessing(int nelements);
    
    PUPable_decl(RingMulticastStrategy);
};
#endif

