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
    RingMulticastStrategy(CkArrayID src, CkArrayID dest);    
    RingMulticastStrategy(CkMigrateMessage *m) {}

    //Destructor
    ~RingMulticastStrategy() { 
        
        CkHashtableIterator *ht_iterator = sec_ht.iterator();
        ht_iterator->seekStart();
        while(ht_iterator->hasNext()){
            void **data;
            data = (void **)ht_iterator->next();        
            RingMulticastHashObject *robj = 
                (RingMulticastHashObject*)(* data);

            *data = NULL;
            if(robj)
                delete robj;
        }

        sec_ht.empty();
    }
    
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    void handleMulticastMessage(void *msg);
    
    void pup(PUP::er &p);    
    void beginProcessing(int nelements);
    
    PUPable_decl(RingMulticastStrategy);
};
#endif

