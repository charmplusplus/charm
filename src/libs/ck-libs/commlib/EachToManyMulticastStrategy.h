#ifndef EACH_TO_MANY_MULTICAST_STRATEGY
#define EACH_TO_MANY_MULTICAST_STRATEGY

#include "ComlibManager.h"

class EachToManyMulticastStrategy: public Strategy {
    CkQ <CharmMessageHolder*> *messageBuf;
    int routerID;
    comID comid;
    
    int MyPe;
    int npes;
    int *pelist;
    
    long handler; //Multicast Handler to be called on the receiving processors.
    int handlerId;
    void checkPeList();

 public:
    EachToManyMulticastStrategy(int strategyId, ComlibMulticastHandler h);
    EachToManyMulticastStrategy(int strategyId, int npes, int *pelist, 
				ComlibMulticastHandler h);

    EachToManyMulticastStrategy(CkMigrateMessage *m){}

    ComlibMulticastHandler getHandler(){return 
                                            (ComlibMulticastHandler)handler;};
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    
    virtual void beginProcessing(int nelements);

    PUPable_decl(EachToManyMulticastStrategy);
};
#endif

