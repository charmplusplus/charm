#ifndef EACH_TO_MANY_MULTICAST
#define EACH_TO_MANY_MULTICAST
#include "ComlibManager.h"

class EachToManyMulticastStrategy: public Strategy {
    CkQ <CharmMessageHolder*> *messageBuf;
    int routerID;
    comID comid;
    
    int npes;
    int *pelist;
    
    void checkPeList();

 public:
    EachToManyMulticastStrategy(int strategyId);
    EachToManyMulticastStrategy(int strategyId, int npes, int *pelist);

    EachToManyMulticastStrategy(CkMigrateMessage *m){}

    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    PUPable_decl(EachToManyMulticastStrategy);
};
#endif

