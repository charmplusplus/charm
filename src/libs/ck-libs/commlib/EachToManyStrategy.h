#ifndef EACH_TO_MANY_STRATEGY
#define EACH_TO_MANY_STRATEGY
#include "ComlibManager.h"

class EachToManyStrategy : public Strategy {
    //CkQ <CharmMessageHolder*> *messageBuf;
    CharmMessageHolder *messageBuf;
    int messageCount;
    int routerID;
    comID comid;
    int *procMap;
    int npes;
    int *pelist;
    
    void checkPeList();

 public:
    EachToManyStrategy(int substrategy);
    EachToManyStrategy(int substrategy, int npes, int *pelist);

    EachToManyStrategy(CkMigrateMessage *m){}

    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    PUPable_decl(EachToManyStrategy);
};
#endif

