#include "ComlibManager.h"

class EachToManyStrategy : public Strategy {
    CharmMessageHolder *messageBuf;
    int messageCount;
    int routerID;
    comID comid;
    int *procMap;
    int npes;

 public:
    EachToManyStrategy(int substrategy);
    EachToManyStrategy(int substrategy, int npes, int *pelist);

    EachToManyStrategy(CkMigrateMessage *m){}

    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    PUPable_decl(EachToManyStrategy);
};
