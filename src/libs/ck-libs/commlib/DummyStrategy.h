#ifndef DUMMY_STRATEGY
#define DUMMY_STRATEGY
#include "ComlibManager.h"

class DummyStrategy : public Strategy {
 public:
    DummyStrategy();
    DummyStrategy(CkMigrateMessage *){}
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    PUPable_decl(DummyStrategy);
};
#endif
