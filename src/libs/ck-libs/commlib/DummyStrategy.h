#include "ComlibManager.h"

class DummyStrategy : public Strategy {
 public:
    DummyStrategy(int substrategy);
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
};
