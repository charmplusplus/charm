#include "ComlibManager.h"

class MPIStrategy : public Strategy {
    CharmMessageHolder *messageBuf;
    int messageCount;

 public:
    MPIStrategy(int substrategy);
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    void setID(comID id);
};
