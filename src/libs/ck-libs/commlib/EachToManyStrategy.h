#include "ComlibManager.h"

class EachToManyStrategy : public Strategy {
    CharmMessageHolder *messageBuf;
    int messageCount;
    int routerID;
    comID comid;

 public:
    EachToManyStrategy(int substrategy);
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    void setID(comID);
};
