#ifndef BRAODCAST_STRATEGY
#define BRAODCAST_STRATEGY

#define BROADCAST_SPANNING_FACTOR 8

#include "ComlibManager.h"

//Broadcast strategy for charm++ programs using the net version
//This stategy will wonly work for groups.
//This strategy implements a tree based broadcast
//I will extent it for arrays later.
//Developed by Sameer Kumar 04/10/04

class BroadcastStrategy : public Strategy {

    CkGroupID _gid;
    int _epid;
    int handlerId;

 public:
    BroadcastStrategy(CkGroupID dest_gid, int epid);
    BroadcastStrategy(CkMigrateMessage *){}
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    void handleMessage(char *msg);
    void beginProcessing(int nelements);

    virtual void pup(PUP::er &p);
    PUPable_decl(BroadcastStrategy);
};
#endif
