#ifndef BRAODCAST_STRATEGY
#define BRAODCAST_STRATEGY

#define DEFAULT_BROADCAST_SPANNING_FACTOR 4

#include "ComlibManager.h"

//Broadcast strategy for charm++ programs using the net version
//This stategy will wonly work for groups.
//This strategy implements a tree based broadcast
//I will extent it for arrays later.
//Developed by Sameer Kumar 04/10/04

class BroadcastStrategy : public CharmStrategy {

    int _topology;         //Topology to use Tree or Hypercube

    int handlerId;          //broadcast handler id
    int spanning_factor;    //the spanning factor of the tree

    double logp;       //ceil of log of number of processors

    void initHypercube();      

    void handleTree(char *msg);
    void handleHypercube(char *msg);    

 public:
    BroadcastStrategy(int topology = USE_HYPERCUBE);
    BroadcastStrategy(CkArrayID aid, int topology = USE_HYPERCUBE);
    BroadcastStrategy(CkMigrateMessage *){}
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    void handleMessage(char *msg);
    void beginProcessing(int nelements);

    virtual void pup(PUP::er &p);
    PUPable_decl(BroadcastStrategy);
};
#endif
