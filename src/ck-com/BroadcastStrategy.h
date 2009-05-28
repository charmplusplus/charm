/**
   @addtogroup ComlibCharmStrategy
   @{
   @file
*/

#ifndef BRAODCAST_STRATEGY
#define BRAODCAST_STRATEGY

#define DEFAULT_BROADCAST_SPANNING_FACTOR 4

#include "ComlibManager.h"

/**
   Broadcast strategy for charm++ programs using the net version.
   This stategy will only work for groups.
   This strategy implements a tree based broadcast

   Developed by Sameer Kumar 04/10/04

   @warning This strategy works only in particular situations and is not
   generic. Its usage is adviced against.

*/

class BroadcastStrategy : public Strategy, public CharmStrategy {

    int _topology;         //Topology to use Tree or Hypercube

    //int handlerId;          //broadcast handler id
    int spanning_factor;    //the spanning factor of the tree

    double logp;       //ceil of log of number of processors

    void initHypercube();      

    void handleTree(void *msg);
    void handleHypercube(void *msg);    

 public:
    BroadcastStrategy(int topology = USE_HYPERCUBE);
    BroadcastStrategy(CkArrayID aid, int topology = USE_HYPERCUBE);
    BroadcastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m) {}
    void insertMessage(MessageHolder *msg) {insertMessage((CharmMessageHolder*)msg);};
    void insertMessage(CharmMessageHolder *msg);
    //void doneInserting();

    void handleMessage(void *msg);
    //void beginProcessing(int nelements);

    virtual void pup(PUP::er &p);
    PUPable_decl(BroadcastStrategy);
};
#endif

/*@}*/
