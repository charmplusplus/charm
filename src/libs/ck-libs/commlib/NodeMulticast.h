#ifndef NODE_MULTICAST
#define NODE_MULTICAST
#include "ComlibManager.h"

class NodeMulticast : public Strategy {
    CkQ <CharmMessageHolder*> *messageBuf;
    int pes_per_node, *nodeMap, numNodes, myRank;
    CkArrayID mAid;
    CkVec<CkArrayIndexMax> *indexVec;
    int NodeMulticastHandlerId, entryPoint, nelements;

 public:
    NodeMulticast(){}
    void setDestinationArray(CkArrayID a, int nelem, 
			     CkArrayIndexMax **idx, int ep);
    NodeMulticast(CkMigrateMessage *){}
    void recvHandler(void *msg);
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    
    virtual void pup(PUP::er &p);
    PUPable_decl(NodeMulticast);
};
#endif
