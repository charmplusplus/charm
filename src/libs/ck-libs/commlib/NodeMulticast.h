#ifndef NODE_MULTICAST
#define NODE_MULTICAST
#include "ComlibManager.h"

#if CMK_PERSISTENT_COMM
#include "persistent.h"
#endif

#define MAX_PES_PER_NODE 16
#define PROCESSOR_MODE 0
#define ARRAY_MODE 1

class NodeMulticast : public Strategy {
    CkQ <CharmMessageHolder*> *messageBuf;
    int pes_per_node, *nodeMap, numNodes, myRank, numCurDestPes;
    int mode; //Array destinations or processor destinations

    CkArrayID mAid;
    CkVec<CkArrayIndexMax> *indexVec;
    int NodeMulticastHandlerId, entryPoint, nelements;
    
    int npes, *pelist, NodeMulticastCallbackHandlerId;
    int validRank[MAX_PES_PER_NODE];
    CkCallback cb;
    long handler;

#if CMK_PERSISTENT_COMM
    PersistentHandle *persistentHandlerArray;
#endif

 public:
    NodeMulticast(){}
    void setDestinationArray(CkArrayID a, int nelem, 
			     CkArrayIndexMax **idx, int ep);

    //void setPeList(int npes, int *pelist, CkCallback callback);
    void setPeList(int npes, int *pelist, ComlibMulticastHandler handler);
    
    NodeMulticast(CkMigrateMessage *){}
    void recvHandler(void *msg);
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    CkCallback getCallback() { return cb;}
    ComlibMulticastHandler getHandler() { return (ComlibMulticastHandler)handler;}

    virtual void pup(PUP::er &p);
    PUPable_decl(NodeMulticast);
};
#endif
