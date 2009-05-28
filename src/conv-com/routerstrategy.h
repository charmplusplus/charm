/**
   @addtogroup ComlibConverseStrategy
   @{
   @file
*/




#ifndef ROUTER_STRATEGY
#define ROUTER_STRATEGY

#include "convcomlibmanager.h"
#include "router.h"

void routerProcManyCombinedMsg(char *msg);
void routerRecvManyCombinedMsg(char *msg);
void routerDummyMsg(DummyMsg *msg);

/**
   Class that calls Krishnan's routers from the new Comlib. Developed to be
   called from Converse and from Charm through inheritance by high level
   strategies.

   Strategy optimizes all-to-all communication. It combines messages and sends
   them along virtual topologies 2d mesh, 3d mesh and hypercube routers.

   For large messages send them directly.

   Sameer Kumar 05/14/04
   
*/
class RouterStrategy : public Strategy {

    Router * router;
    CkQ<MessageHolder *> msgQ;
    CkQ<char *> recvQ, procQ;
    CkQ<DummyMsg *> dummyQ;

    comID id;
    /// A list of all the processors involved in the operation, used when
    /// sending direcly without routing. This array is shared with the router.
    int *pelist;
    /// Size of pelist
    int npes;
    /// A sublist of pelist containing only processors which are source
    int *srcPelist;
    /// Size of srcPelist
    int nsrcPes;
    /// A sublist of pelist containing only processors which are destination
    int *destPelist;
    /// size of destPelist
    int ndestPes;
    
    /// A list of size CkNumPes() which associate each processor number with the
    /// position it occupies in pelist (basically procMap[i]=pelist.indexOf(i))
    int *procMap;
    /// A simple array of size npes, where bcast_pemap[i]=i
    int *bcast_pemap;
    /// Position occupied by this processor in the list of processors currently
    /// involved in this operation
    int myPe;
    /// Type of Router used by the strategy as subsystem
    int routerID;

    int doneHandle;    //Array strategy done handle
    //int myDoneHandle;   //my own done handle, which will inturn call
                       //array strategy done handle
 
    int doneFlag, bufferedDoneInserting;

    /// The processor list used to update the knowledge of the strategy is
    /// stored here if when it is delivered there is still an operation in
    /// execution. In this case we have to wait for it to finish, and then
    /// proceed to update the Router.
    int *newKnowledge;
    /// Size of newKnowledge when it is in use
    int newKnowledgeSize;
    /// Similar to newKnowledge only for the source list
    int *newKnowledgeSrc;
    /// Size of newKnowledgeSrc when it is in use
    int newKnowledgeSrcSize;
    /// Similar to newKnowledge only for the destination list
    int *newKnowledgeDest;
    /// Size of newKnowledgeDest when it is in use
    int newKnowledgeDestSize;

    void setReverseMap();

 protected:
    /// Used only by subclasses to initialize partially.
    /// bracketedUpdatePeKnowledge is then used to finish the setup and create
    /// all the final information.
    RouterStrategy(int stratid);

    /// Type of subsystem specified for the strategy, routerID can become
    /// USE_DIRECT if no objects are local, but then it can become again what
    /// required
    int routerIDsaved;

    char isAllToAll() {return id.isAllToAll;}
 public:
    /// Constructor
    /// @param stratid which topology to use ? (Mesh?, Grid?, Hypercube ?).
    /// @param handle converse handle that will be called upon finish of an iteration.
    /// @param nsrc number of source processors in the all to all operation.
    /// @param srclist list of source processors.
    /// @param ndest number of destination processors in the all to all operation.
    /// @param destlist list of destination processors.
    /// If the last two paramenters are not specified then it is assumes source
    /// and destination to be equivalent.
    RouterStrategy(int stratid, int handle, int nsrc, int *srclist, int ndest=0, int *destlist=0);
    RouterStrategy(CkMigrateMessage *m): Strategy(m) {
      ComlibPrintf("[%d] RouterStrategy migration constructor\n",CkMyPe());
    }
    void setupRouter();
    
    ~RouterStrategy();

    Router *getRouter() {return router;}
    comID &getComID() {return id;}

    //Insert messages
    virtual void insertMessage(MessageHolder *msg);
    //Finished inserting
    virtual void doneInserting();

    /// This function is not used since the Router uses its own handlers
    virtual void handleMessage(void *m) {
      CmiAbort("No message should go through RoutingStrategy::handleMessage\n");
    }

    // In converse simply call CmiSyncSendAndFree
    virtual void deliver(char *, int); 
    //{ CmiAbort("No message should go through RoutingStrategy::deliver\n");
    //}

    void bracketedUpdatePeKnowledge(int *count);

    
    virtual void notifyDone();

    int * getProcMap() {return procMap;}

    virtual void pup(PUP::er &p);
    PUPable_decl(RouterStrategy);
};



#endif

/*@}*/
