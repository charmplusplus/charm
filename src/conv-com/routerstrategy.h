
/* Class that calls Krishnan's routers from the new Comlib.
   Developed to be called from Converse
   Sameer Kumar 05/14/04
*/
   

#ifndef ROUTER_STRATEGY
#define ROUTER_STRATEGY

#include "convcomlibmanager.h"

#include "gridrouter.h"
#include "graphrouter.h"
#include "de.h"
#include "treerouter.h"
#include "3dgridrouter.h"

class RouterStrategy : public Strategy {

    Router * router;
    CkQ<MessageHolder *> msgQ;
    CkQ<char *> recvQ, procQ;
    CkQ<DummyMsg *> dummyQ;

    comID id;
    int *pelist;
    int npes;
    int *procMap;
    int myPe;
    int routerID;

    int doneHandle;    //Array strategy done handle
    int myDoneHandle;   //my own done handle, which will inturn call
                       //array strategy done handle
 
   int doneFlag, bufferedDoneInserting;

    void setReverseMap();

 public:
    //constructor
    //stratid = which topology to use ? (Mesh?, Grid?, Hypercube ?)
    //npes = number of processors in the all to all operation
    //pelist = list of processors
    RouterStrategy(int stratid, int handle, int npes, int *pelist);
    RouterStrategy(CkMigrateMessage *m): Strategy(m){}

    //Insert messages
    void insertMessage(MessageHolder *msg);
    //Finished inserting
    void doneInserting();

    //Call Krishnan's router functions
    void RecvManyMsg(char *msg);
    void ProcManyMsg(char *msg);
    void DummyEP(DummyMsg *m);
    
    void Done(DummyMsg *m);

    int * getProcMap() {return procMap;}

    virtual void pup(PUP::er &p);
    PUPable_decl(RouterStrategy);
};
#endif
