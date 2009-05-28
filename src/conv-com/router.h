/**
   @addtogroup ConvComlibRouter
   @{
   @file 
   @brief Base class for routers. 
*/

#ifndef ROUTER_H
#define ROUTER_H


#include "cklists.h"
#include "convcomlibmanager.h"
#include <math.h>

/*******************************
 * Structures copied from old convcomlib.h - Filippo
 *******************************/
#define MAXNUMMSGS 1000

/** Used between RouterStrategy and Router(s) to pass some information: a
    RouterStrategy passes its comID to the Router it uses, so the second have
    some knowledge of the containing strategy. */
typedef struct {
    int refno;
    int instanceID;  
    char isAllToAll;
} comID;
inline void operator|(PUP::er& p, comID& id) {
  p | id.refno;
  p | id.instanceID;
  p | id.isAllToAll;
}

typedef struct {
  int msgsize;
  void *msg;
} msgstruct ;

typedef struct { 
    char core[CmiReservedHeaderSize];
    comID id;
    int magic;
    int refno;
} DummyMsg ;

//The handler to invoke the RecvManyMsg method of router
CkpvExtern(int, RouterRecvHandle);
//The handler to invoke the ProcManyMsg method of router
CkpvExtern(int, RouterProcHandle);
//The handler to invoke the DoneEP method of router
CkpvExtern(int, RouterDummyHandle);

//Dummy msg handle.
//Just deletes and ignores the message
CkpvExtern(int, RecvdummyHandle);

inline double cubeRoot(double d) {
  return pow(d,1.0/3.0);
}




/** Base class for routers.
    Imported from Krishnan's Each To Many Communication Framework.
    Modified to suit the new communication library.
    Sameer Kumar 05/14/04
*/

class Router 
{
  //int doneHandle;
 protected:
  Strategy *container;
    
 public:
    Router(Strategy *cont) : container(cont) {};
    virtual ~Router() {};
    
    /// Insert messages for the all to all operation,
    /// All processors will be multicast the same message here.
    virtual void EachToAllMulticast(comID id, int size, void *msg, int more)
      {CmiPrintf("Not impl\n");}
    
    /// Insert messages for the all to all operation, The destination
    /// processors to which a message is multicast to be can be
    /// specified.
    /// @param id communication operation identifier
    /// @param size size of the message
    /// @param msg message to be sent
    /// @param numPes number of processors the message has to be sent to
    /// @param pelist list of relative proc ids the message has to be multicast to
    virtual void EachToManyMulticast(comID id, int size, void *msg, 
                                     int numPes, int *pelist, int more)
      {CmiPrintf("Not impl\n");}

    /// Same as EachToManyMulticast, only it receives all the messages into a list
    virtual void EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq){
        MessageHolder *mhdl;
        int len = msgq.length();
        for(int count = 0; count < len - 1; count++) {
            mhdl = msgq.deq();
            EachToManyMulticast(id, mhdl->size, mhdl->getMessage(), 
                                mhdl->npes, mhdl->pelist, 1);
            delete mhdl;
        }
        mhdl = msgq.deq();
        EachToManyMulticast(id, mhdl->size, mhdl->getMessage(), 
                            mhdl->npes, mhdl->pelist, 0);
        delete mhdl;
    }

    //The first iteration of message combining should call this
    //entry function
    virtual void RecvManyMsg(comID, char *) {CmiPrintf("Not Impl\n");}
    
    //The send and the rest of the iterations should call this
    //entry function
    virtual void ProcManyMsg(comID, char *) {CmiPrintf("Not Impl\n");}
    
    //Received a dummy
    virtual void DummyEP(comID, int ) 	{CmiPrintf("Base Dummy\n");}
    
    /// Set the map between processors and virtual processor id's,
    /// Useful when only a subset of processors are involved in the
    /// communication operation.
    virtual void SetMap(int *) {}

    //Utility function
    void SendDummyMsg(comID id, int pe, int magic) {
        
        ComlibPrintf("[%d] Send Dummy to %d\n", CkMyPe(), pe);

        DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        CmiSetHandler(m, CkpvAccess(RouterDummyHandle));
        m->id=id;
        m->magic=magic;
        CmiSyncSendAndFree(pe, sizeof(DummyMsg),(char*) m);
    }

    /// set the handler that will be called when an iteration finishes
    //void setDoneHandle(int handle) {
    //    doneHandle = handle;
    //}

    /// called by the subclasses when the iteration has completed
    void Done(comID id) {
      Strategy *myStrategy = ConvComlibGetStrategy(id.instanceID);
      myStrategy->notifyDone();

        //ComlibPrintf("Router Iteration Finished %d", CkMyPe());
      /*
        if(doneHandle == 0)
            return;

        DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        m->id=id;
        CmiSetHandler(m, doneHandle);
        CmiSyncSendAndFree(CkMyPe(), sizeof(DummyMsg), (char*)m);
      */
    }
};

#endif


/*@}*/
