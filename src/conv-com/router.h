#ifndef ROUTER_H
#define ROUTER_H

//Base class for routers
//Imported from Krishnan's Each To Many Communication Framework
//Modified to suit the new communication library
//Sameer Kumar 05/14/04

class Router 
{
    int doneHandle;
    
 public:
    Router() {};
    virtual ~Router() {};
    
    //Insert messages for the all to all operation
    //All processors will be multicast the same message here
    virtual void EachToAllMulticast(comID id, int size, void *msg, 
                                    int more) 
        {CmiPrintf("Not impl\n");}
    
    //Insert messages for the all to all operation. The destination
    //processors to which a message is multicast to be can be
    //specified
    //id = communication operation identifier
    //size = size of the message
    //msg = message to be sent
    //numPes = number of processors the message has to be sent to
    //pelist = list of relative proc ids the message has to be multicast to
    //more = do I get more messages ?
    virtual void EachToManyMulticast(comID id, int size, void *msg, 
                                     int numPes, int *pelist, 
                                     int more) 
        {CmiPrintf("Not impl\n");}
    
    //The first iteration of message combining should call this
    //entry function
    virtual void RecvManyMsg(comID, char *) {CmiPrintf("Not Impl\n");}
    
    //The send and the rest of the iterations should call this
    //entry function
    virtual void ProcManyMsg(comID, char *) {CmiPrintf("Not Impl\n");}
    
    //Received a dummy
    virtual void DummyEP(comID, int ) 	{CmiPrintf("Base Dummy\n");}
    
    //Set the map between processors and virtual processor id's
    //Useful when only a subset of processors are involved in the
    //communication operation
    virtual void SetMap(int *) {;}

    //Utility function
    void SendDummyMsg(comID id, int pe, int magic) {
        
        ComlibPrintf("[%d] Send Dummy to %d\n", CkMyPe(), pe);

        DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        CmiSetHandler(m, CkpvAccess(DummyHandle));
        m->id=id;
        m->magic=magic;
        CmiSyncSendAndFree(pe, sizeof(DummyMsg),(char*) m);
    }

    void setDoneHandle(int handle) {
        doneHandle = handle;
    }

    void Done(comID id) {

        ComlibPrintf("Router Iteration Finished %d", CkMyPe());

        if(doneHandle == 0)
            return;

        DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        m->id=id;
        CmiSetHandler(m, doneHandle);
        CmiSyncSendAndFree(CkMyPe(), sizeof(DummyMsg), (char*)m);
    }
};

#endif
