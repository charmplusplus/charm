#ifndef COMMLIBMANAGER_H
#define COMMLIBMANAGER_H

#include "charm++.h"
#include "cksection.h"
#include "envelope.h"
#include "commlib.h"
#include <math.h>

#define USE_TREE 1            //Organizes the all to all as a tree
#define USE_MESH 2            //Virtual topology is a mesh here
#define USE_HYPERCUBE 3       //Virtual topology is a hypercube
#define USE_DIRECT 4          //A dummy strategy that directly forwards 
                              //messages without any processing.
#define USE_GRID 5            //Virtual topology is a 3d grid
#define USE_LINEAR 6          //Virtual topology is a linear array

#define CHARM_MPI 0 
#define MAX_NSTRAT 1024
#define LEARNING_PERIOD 1000 //Number of iterations after which the
                             //learning framework will discover 
                             //the appropriate strategy, not completely 
                             //implemented
#define IS_MULTICAST -1

#define ALPHA 5E-6
#define BETA 3.33E-9

PUPbytes(comID);

#include "commlib.decl.h"

//Dummy message to be sent incase there are no messages to send. 
//Used by only the EachToMany strategy!
class ComlibDummyMsg: public CMessage_ComlibDummyMsg {
    int dummy;
};

/*
//Priority message to call end iteration
class PrioMsg: public CMessage_PrioMsg {
 public:
    int instID;
};
*/

class ComlibMulticastMsg : public CkMcastBaseMsg, 
               public CMessage_ComlibMulticastMsg {
    
  public:
    int nIndices;
    char *usrMsg;        
    CkArrayIndexMax *indices;
};

extern CkGroupID cmgrID;

//An Instance of the communication library.
class ComlibInstanceHandle {
 public:    
    
    int _instid;
    CkGroupID _dmid;

    ComlibInstanceHandle();
    ComlibInstanceHandle(int instid, CkGroupID dmid);    
    
    void init();
    void beginIteration();
    void endIteration();
    
    CkGroupID getComlibManagerID();
    void setStrategy(Strategy *);
};

PUPbytes(ComlibInstanceHandle);

class ComlibManager: public CkDelegateMgr {
    friend class ComlibInstanceHandle;

    int npes;
    int *pelist;

    //CkArrayID dummyArrayID;
    CkArrayIndexMax dummyArrayIndex;

    //For compatibility and easier use!
    int strategyID; //Identifier of the strategy

    StrategyTable strategyTable[MAX_NSTRAT]; //A table of strategy pointers
    CkVec<Strategy *> ListOfStrategies;
    int nstrats, curStratID, prevStratID;      
    //Number of strategies created by the user.

    //flags
    int receivedTable, flushTable, barrierReached, barrier2Reached;
    int totalMsgCount, totalBytes, nIterations;

    ComlibArrayListener *alistener;
    int prioEndIterationFlag;

    void init(); //initialization function

    //charm_message for multicast for a section of that group
    void multicast(CharmMessageHolder *cmsg); //charm message holder here
    //void multicast(void *charm_msg, int npes, int *pelist);

    //The following funtions can be accessed only from ComlibInstanceHandle
    void beginIteration();     //Notify begining of a bracket 
                               //with strategy identifier
    void endIteration();       //Notify end, endIteration must be called if 
                               //a beginIteration is called. Otherwise 
                               //end of the entry method is assumed to 
                               //be the end of the bracket.
    void setInstance(int id); 
    //void prioEndIteration(PrioMsg *pmsg);
    void registerStrategy(int pos, Strategy *s);

 public:
    ComlibManager();  //Receommended constructor
    ComlibManager(CkMigrateMessage *m){ }
    int useDefCtor(void){ return 1; } //Use default constructor should
    //be pupped and store all the strategies.
    
    void barrier(void);
    void barrier2(void);
    void resumeFromBarrier2(void);
    void receiveTable(StrategyWrapper sw);     //Receive table of strategies.

    void ArraySend(int ep, void *msg, const CkArrayIndexMax &idx, 
                   CkArrayID a);
    void GroupSend(int ep, void *msg, int onpe, CkGroupID gid);
    
    virtual void ArrayBroadcast(int ep,void *m,CkArrayID a);
    virtual void GroupBroadcast(int ep,void *m,CkGroupID g);
    virtual void ArraySectionSend(int ep,void *m,CkArrayID a,CkSectionID &s);

    Strategy *getStrategy(int instid)
        {return strategyTable[instid].strategy;}
    StrategyTable *getStrategyTableEntry(int instid)
        {return &strategyTable[instid];}

    //To create a new strategy, returns handle to the strategy table;
    ComlibInstanceHandle createInstance();  
    void doneCreating();             //Done creating instances

    //Learning functions
    void learnPattern(int totalMessageCount, int totalBytes);
    void switchStrategy(int strat);
};

void ComlibDelegateProxy(CProxy *proxy);

ComlibInstanceHandle CkCreateComlibInstance();
ComlibInstanceHandle CkGetComlibInstance();
ComlibInstanceHandle CkGetComlibInstance(int id);

//Only Called when the strategies are not being created in main::main
void ComlibDoneCreating(); 

void ComlibInitSectionID(CkSectionID &sid);
void ComlibDeleteSection();

#endif
