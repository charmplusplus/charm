#ifndef COMMLIBMANAGER_H
#define COMMLIBMANAGER_H

#include "charm++.h"
#include "converse.h"
#include "envelope.h"
#include "commlib.h"
#include <math.h>

#define USE_DIRECT 0          //A dummy strategy that directly forwards 
                              //messages without any processing.
#define USE_TREE 1            //Organizes the all to all as a tree
#define USE_MESH 2            //Virtual topology is a mesh here
#define USE_HYPERCUBE 3       //Virtual topology is a hypercube
#define USE_GROUP_BY_PROCESSOR 4 //Groups messages by destination processor 
                                 //(does not work as of now)
#define USE_GRID 5            //Virtual topology is a 3d grid
#define NAMD_STRAT 6          //A speciliazed strategy for Namd, commented out
#define USE_MPI 7             //Calls MPI_Alltoall
#define USE_STREAMING 8       //Creates a message stream with periodic combining

#define CHARM_MPI 0 
#define MAX_NSTRAT 1024
#define LEARNING_PERIOD 1000     //Number of iterations after which the 
                              //learning framework will discover the appropriate 
                              //strategy, not completely implemented
#define ALPHA 5E-6
#define BETA 3.33E-9

PUPbytes(comID);

//An abstract data structure that holds a charm++ message 
//and provides utility functions to manage it.
class CharmMessageHolder {
 public:
    int dest_proc;
    char *data;
    CharmMessageHolder *next; // also used for the refield at the receiver
    
    //For multicast, the user can pass the pelist and list of Pes he
    //wants to send the data to.
    int npes;
    int *pelist;

    CharmMessageHolder(char * msg, int dest_proc);
    char * getCharmMessage();
    void copy(char *buf);
    int getSize();
    void init(char *root);
    void setRefcount(char * root_msg);
};

//Class that defines the entry methods that a strategy must define. 
//To write a new strategy inherit from this class and define the methods. 
//Notice there is no constructor. Every strategy can define its own 
//constructor and have any number of arguments.
//But for now the strategies can only receive an int in their constructor.
class Strategy : public PUP::able{
 public:
    Strategy() {};
    Strategy(CkMigrateMessage *) {};

    //Called for each message
    virtual void insertMessage(CharmMessageHolder *msg) {};

    //Called after all chares and groups have finished depositing their 
    //messages on that processor.
    virtual void doneInserting() {};

    //Each strategy must define his own Pup interface.
    virtual void pup(PUP::er &p){
        PUP::able::pup(p);
    }
    PUPable_decl(Strategy);
};

class StrategyWrapper  {
 public:
    Strategy **s_table;
    int nstrats;

    void pup(PUP::er &p);
};
PUPmarshall(StrategyWrapper);

#include "ComlibModule.decl.h"

//Dummy message to be sent incase there are no messages to send. 
//Used by only the EachToMany strategy!
class DummyMsg: public CMessage_DummyMsg {
    int dummy;
};

struct StrategyTable {
    Strategy *strategy;
    CkQ<CharmMessageHolder*> tmplist;
    int numElements;
    int elementCount;
    int call_doneInserting;
};

/*
//A wrapper message for many charm++ messages.
class ComlibMsg: public CMessage_ComlibMsg {
 public:
    int nmessages;
    int curSize;
    int destPE;
    char* data;
    
    void insert(CharmMessageHolder *msg);
    CharmMessageHolder * next();
};
*/

class ComlibManager: public CkDelegateMgr{

    CkGroupID cmgrID;

    int npes;
    int *pelist;

    //For compatibility and easier use!
    int strategyID; //Identifier of the strategy

    StrategyTable strategyTable[MAX_NSTRAT]; //A table of strategy pointers
    CkQ<Strategy *> ListOfStrategies;
    int nstrats, curStratID;      //Number of strategies created by the user.

    //flags
    int receivedTable, flushTable, barrierReached, barrier2Reached;
    int totalMsgCount, totalBytes, nIterations;
   
    void init(); //initialization function

 public:
    ComlibManager();             //Receommended constructor
    ComlibManager(int s);        //strategy
    ComlibManager(int s, int n); //strategy, nelements

    void barrier(void);
    void barrier2(void);
    void resumeFromBarrier2(void);

    void localElement();
    void registerElement(int strat);    //Register a chare for an instance
    void unRegisterElement(int strat);  //UnRegister a chare for an instance

    void receiveID(comID id);                        //Depricated
    void receiveID(int npes, int *pelist, comID id); //Depricated
    void receiveTable(StrategyWrapper sw);      //Receive table of strategies.

    void ArraySend(int ep, void *msg, const CkArrayIndexMax &idx, CkArrayID a);
    void GroupSend(int ep, void *msg, int onpe, CkGroupID gid);
    void multicast(void *charm_msg); //charm_message here.

    void beginIteration();
    void beginIteration(int id); //Notify begining of an iteration 
                                 //with strategy identifier
    void endIteration();         //Notify end

    void createId();                 //depricated
    void createId(int *, int);       //depricated
    int createInstance(Strategy *);  //To create a new strategy, 
                                     //returns index to the strategy table;
    void doneCreating();             //Done creating instances

    //Learning functions
    void learnPattern(int totalMessageCount, int totalBytes);
    void switchStrategy(int strat);
};

#endif
