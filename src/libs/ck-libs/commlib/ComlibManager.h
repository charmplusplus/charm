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
#define USE_GROUP_BY_PROCESSOR //Groups messages by destination processor 
                               //(does not work as of now)
#define USE_GRID 5            //Virtual topology is a 3d grid
#define NAMD_STRAT 6          //A speciliazed strategy for Namd, commented out
#define USE_MPI 7             //Calls MPI_Alltoall
#define USE_STREAMING 8       //Creates a message stream with periodic combining

#define CHARM_MPI 0 

#define LEARNING_PERIOD 2     //Number of iterations after which the 
                              //learning framework will discover the appropriate 
                              //strategy, not completely implemented
#define ALPHA 5E-6
#define BETA 3.33E-9

//An abstract data structure that holds a charm++ message 
//and provides utility functions to manage it.
class CharmMessageHolder {
 public:
    int dest_proc;
    char *data;
    CharmMessageHolder *next; // also used for the refield at the receiver

    CharmMessageHolder(char * msg, int dest_proc);
    char * getCharmMessage();
    void copy(char *buf);
    int getSize();
    void init(char *root);
    void setRefcount(char * root_msg);
};

//An abstract class that defines the entry methods that a strategy must define. 
//To write a new strategy inherit from this class and define the methods. 
//Notice there is no constructor. Every strategy can define its own 
//constructor and have any number of arguments.
//But for now the strategies can only receive an int in their constructor.
class Strategy {
 public:
    //Called for each message
    virtual void insertMessage(CharmMessageHolder *msg) = 0;
    //Called after all chares and groups have finished depositing their messages 
    //on that processor.
    virtual void doneInserting() = 0;

    //Needed for compatibility with older versions, I will get rid of it soon!
    virtual void setID(comID id) {}

    //Will enable this later
    //Each strategy must define his own Pup interface.
    //    virtual void pup(PUP::er &p) {}
};
//PUPmarshall(Strategy);

#include "ComlibModule.decl.h"

//Dummy message to be sent incase there are no messages to send. 
//Used by only the EachToMany strategy!
class DummyMsg: public CMessage_DummyMsg {
    int dummy;
};
/*
class StrategyMsg : public CMessage_StrategyMsg {
    
}
*/
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

    //int *procMap; //Should ideally belong to the strategy

    int createDone, doneReceived;

    int npes;
    int *pelist;

    int nelements; //number of array elements on one processor
    int elementCount; //counter for the above
    
    int strategyID; //Identifier of the strategy
    Strategy *strategy; //Pointer to the strategy class
    comID comid;
    
    //flags
    int idSet, iterationFinished;
    
    void init(Strategy *s); //strategy, nelements 
    void setReverseMap(int *, int);

    int totalMsgCount, totalBytes, nIterations;

 public:
    ComlibManager(int s); //strategy, nelements 
    ComlibManager(int s, int n); //strategy, nelements 

    //ComlibManager(Strategy *str); //Needs pup routines for strategies to be written

    void done();
    void localElement();

    void receiveID(comID id);
    void receiveID(int npes, int *pelist, comID id);

    void ArraySend(int ep, void *msg, const CkArrayIndexMax &idx, CkArrayID a);
    void GroupSend(int ep, void *msg, int onpe, CkGroupID gid);
    //    void setNumMessages(int nmessages);
    
    void beginIteration();
    void endIteration();

    //void receiveNamdMessage(ComlibMsg * msg);
    void createId();
    void createId(int *, int);

    //Learning functions
    void learnPattern(int totalMessageCount, int totalBytes);
    void switchStrategy(int strat);
};

#endif
