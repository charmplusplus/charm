#ifndef CONVCOMMLIBSTRATEGY_H
#define CONVCOMMLIBSTRATEGY_H

#include "converse.h"
#include "pup.h"
#include "cklists.h"

//An abstract data structure that holds a converse message and which
//can be buffered by the communication library Message holder is a
//wrapper around a message. Has other useful data like destination
//processor list for a multicast etc.

class MessageHolder : public PUP::able {
 public:
    int dest_proc;
    char *data;
    int size;
    MessageHolder *next; // also used for the refield at the receiver
    int isDummy;
    
    //For multicast, the user can pass the pelist and list of Pes he
    //wants to send the data to.
    int npes;
    int *pelist;
    
    MessageHolder() 
        {dest_proc = size = isDummy = 0; data = NULL;}    

    MessageHolder(CkMigrateMessage *m) {}

    inline MessageHolder(char * msg, int proc, int sz) {
        data = msg;
        dest_proc = proc;
        size = sz;
        
        isDummy = 0;
        
        npes = 0;
        pelist = 0;

    }

    inline ~MessageHolder() {
        /*
          if(pelist != NULL && npes > 0)
          delete[] pelist;
        */
    }

    inline char * getMessage() {
        return data;
    }

    inline int getSize() {
      return size;
    }

    inline void * operator new(size_t size) {
        return CmiAlloc(size);
    }

    inline void operator delete (void *buf) {
        CmiFree(buf);
    }

    virtual void pup(PUP::er &p);
    PUPable_decl(MessageHolder);
};

#define CONVERSE_STRATEGY 0     //The strategy works for converse programs
#define NODEGROUP_STRATEGY 1    //Node group level optimizations 
#define GROUP_STRATEGY 2        //Charm Processor level optimizations
#define ARRAY_STRATEGY 3        //Array level optimizations

//Class that defines the entry methods that a Converse level strategy
//must define. To write a new strategy inherit from this class and
//define the virtual methods.  Every strategy can also define its own
//constructor and have any number of arguments. Also call the parent
//class methods in the virtual methods.

class Strategy : public PUP::able{
 protected:
    int type;
    int isStrategyBracketed;
    int myInstanceID;
    int destinationHandler;

    //Charm strategies for modularity may have converse strategies in
    //them.  For the code to work in both Charm and converse, this
    //variable can be used.    
    Strategy *converseStrategy;
    Strategy *higherLevel;

 public:
    Strategy();
    Strategy(CkMigrateMessage *m) : PUP::able(m) {
        converseStrategy = this;
	higherLevel = this;
    }

    inline void setBracketed(){isStrategyBracketed = 1;}
    inline int isBracketed(){return isStrategyBracketed;}

    //Called for each message
    virtual void insertMessage(MessageHolder *msg) {}
    
    //Called after all chares and groups have finished depositing their 
    //messages on that processor.
    virtual void doneInserting() {}

    inline void setInstance(int instid){myInstanceID = instid;}
    inline int getInstance(){return myInstanceID;}
    inline int getType() {return type;}
    inline void setType(int t) {type = t;}

    inline void setDestination(int handler) {destinationHandler = handler;}
    inline int getDestination() {return destinationHandler;}

    inline void setConverseStrategy(Strategy *s){
        converseStrategy = s;
    }

    inline Strategy * getConverseStrategy() {
        return converseStrategy;
    }

    inline void setHigherLevel(Strategy *s) {
        higherLevel = s;
    }

    Strategy * getHigherLevel() {
      return higherLevel;
    }

    //This method can be used to deliver a message through the correct class
    //when converse does not know if the message was originally sent from
    //converse itself of from a higher level language like charm
    virtual void deliverer(char*, int) {CmiAbort("Strategy::deliverer: If used, should be first redefined\n");};

    //Each strategy must define his own Pup interface.
    virtual void pup(PUP::er &p);
    PUPable_decl(Strategy);
};

//Enables a list of strategies to be stored in a message through the
//pupable framework
class StrategyWrapper  {
 public:
    Strategy **s_table;
    int nstrats;

    void pup(PUP::er &p);
};
PUPmarshall(StrategyWrapper);

//Table of strategies. Each entry in the table points to a strategy.
//Strategies can change during the execution of the program but the
//StrategyTableEntry stores some persistent information for the
//strategy. The communication library on receiving a message, calls
//the strategy in this table given by the strategy id in the message.

struct StrategyTableEntry {
    Strategy *strategy;
    //A buffer for all strategy messages
    CkQ<MessageHolder*> tmplist;
    
    int numElements;   //used by the array listener, 
                       //could also be used for other objects
    int elementCount;  //Count of how many elements have deposited
                       //their data

    //Used during a fence barrier at the begining or during the
    //learning phases. Learning is only available for Charm++
    //programs.
    int nEndItr;       //#elements that called end iteration
    int call_doneInserting; //All elements deposited their data

    StrategyTableEntry();
};

typedef CkVec<StrategyTableEntry> StrategyTable;

#endif
