#ifndef COMMLIBSTRATEGY_H
#define COMMLIBSTRATEGY_H

//An abstract data structure that holds a charm++ message 
//and provides utility functions to manage it.
class CharmMessageHolder {
 public:
    int dest_proc;
    char *data;
    CharmMessageHolder *next; // also used for the refield at the receiver
    int isDummy;
    
    //For multicast, the user can pass the pelist and list of Pes he
    //wants to send the data to.
    int npes;
    int *pelist;

    //For array multicast
    int nIndices;
    CkArrayIndexMax *indexlist;
    
    CharmMessageHolder(char * msg, int dest_proc);
    ~CharmMessageHolder();

    char * getCharmMessage();
};

//Class that defines the entry methods that a strategy must define.
//To write a new strategy inherit from this class and define the
//virtual methods.  Every strategy can also define its own constructor
//and have any number of arguments. Also call the parent class methods
//in those methods.

#include "charm++.h"

class Strategy : public PUP::able{
 protected:
    int isArray;
    int isGroup;

    CkArrayID aid;
    CkGroupID gid;

    CkArrayIndexMax *elements; //src array elements
    int nIndices;              //number of source indices    
    
    int *srcpelist, nsrcpes; //src processors for the elements

    int myInstanceID;
    CkVec<CkArrayIndexMax> localSrcIndices;

 public:
    Strategy();
    Strategy(CkMigrateMessage *) {};

    void setSourceArray(CkArrayID aid, CkArrayIndexMax *e=0, int nind=0);
    void setSourceGroup(CkGroupID gid, int *srcpelist=0, int nsrcpes=0) ;    
    
    int isSourceArray(){return isArray;}
    int isSourceGroup(){return isGroup;}

    void getSourceArray(CkArrayID &aid, CkArrayIndexMax *&e, int &nind);
    void getSourceGroup(CkGroupID &gid, int *&pelist, int &npes);

    //Called for each message
    virtual void insertMessage(CharmMessageHolder *msg) {};

    //Called after all chares and groups have finished depositing their 
    //messages on that processor.
    virtual void doneInserting() {};

    //Each strategy must define his own Pup interface.
    virtual void pup(PUP::er &p);

    virtual void beginProcessing(int nelements){};

    virtual void setInstance(int instid){myInstanceID = instid;}
    virtual int getInstance(){return myInstanceID;}
    
    //virtual void insertLocalIndex(CkArrayIndexMax idx)
    // {localSrcIndices.insertAtEnd(idx);}    
    //virtual void removeLocalIndex(CkArrayIndexMax);

    PUPable_decl(Strategy);
};

class StrategyWrapper  {
 public:
    Strategy **s_table;
    int nstrats;

    void pup(PUP::er &p);
};
PUPmarshall(StrategyWrapper);

struct StrategyTable {
    Strategy *strategy;
    CkQ<CharmMessageHolder*> tmplist;
    int numElements;
    int elementCount;
    int nEndItr;
    int call_doneInserting;
};

#endif
