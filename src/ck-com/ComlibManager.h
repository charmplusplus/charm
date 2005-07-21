#ifndef COMMLIBMANAGER_H
#define COMMLIBMANAGER_H

#include "charm++.h"
#include "cksection.h"
#include "envelope.h"
#include "comlib.h"
#include <math.h>

#include "charm++.h"
#include "convcomlibmanager.h"

#define USE_TREE 1            //Organizes the all to all as a tree
#define USE_MESH 2            //Virtual topology is a mesh here
#define USE_HYPERCUBE 3       //Virtual topology is a hypercube
#define USE_DIRECT 4          //A dummy strategy that directly forwards 
                              //messages without any processing.
#define USE_GRID 5            //Virtual topology is a 3d grid
#define USE_LINEAR 6          //Virtual topology is a linear array
#define USE_PREFIX 7          //Prefix router to avoid contention

#define CHARM_MPI 0 
#define MAX_NSTRAT 1024
#define LEARNING_PERIOD 1000 //Number of iterations after which the
                             //learning framework will discover 
                             //the appropriate strategy, not completely 
                             //implemented
PUPbytes(comID);

#include "ComlibStats.h"

#include "comlib.decl.h"

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

/**
 * Structure used to hold a count of the indeces associated to each pe in a multicast message.
 */
struct ComlibMulticastIndexCount {
  int pe;
  int count;
};

///for use of qsort
inline int indexCountCompare(const void *a, const void *b) {
  ComlibMulticastIndexCount a1 = *(ComlibMulticastIndexCount*) a;
  ComlibMulticastIndexCount b1 = *(ComlibMulticastIndexCount*) b;

  if(a1.pe < b1.pe)
    return -1;
  
  if(a1.pe == b1.pe)
    return 0;
  
  if(a1.pe > b1.pe)
    return 1;
  
  return 0;
}

class ComlibMulticastMsg : public CkMcastBaseMsg, 
               public CMessage_ComlibMulticastMsg {
    
  public:
    int nPes;
    ComlibMulticastIndexCount *indicesCount;
    CkArrayIndexMax *indices;
    char *usrMsg;        
};

class ComlibManager;

//An Instance of the communication library.
class ComlibInstanceHandle : public CkDelegateData {
 private:    
    int _instid;
    CkGroupID _dmid;
    int _srcPe;
    int toForward;

 public:
    ComlibInstanceHandle();
    ComlibInstanceHandle(const ComlibInstanceHandle &h);
    ComlibInstanceHandle(int instid, CkGroupID dmid);    
   
    ComlibInstanceHandle &operator=(const ComlibInstanceHandle &h);

    void setForwardingOnMigration(){toForward = 1;} 
    void beginIteration();
    void endIteration();
    
    CkGroupID getComlibManagerID();
    void setStrategy(CharmStrategy *);
    CharmStrategy *getStrategy();        
    int getSourcePe() {return _srcPe;}

    void setSourcePe() {_srcPe = CkMyPe();}

    friend class ComlibManager;
    void pup(PUP::er &p) {

        if(p.isUnpacking())
             reset();        

        p | _instid;
        p | _dmid;
        p | _srcPe;
        p | toForward;
    }
};

class LBMigrateMsg;

class ComlibManager: public CkDelegateMgr {
    friend class ComlibInstanceHandle;

    int *bcast_pelist;  //Pelist passed to all broadcast operations

    int section_send_event;

    int remotePe;
    CmiBool isRemote;
    CmiBool strategyCreated;

    int npes;
    int *pelist;

    CkArrayIndexMax dummyArrayIndex;

    //For compatibility and easier use!
    int strategyID; //Identifier of the strategy

    //Pointer to the converse comm lib strategy table
    StrategyTable *strategyTable;

    CkQ<CharmStrategy *> ListOfStrategies; //temporary list of strategies
    
    CkQ<CharmMessageHolder *> remoteQ;  //list of remote messages
                                        //after the object has
                                        //migrated

    //The number of strategies created by the user
    //int nstrats; //now part of conv comlib
    
    int curStratID, prevStratID;      
    //Number of strategies created by the user.

    //flags
    int receivedTable, setupComplete, barrierReached, barrier2Reached;
    CmiBool lbUpdateReceived;

    int bcount , b2count;
    //int totalMsgCount, totalBytes, nIterations;

    ComlibArrayListener *alistener;
    int prioEndIterationFlag;

    ComlibGlobalStats clib_gstats; 
    int    numStatsReceived;

    int curComlibController;   //Processor where strategies are  recreated
    int clibIteration;         //Number of such learning iterations,
                               //each of which is triggered by a
                               //loadbalancing operation

    void init(); //initialization function

    //charm_message for multicast for a section of that group
    void multicast(CharmMessageHolder *cmsg); //charm message holder here
    //void multicast(void *charm_msg, int npes, int *pelist);

    //The following funtions can be accessed only from ComlibInstanceHandle
    void beginIteration();     //Notify begining of a bracket with
                               //strategy identifier

    void endIteration();       //Notify end, endIteration must be
                               //called if a beginIteration is
                               //called. Otherwise end of the entry
                               //method is assumed to be the end of
                               //the bracket.
    
    void setInstance(int id); 

    //void prioEndIteration(PrioMsg *pmsg);
    void registerStrategy(int pos, CharmStrategy *s);

 public:

    ComlibLocalStats clib_stats;   //To store statistics of
                                   //communication operations
    
    ComlibManager();  //Recommended constructor

    ComlibManager(CkMigrateMessage *m) { }
    int useDefCtor(void){ return 1; } //Use default constructor should
    //be pupped and store all the strategies.
    
    void barrier(void);
    void barrier2(void);
    void resumeFromBarrier2(void);

    //Receive table of strategies.
    void receiveTable(StrategyWrapper &sw, 
                      CkHashtableT <ClibGlobalArrayIndex, int>&); 

    void ArraySend(CkDelegateData *pd,int ep, void *msg, 
                   const CkArrayIndexMax &idx, CkArrayID a);

    void receiveRemoteSend(CkQ<CharmMessageHolder*> &rq, int id);
    void sendRemote();

    void GroupSend(CkDelegateData *pd, int ep, void *msg, int onpe, 
                   CkGroupID gid);
    
    virtual void ArrayBroadcast(CkDelegateData *pd,int ep,void *m,CkArrayID a);
    virtual void GroupBroadcast(CkDelegateData *pd,int ep,void *m,CkGroupID g);
    virtual void ArraySectionSend(CkDelegateData *pd, int ep ,void *m, 
                                  CkArrayID a, CkSectionID &s, int opts);

    CharmStrategy *getStrategy(int instid)
        {return (CharmStrategy *)(* strategyTable)[instid].strategy;}

    StrategyTableEntry *getStrategyTableEntry(int instid)
        {return &((*strategyTable)[instid]);}

    //To create a new strategy, returns handle to the strategy table;
    ComlibInstanceHandle createInstance();  
    void broadcastStrategies();             //Done creating instances

    void AtSync();           //User program called loadbalancer
    void lbUpdate(LBMigrateMsg *); //loadbalancing updates

    //Learning functions
    //void learnPattern(int totalMessageCount, int totalBytes);
    //void switchStrategy(int strat);

    void setRemote(int remotePe);

    void collectStats(ComlibLocalStats &s, int src,CkVec<ClibGlobalArrayIndex>&);

    //Returns the processor on which the comlib sees the array element
    //belonging to
    inline int getLastKnown(CkArrayID a, CkArrayIndexMax &idx) {
        return ComlibGetLastKnown(a, idx);
    }

    CkDelegateData* ckCopyDelegateData(CkDelegateData *data); 
    CkDelegateData *DelegatePointerPup(PUP::er &p,CkDelegateData *pd);
};

void ComlibDelegateProxy(CProxy *proxy);
void ComlibAssociateProxy(ComlibInstanceHandle *cinst, CProxy &proxy);
void ComlibAssociateProxy(CharmStrategy *strat, CProxy &proxy); 
ComlibInstanceHandle ComlibRegister(CharmStrategy *strat);
void ComlibBegin(CProxy &proxy);    
void ComlibEnd(CProxy &proxy);    

ComlibInstanceHandle CkCreateComlibInstance();
ComlibInstanceHandle CkGetComlibInstance();
ComlibInstanceHandle CkGetComlibInstance(int id);

void ComlibResetSectionProxy(CProxySection_ArrayBase *sproxy);

inline void ComlibResetProxy(CProxy *aproxy) {
  ComlibInstanceHandle *handle = 
    (ComlibInstanceHandle *) aproxy->ckDelegatedPtr();
  handle->setSourcePe();
}

//Only Called when the strategies are not being created in main::main
void ComlibDoneCreating(); 

void ComlibInitSectionID(CkSectionID &sid);

void ComlibAtSync(void *msg);
void ComlibNotifyMigrationDoneHandler(void *msg);
void ComlibLBMigrationUpdate(LBMigrateMsg *);

#define RECORD_SEND_STATS(sid, bytes, dest) {             \
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID));               \
        cgproxy.ckLocalBranch()->clib_stats.recordSend(sid, bytes, dest); \
}\

#define RECORD_RECV_STATS(sid, bytes, src) { \
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID)); \
        cgproxy.ckLocalBranch()->clib_stats.recordRecv(sid, bytes, src); \
}\

#define RECORD_SENDM_STATS(sid, bytes, dest_arr, ndest) {       \
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID)); \
        cgproxy.ckLocalBranch()->clib_stats.recordSendM(sid, bytes, dest_arr, ndest); \
}\

#define RECORD_RECVM_STATS(sid, bytes, src_arr, nsrc) {        \
        CProxy_ComlibManager cgproxy(CkpvAccess(cmgrID)); \
        cgproxy.ckLocalBranch()->clib_stats.recordRecvM(sid, bytes, src_arr, nsrc); \
}\

#endif
