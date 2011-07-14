#ifndef COMLIBMANAGER_H
#define COMLIBMANAGER_H


/**
   @{
   
   @file

   @brief Charm ComlibManager
   Contains the classes ComlibManager (charm++ delegated group) and
   ComlibDelegateData (its delegated data). It also declares the user
   interface to the framework, all beginning with the prefix "Comlib".


   @defgroup Comlib  Communication Optimization Framework
   @brief A communication optimization framework.

   Comlib is a framework for delegating converse and charm++ level communications to optimizing strategies which can use appropriate topological routers.

   The comlib framework underwent a large refactoring that was committed into CVS in February 2009. The framework was extricated from ck-core. Various files and classes were renamed, and many bugs were fixed. A new system for bracketed strategies was introduced. 

   Bracketed communication strategies, such as those for all-to-all communication, now have knowledge that is updated on demand. If errors are detected in the knowledge base (eg. a chare array element is no longer located on some PE), the strategies enter an error mode. After the knowledge has been updated, the strategies will exit the error mode and send messages along the new optimized paths. While in the error states, the strategies send existing messages directly, and buffer new messages.


<h2>Usage Restrictions</h2>
Strategies should be created in a MainChare's Main method (hence on PE 0). Proxies can be created later and these can be delegated to the existing strategies. 

  
<h2>Startup</h2>

  The initilization of Comlib is done both asynchronously for parts and at startup for other parts. There is an initproc routine 
  initConvComlibManager() that instantiates the Ckpv processor local conv_com_object. This needs to be created before the ComlibManagerMain method is called. Because we cannot guarantee the order for which the mainchares execute, we must do this in an initproc routine.


   The startup of Comlib happens asynchronously. The <i>mainchare ComlibManagerMain</i> has a constructor that runs along with all other mainchares while charm++ messages are not activated at startup. This constructor simply sets a few variables from command line arguments, and then creates the <i>ComlibManager group </i>. After all mainchares have run (in no determinable order), then the charm++ system will release all charm++ messages. 

   At this point the user program will continue asynchronously with the comlib startup.
  
   Then ComlibManager::ComlibManager() calls ComlibManager::init(), sets up a few variables, and then calls ComlibManager::barrier(). After barrier() has been called by all PEs, it calls ComlibManager::resumeFromSetupBarrier(). 

   ComlibManager::resumeFromSetupBarrier() completes the initialization of the charm layer of comlib after all group branches are created. It is guaranteed that Main::Main has already run at this point, so all strategies created there can be broadcast.  This function calls ComlibDoneCreating() and then it sends all messages that were buffered (in unCompletedSetupBuffer).

   ComlibDoneCreating() will do nothing on all PE != 0. On PE 0, it will call ConvComlibManager::doneCreating(). The strategy table will broadcast at this point.

   The process for broadcasting the strategy table is as follows (see convcomlibmanager.C):
 
   <ol>
   <li>the strategies are inserted on processor 0 (and possibly in other
   processors with the same order. The strategies are marked as "new"
   <li>when ConvComlibManager::doneCreating() is called, processor 0 broadcasts all the new
   strategies to all the processors, and marks them as "inSync"  
   <li>when a processor receives a table it updates its personal table with the
   incoming, it marks all the strategies just arrived as "inSync", and it
   sends an acknowledgement back to processor 0.   
   <li>when an acknowledgement is received by processor 0, a counter is
   decremented. When it reaches 0, all the "inSync" strategies are switched
   to status "ready" and they can start working. All the messages in the
   tmplist are delivered. The sync is broadcasted.   
   <li>when an acknowledgement is received by a processor other than 0, all the
   "inSync" strategies are switched to "ready" and the messages in tmplist
   are delivered.   
   <li>in order to prevent two consecutive table broadcasts to interfere with
   each other, an additional acknowledgement is sent back by each processor
   to processor 0 to allow a new table update to happen.
   </ol>

<h2>Startup: Buffering of Messages</h2>

   Because the startup of Comlib happens asynchronously. Thus, if a user program sends messages through a comlib strategy, and the strategy has not yet started up completely, then the messages may be delayed in one of two queues. 
   
<ol>
   <li>CkQ<MessageHolder*> tmplist; found in convcomlibstrategy.h buffers converse level messages when the converse strategies are not ready.
   <li>std::map<ComlibInstanceHandle, std::set<CharmMessageHolder*> > ComlibManager::delayMessageSendBuffer in ComlibManager.h buffers charm level messages at startup before ComlibManager::resumeFromSetupBarrier() or while a strategy is in an error state. Messages are flushed from here once both the startup has finished and the strategy is not in an error state. The messages are flushed from one place: ComlibManager::sendBufferedMessages().
</ol>
  



<h2>Bracketed Strategies</h2>
<h3>Usage of Bracketed Strategies</h3>
Bracketed strategies have the following usage pattern. For each iteration of the program: 
<ol>
<li>Each source object calls ComlibManager::beginIteration(int instid, int iteration)
<li>Each source object invokes one or more entry method(s) on the delegated proxy
<li>Each source object then calls ComlibManager::endIteration().
</ol>

<h3>Restrictions on Bracketed Strategies</h3>
<ol>
<li>The user application is not allowed to call beginIteration for iteration n until all messages from iteration n-1 have been received.
<li>Migrations of elements are not allowed between when they call ComlibManager::beginIteration and the associated ComlibManager::endIteration for the same iteration.
</ol>

<h3>Detecting migrations in Bracketed Strategies</h3>
The instance of each strategy on each PE maintains a list of the local array elements, and the last known iteration value. The current implementation only detects migrations when a PE gains a net positive number of migrated objects. All of the objects on that PE will call ComlibManager::beginIteration. Because the strategy knows how many elements were previously on the PE, it will detect more calls to ComlibManager::beginIteration than its previous element count. At this point, the future messages for the strategy will be enqueued in a buffer (ComlibManager::delayMessageSendBuffer). Once ComlibManager::endIteration() is called, the error recovery protocol will be started. All PEs will cause any objects that have migrated away to report back to PE 0, which updates a list of object locations. Once all PEs and migrated objects have reported back to PE 0, the updated PE list will be broadcast to all PEs, and the strategy will be enabled again. At this point any buffered messages will be released. The subsequent iteration of the application should then be optimized.

If two objects swap places between two PEs, the current implementation does not detect this change. In the future ComlibManager::beginIteration should compare the object to the list of known local objects, and start buffering messages and correcting this error condition.



   
   @defgroup ConvComlib  Converse Communication Optimization Framework 
   @ingroup Comlib
   @brief Framework for delegating converse level communication to Comlib.


   @defgroup CharmComlib  Charm++ Communication Optimization Framework 
   @ingroup Comlib
   @brief Framework for delegating Charm++ level communication to Comlib. 


   @defgroup ConvComlibRouter Converse level message Routers
   @ingroup ConvComlib
   @ingroup Comlib
   @brief Routers used by converse strategies to route messages in certain topologies : grid, hypercube, etc.

   Each router inherits from Router. 


   @defgroup ComlibCharmStrategy Strategies for use in Charm++
   @ingroup CharmComlib
   @ingroup Comlib
   @brief Communication optimizing strategies for use in Charm++ programs.

   These strategies are used in Charm++ programs, by creating proxies that are then associated with a strategy. The future method invocations on the proxy are then handled by the strategy. 


   @defgroup ComlibConverseStrategy Strategies for use in converse
   @ingroup ConvComlib
   @ingroup Comlib
   @brief Communication optimizing strategies for use in converse programs or in other comlib strategies.

*/



#include <math.h>
#include <map>
#include <set>

#include "convcomlibmanager.h"
#include "ComlibStrategy.h"
#include "ComlibArrayListener.h"
#include "cksection.h"

#define CHARM_MPI 0 
#define MAX_NSTRAT 1024
#define LEARNING_PERIOD 1000 //Number of iterations after which the
                             //learning framework will discover 
                             //the appropriate strategy, not completely 
                             //implemented
#include "ComlibStats.h"

#include "comlib.decl.h"

CkpvExtern(CkGroupID, cmgrID);
///Dummy message to be sent in case there are no messages to send. 
///Used by only the EachToMany strategy!
class ComlibDummyMsg: public CMessage_ComlibDummyMsg {
    int dummy;
};



/**
 * Structure used to hold a count of the indeces associated to each pe in a
 * multicast message.
 */
// TO BE MOVED TO MULTICAST SPECIFIC
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


/** A class for multicast messages that contains the user message 
    as well as a list of all destination indices and corresponding PEs 
*/
class ComlibMulticastMsg : public CkMcastBaseMsg, 
               public CMessage_ComlibMulticastMsg {
  public:
    int nPes;
    ComlibMulticastIndexCount *indicesCount;
    CkArrayIndex *indices;
    char *usrMsg;        
};


void ComlibAssociateProxy(ComlibInstanceHandle cinst, CProxy &proxy);
void ComlibAssociateProxy(Strategy *strat, CProxy &proxy); 

/// @deprecated
ComlibInstanceHandle ComlibRegister(Strategy *strat);

void ComlibBegin(CProxy &proxy, int iteration);    
void ComlibEnd(CProxy &proxy, int iteration);    

/** The behaviour has changed from the previous version:
    while before this function was used to reset a proxy after a migration,
    now it is used to reset a proxy before it is reassociated with another strategy. 
 */
inline void ComlibResetSectionProxy(CProxySection_ArrayBase &sproxy) {
  sproxy.ckGetSectionInfo().info.sInfo.cInfo.id = 0;
}


void ComlibInitSectionID(CkSectionID &sid);

class LBMigrateMsg;

/** The main group doing the management of all the charm system. It takes care
    of calling the strategies when a message arrives, and modifying them when
    needed by the learning framework. It relies on ConvComlibManager for the
    processor operations involved. It installes itself as a delegated class from
    CkDelegateMgr, overwriting the standard path of message sending in charm.
 */
class ComlibManager: public CkDelegateMgr {
  //friend class ComlibInstanceHandle;

    int *bcast_pelist;  //Pelist passed to all broadcast operations

    /// Used to register and record events into projection
    int section_send_event;

    CkArrayIndex dummyArrayIndex;

    /// Pointer to the converse comlib object, for efficiency over calling CkpvAccess
    ConvComlibManager *converseManager;


    /// Lists of messages whose delivery will be postponed until the comlib strategy has been fully setup, and 
    /// the strategy has exited an error state
    /// The map key is a comlib instance handle
    /// The map value is a set of messages
    std::map<ComlibInstanceHandle, std::set<CharmMessageHolder*> > delayMessageSendBuffer;
   
 
    /// Different than 0 once this group has been created on all processors
    int setupComplete;
  
    int prioEndIterationFlag;

    ComlibGlobalStats clib_gstats; 
    int numStatsReceived;

    int curComlibController;   //Processor where strategies are created
    int clibIteration;         //Number of such learning iterations,
                               //each of which is triggered by a
                               //loadbalancing operation

    void init(); ///Initialization function

    //charm_message for multicast for a section of that group
    void multicast(CharmMessageHolder *cmsg, int instid); //charm message holder here
    //void multicast(void *charm_msg, int npes, int *pelist);

    //The following funtions can be accessed only from the user provided hooks
    friend void ComlibBegin(CProxy&, int iteration);
    friend void ComlibEnd(CProxy&, int iteration);

    ///Notify begining of a bracket with strategy identifier
    void beginIteration(int instid, int iteration);
    
    ///Notify end, endIteration must be called if a beginIteration is called.
    ///Otherwise end of the entry method is assumed to be the end of the
    ///bracket.
    void endIteration(int instid, int iteration);
    
    void printPeList(const char* note, int *peList);

    
    bool shouldBufferMessagesNow(int instid);
    void sendBufferedMessages(int instid, int step=-1);
    void sendBufferedMessagesAllStrategies();

 public:

    ComlibLocalStats clib_stats;   //To store statistics of
                                   //communication operations
    
    ComlibManager();  //Recommended constructor

    ComlibManager(CkMigrateMessage *m) { }
    int useDefCtor(void){ return 1; } //Use default constructor should
    //be pupped and store all the strategies.

    /* Initialization routines */
    
    void barrier(void);
    void resumeFromSetupBarrier();
   
    /* The delegation framework reimplemented functions */

    void ArraySend(CkDelegateData *pd,int ep, void *msg, 
                   const CkArrayIndex &idx, CkArrayID a);
    void GroupSend(CkDelegateData *pd, int ep, void *msg, int onpe, 
                   CkGroupID gid);
    void ArrayBroadcast(CkDelegateData *pd,int ep,void *m,CkArrayID a);
    void GroupBroadcast(CkDelegateData *pd,int ep,void *m,CkGroupID g);
    void ArraySectionSend(CkDelegateData *pd, int ep ,void *m, 
                                  int nsid, CkSectionID *s, int opts);
    CkDelegateData* ckCopyDelegateData(CkDelegateData *data); 
    CkDelegateData *DelegatePointerPup(PUP::er &p,CkDelegateData *pd);

    inline Strategy *getStrategy(int instid)
        {return converseManager->getStrategy(instid);}

    inline StrategyTableEntry *getStrategyTableEntry(int instid)
        {return converseManager->getStrategyTable(instid);}

    // Entry methods used by bracketed strategies when there is some object
    // migration. The comlib Manager realizes there is an error, it enters error
    // mode by suspending the delivery of the strategy if it has already
    // started, and it globally reduces the number of elements which already
    // called endIteration. When this number matches the number of elements
    // involved in the operation (or a multiple of it in case the user overlaps
    // iterations), the system ri-deliver the doneInserting to the strategy.
    //  void bracketedFinishSetup(int instid);
    void bracketedCatchUpPE(int instid, int step);
    void bracketedReceiveCount(int instid, int pe, int count, int isFirst, int step);
    void bracketedStartErrorRecoveryProcess(int instid, int step);
    void bracketedErrorDetected(int instid, int step);
    void bracketedConfirmCount(int instid, int step);
    void bracketedCountConfirmed(int instid, int count, int step);
    void bracketedReceiveNewCount(int instid, int step);
    void bracketedReceiveNewPeList(int instid, int step, int *count);

    void bracketedFinalBarrier(int instid, int step);
    void bracketedReleaseCount(int instid, int step);
    void bracketedReleaseBufferedMessages(int instid, int step);

    void bracketedStartDiscovery(int instid);
    void bracketedDiscover(int instid, CkArrayID aid, CkArrayIndex &idx, int isSrc);
    void bracketedContributeDiscovery(int instid, int pe, int nsrc, int ndest, int step);


    // TODO: Delete the following two methods!!!!
    void AtSync();           //User program called loadbalancer
    void lbUpdate(LBMigrateMsg *); //loadbalancing updates

    //Learning functions
    //void learnPattern(int totalMessageCount, int totalBytes);
    //void switchStrategy(int strat);

    //void collectStats(ComlibLocalStats &s, int src,CkVec<ClibGlobalArrayIndex>&);
    void collectStats(ComlibLocalStats &s, int src);
 

    /// Print information about the number of undelivered messages awaiting a specified strategy to be ready    
    void printDiagnostics(int cinst);

    /// Print information about the number of undelivered messages awaiting any strategy to be ready
    void printDiagnostics();   
       
}; 

/** This class is used by ComlibManager (the delegator manager) as its delegated
    data. The only thing it contains is the position in the system of the
    strategy it represents.
 */
class ComlibDelegateData : public CkDelegateData {
 private:
  int _instid; ///< Position of this instance in the strategy table

  friend void ComlibAssociateProxy(ComlibInstanceHandle, CProxy&);
  friend CkDelegateData *ComlibManager::ckCopyDelegateData(CkDelegateData*);
  ComlibDelegateData(int instid);

 public:
  ComlibDelegateData(CkMigrateMessage *) : CkDelegateData() { ref(); }
  
  void beginIteration();
  void endIteration();

  /// Get the position of this instance in the strategy table
  inline int getID() {return _instid;}
  
  void pup(PUP::er &p) {
    p | _instid;
  }

};

void ComlibAtSync(void *msg);
void ComlibNotifyMigrationDoneHandler(void *msg);
void ComlibLBMigrationUpdate(LBMigrateMsg *);


#ifdef filippo
// The old interface
#define RECORD_SENDSTATS(sid, bytes, dest) {             \
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

#else
// the new version of comlib does not do anything with these yet
#define RECORD_SEND_STATS(sid, bytes, dest) /* empty */
#define RECORD_RECV_STATS(sid, bytes, src) /* empty */
#define RECORD_SENDM_STATS(sid, bytes, dest_arr, ndest) /* empty */
#define RECORD_RECVM_STATS(sid, bytes, src_arr, nsrc) /* empty */
#endif

/** @} */

#endif
