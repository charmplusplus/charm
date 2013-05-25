

/**
   @addtogroup ConvComlib
   @{
   @file
   
   @brief Converse Strategy This file contains the main declarations for
   converse strategies. This include the classes Strategy, MessageHolder and
   StrategyWrapper.
   
   This file is the lowest in the inclusion chain of comlib, so it is included
   in all other pars of comlib.
   
   Heavily revised, Filippo Gioachin 01/06
*/

#ifndef CONVCOMLIBSTRATEGY_H
#define CONVCOMLIBSTRATEGY_H

#if 0
// FILIPPO: TEMPORARY HACK
#ifndef COMLIBSTRATEGY_H
#undef CkpvDeclare
#undef CkpvAccess
#undef CkpvExtern
#undef CkpvInitialize
#undef CkpvInitialized
#define CkpvDeclare CpvDeclare
#define CkpvAccess CpvAccess
#define CkpvExtern CpvExtern
#define CkpvInitialize CpvInitialize
#define CkpvInitialized CpvInitialized
#define CkMyPe CmiMyPe
#define CkNumPes CmiNumPes
#undef CkRegisterHandler
#define CkRegisterHandler(x) CmiRegisterHandler(x)
#endif
#endif




#include "converse.h"
#include "pup.h"
#include "cklists.h"
#include <set>

/***************************************************************
 * Definitions used by the strategies as constants plus information to
 * debug comlib.
 * Copied from old convcomlib.h - Filippo
 ****************************************************************/
extern int com_debug;
#ifdef CMK_OPTIMIZE
inline void ComlibPrintf(...) {}
#else
#define ComlibPrintf if(com_debug) CmiPrintf
//#define ComlibPrintf  CmiPrintf
#endif

#define USE_TREE 1            //Organizes the all to all as a tree
#define USE_MESH 2            //Virtual topology is a mesh here
#define USE_HYPERCUBE 3       //Virtual topology is a hypercube
#define USE_DIRECT 4          //A dummy strategy that directly forwards 
                              //messages without any processing.
#define USE_GRID 5            //Virtual topology is a 3d grid
#define USE_LINEAR 6          //Virtual topology is a linear array
#define USE_PREFIX 7	      //Prefix router to avoid contention

#define IS_BROADCAST -1
#define IS_SECTION_MULTICAST -2

/** This class is a handle for a strategy, used and passed by the user to comlib
    funcions that need it. Whenever necessary, the system can create additional
    handles used for specific purposes starting from an instance of this class.
    At the moment it is simply a typedef to an int.
 */
typedef int ComlibInstanceHandle;


/** An abstract data structure that holds a converse message and which can be
    buffered by the communication library Message holder is a wrapper around a
    message. Has other useful data like destination processor list for a
    multicast etc. */

class MessageHolder : public PUP::able {
 public:
    char *data;
    int size;
    int isDummy;
    
    //For multicast, the user can pass the pelist and list of Pes he
    //wants to send the data to.

    /** npes=0 means broadcast, npes=1 means one destination specified by
	dest_proc, npes>1 means multicast with destinations specified in the
	array pelist */
    int npes;
    union {
      int *pelist;
      int dest_proc;
    };
    
    MessageHolder() 
        {dest_proc = size = isDummy = 0; data = NULL;}    

    MessageHolder(CkMigrateMessage *m) {}

    /// Single destination constructor
    inline MessageHolder(char * msg, int sz, int proc) {
        data = msg;
        size = sz;

        npes = 1;
        dest_proc = proc;
        
        isDummy = 0;
    }

    /// Broadcast constructor
    inline MessageHolder(char * msg, int sz) {
        data = msg;
        size = sz;

        npes = 0;
        dest_proc = 0;
        
        isDummy = 0;
    }

    /// Multicast constructor
    inline MessageHolder(char * msg, int sz, int np, int *pes) {
        data = msg;
        size = sz;

        npes = np;
        pelist = pes;
        
        isDummy = 0;
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

// These should not be here by in Charm!!
#define CONVERSE_STRATEGY 0     //The strategy works for converse programs
#define NODEGROUP_STRATEGY 1    //Node group level optimizations 
#define GROUP_STRATEGY 2        //Charm Processor level optimizations
#define ARRAY_STRATEGY 3        //Array level optimizations

/** Class that defines the entry methods that a Converse level strategy must
    define. To write a new strategy inherit from this class and define the
    virtual methods. Every strategy can also define its own constructor and
    have any number of arguments. */

class Strategy : public PUP::able {
 protected:


    short type;
    /// 1 if the strategy is bracketed, 0 otherwise
    short isStrategyBracketed;
    //int myInstanceID; DEPRECATED in favor of myHandle / A handle used by the
    //user to identify this strategy. This is a wrapper to the real handle which
    //is virtual.
    ComlibInstanceHandle myHandle;
    /* Used for pure converse strategies which need to deliver their message
	directly to a converse handler (and not to higher languages). This will
	be used as final deliverer. */
    // DEPRECATED! The destination should be read from the header of the envelope
    //int destinationHandler;

    inline int getInstance() {return myHandle;}

 public:
    Strategy();
    Strategy(CkMigrateMessage *m) : PUP::able(m) {
      // By default assume that the class is self contained, if a higher
      // language class desires, it can change the following two fields.
      //converseStrategy = this;
      //higherLevel = this;
    }

    // Is the knowledge of bracketized necessary? Currently it is used only to
    // register the strategy to the ComlibArrayListener...
    inline void setBracketed(){isStrategyBracketed = 1;}
    inline int isBracketed(){return isStrategyBracketed;}

    //virtual void bracketedCountingError() {}

    /// This function is called to update the knowledge hold by a strategy which
    /// cares about migratable objects. count is of size CkNumPes() and has the
    /// following format:
    /// 0 if proc "i" has no objects,
    /// 1 if proc "i" has only source objects,
    /// 2 if proc "i" has only destination objects,
    /// 3 if proc "i" has both source and destination objects
    virtual void bracketedUpdatePeKnowledge(int *count) {}

    /// Called for each message
    virtual void insertMessage(MessageHolder *msg)=0;// {}
    
    /** Called after all messages have been deposited in this processor. This
	corresponds to a call to ComlibEnd(cinst), where cinst is the
	ComlibInstanceHandle returned when registering the Strategy with Comlib.
	In higher levels this may need many ComlibEnd calls before invoking
	doneInserting. */
    virtual void doneInserting() {}

    //inline void setInstance(int instid){myInstanceID = instid;}
    //inline int getInstance(){return myInstanceID;}

    inline int getType() {return type;}
    // TO DEPRECATE! we shouldn't change the type after creation
    inline void setType(int t) {type = t;}

    /// Return a handle to this strategy
    ComlibInstanceHandle getHandle() {return myHandle;}

    /** Called when a message is received in the strategy handler */
    virtual void handleMessage(void *msg)=0;// {}
    
    /** This method can be used to deliver a message through the correct class */
    virtual void deliver(char* msg, int size) {
      CmiAbort("Strategy::deliverer: If used, should be first redefined\n");
    };

    /** Called when a subsystem scheme (like in bracketed EachToMany) terminates
	the requested routing operation */
    virtual void notifyDone() {}

    /** Called on processor 0 after the strategy has been packed for
	propagation. In this way the strategy can leave some portion not fully
	initialized and initialize it after it has been packed. This gives the
	possibility to propagate some data everywhere before doing processor
	specific handling. */
    virtual void finalizeCreation() {}

    /** Each strategy must define his own Pup interface. */
    virtual void pup(PUP::er &p);
    //PUPable_decl(Strategy);
    PUPable_abstract(Strategy);
};

/** Enables a list of strategies to be stored in a message through the PUPable
    framework. */
class StrategyWrapper {
 public:
  Strategy **strategy;
  int *position;
  bool *replace;
  int nstrats;
  //int total_nstrats;

  StrategyWrapper() : nstrats(0), position(NULL), strategy(NULL) {}
  StrategyWrapper(int count);
  ~StrategyWrapper();
  void pup(PUP::er &p);
};



/// Definition of error modes that are used when detecting migration of elements for a strategy
enum CountingErrorMode { NORMAL_MODE = 0, 
			 ERROR_MODE = 1, 
			 CONFIRM_MODE = 2,   
			 ERROR_FIXED_MODE = 3  };


enum CountingServerErrorMode { NORMAL_MODE_SERVER, 
			       ERROR_MODE_SERVER,
			       CONFIRM_MODE_SERVER,  
			       ERROR_FIXED_MODE_SERVER,
			       NON_SERVER_MODE_SERVER    };


enum DiscoveryErrorMode { NORMAL_DISCOVERY_MODE = 200, 
			  STARTED_DISCOVERY_MODE = 201, 
			  FINISHED_DISCOVERY_MODE = 202   };

/** Information about each instance of a strategy.
 
    Each StrategyTableEntry points to a strategy, as well as containing 
    information about which state the strategy instance is in.

    Strategies can change during the execution of the program but the
    StrategyTableEntry stores some persistent information for the
    strategy. The communication library on receiving a message, calls
    the strategy in this table given by the strategy id in the message.
    
    With the philosophy that strategies are not recreated, but only updated
	during the execution of the program, this struct should be useless?
*/

class StrategyTableEntry {
public:
  Strategy *strategy;

  /// A flag to specify if the strategy is active or suspended
  int isReady;
  /// A flag to specify if the strategy is newly created and not yet synchronized
  int isNew;
  /// A flag to specify if the strategy is currently being synchronized
  int isInSync;

  
  /** 
   A flag that specifies whether outgoing messages should be buffered, 
   currently only used in the bracketed strategies when an error has been 
   detected. Once the error has been corrected, no longer are the outgoing messages 
   buffered.
   
   errorDelaySendBuffer (see below) is the name of the queue of messages buffered 
   while this variable is 1.
   
   Currently the following methods respect the wishes of this variable: 
   		ComlibManager::multicast()
    
  */
  int bufferOutgoing;
  
  
  /// A flag to specify that the bracketed strategy is fully setup, and it can operate normally.
  /// A value of 0 means strategy is fully setup
  int bracketedSetupFinished;

  /// A buffer for all the messages to this strategy while the strategy is suspended
  CkQ<MessageHolder*> tmplist;

  int numBufferReleaseReady;
  
  int numElements;   //< Count how many src(?) elements reside here

  // 	Used to ensure strategies work in presence of migration
  int nBeginItr;     //< #elements that called begin iteration
  int nEndItr;       //< #elements that called end iteration
  int call_doneInserting; //< All elements deposited their data


  int lastKnownIteration;

  // values used during bracketed error/confirm mode
  int nEndSaved;        //< during error mode, number sent to processor 0
  int totalEndCounted;  //< on processor 0, total amount of endIterations counted
  int nProcSync;         //< on processor 0, number of processors already in the count

  // values used during the discovery process to count the number of objects
  // valid only on processor 0

  /// list of where objects are, will be broadcasted at the end, the list is
  /// CkNumPes() elements long + 2. These two elements are: peList[CkNumPes()]
  /// number of source objects discovered, peList[CkNumPes()+1] number of
  /// destination objects discovered. This trick eliminates two integer
  /// variables
  int *peList;
  
  
  /// The number of PEs that have confirmed the count to ComlibManager::bracketedCountConfirmed
  int peConfirmCounter;
  /// This was formerly a static variable in ComlibManager::bracketedCountConfirmed()
  int total; 
 
  
  void reset();
  


  StrategyTableEntry();



 private:
  /** 
   A flag to specify that this bracketed strategy is in an error mode,
   due to an object migration causing more objects to call BeginIteration 
   on one of the PEs than was expected. The detecting PE immediately enter the error
   mode state, while other PEs will be notified of the error, and they will then 
   enter the error state.
  */
  CountingErrorMode errorMode;
  
  /** 
   The state of the coordinator (PE 0 of the chare group).
  */
  CountingServerErrorMode errorModeServer;
  
  
  /** 
      A flag to specify what stage of Discovery process is underway for the PE 
  */
  DiscoveryErrorMode discoveryMode;


 public:

  CountingErrorMode getErrorMode(){return errorMode;}
  CountingServerErrorMode getErrorModeServer(){return errorModeServer;}
  DiscoveryErrorMode getDiscoveryMode(){return discoveryMode;}
   
  void setErrorMode(CountingErrorMode mode){
    errorMode=mode;
    ComlibPrintf("[%d] %s\n", CmiMyPe(), errorModeString());
  }
 
  void setErrorModeServer(CountingServerErrorMode mode){
    errorModeServer=mode;
    ComlibPrintf("[%d] %s lastKnownIteration=%d\n", CmiMyPe(), errorModeServerString(), lastKnownIteration);
  }
 
  void setDiscoveryMode(DiscoveryErrorMode mode){
    discoveryMode=mode;
    ComlibPrintf("[%d] %s\n", CmiMyPe(), discoveryModeString());
  }

  const char *errorModeString();
  const char *errorModeServerString();
  const char *discoveryModeString();  

};

typedef CkVec<StrategyTableEntry> StrategyTable;


#endif

/*@}*/
