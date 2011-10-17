#ifndef CONVCOMLIBMANAGER
#define CONVCOMLIBMANAGER


/**
   @addtogroup ConvComlib
   *@{
   
   @file
   @brief Converse ComlibManager.
   Declares the classes ConvComlibManager, plus
   the interface provided to the user.
   
   * ConvComlibManager, responsible for the management of all the strategies in
   the system, their initial broadcast and synchronization (at strategy
   instantiation), as well as some runtime coordination. Mostly gives support
   for the higher classes.
    
   The interface provided to the user es prefixed with "Comlib" except
   ConvComlibRegister which has an extra prefix for compatibility with charm.

   Sameer Kumar 28/03/04.
   Heavily revised, Filippo Gioachin 01/06.
*/

#include <convcomlibstrategy.h>
#include "ComlibStrategy.h"

// Are we crazy? Only 32 strategies allowed in the entire system?!?!?
#define MAX_NUM_STRATS 32


#define STARTUP_ITERATION -10000


/** Converse interface to handle all the Comlib strategies registered in the
    Converse of Charm++ program. Being in Converse, this is a pure class
    allocated into a global variable. */
class ConvComlibManager {

  /// a table containing all the strategies in the system
  StrategyTable strategyTable;
  /// it is true after the system has been initialized at the program startup
  CmiBool init_flag;
  /// the number of strategies currently present in the system
  int nstrats;

  friend class Strategy;
  int insertStrategy(Strategy *s);
 public:
  // a few variables used in the process of strategy synchronization
  /// count how many acks have been received for the current synchronization
  int acksReceived;
  /// used when a doneCreating is called before the previous one is finished
  CmiBool doneCreatingScheduled;
  /// true when the manager is busy synchronizing the strategies
  CmiBool busy;

  ConvComlibManager();

  void setInitialized() {init_flag = CmiTrue;}
  CmiBool getInitialized() {return init_flag;}

  // used only by ComlibManager.C to broadcast the strategies: DEPRECATED!
  // the broadcast has to be done by converse
  //void insertStrategy(Strategy *s, int loc);

  /** Used to broadcast all strategies to all processors after inserting them */
  void doneCreating();
  /** Switch all the strategies in the "inSync" status to "ready" and deposit
      into the strategies all pending messages */
  void tableReady();
  void enableStrategy(int loc);

  // private:  //Why do we need private here?? XLC breaks with this private decl
  inline void setStrategy(int loc, Strategy *s) {
#ifndef CMK_OPTIMIZE
    if (loc == 0) CmiAbort("Trying to set strategy zero, not valid!\n");
#endif
    strategyTable[loc].strategy = s;
  }
  friend void *comlibReceiveTableHandler(void *msg);
 public:
  inline int getNumStrats() {return nstrats;}
  inline void incrementNumStrats() {nstrats++;}
  inline void decrementNumStrats() {nstrats--;}
  inline Strategy *getStrategy(int loc) {return strategyTable[loc].strategy;}
  inline int isReady(int loc) {return strategyTable[loc].isReady;}
  inline int isNew(int loc) {return strategyTable[loc].isNew;}
  inline int isInSync(int loc) {return strategyTable[loc].isInSync;}
  inline void inSync(int loc) {
    strategyTable[loc].isNew = 0;
    strategyTable[loc].isInSync = 1;
  }
/*   inline int getErrorMode(int loc) {return strategyTable[loc].errorMode;} */
  inline CkQ<MessageHolder*> *getTmpList(int loc) {return &strategyTable[loc].tmplist;}

  
  // TODO inline this function again
  void insertMessage(MessageHolder* msg, int instid);
  
  inline void doneInserting(int loc) {
	  if (isReady(loc))
		  strategyTable[loc].strategy->doneInserting();
	  else
		  strategyTable[loc].call_doneInserting ++;
  }

  StrategyTableEntry *getStrategyTable(int loc) {return &strategyTable[loc];}
  //StrategyTable *getStrategyTable() {return &strategyTable;}

  void printDiagnostics();

};

/// This processor's converse Comlib manager
CkpvExtern(ConvComlibManager, conv_com_object);


void initConvComlibManager();


/***************************************************************************
 * User section:
 *
 * Implementation of the functions used by the user
 ***************************************************************************/

Strategy *ConvComlibGetStrategy(int loc);


/** Iterate over all the inserted strategies and broadcast all those that are
    marked as new. This has some effect only on processor 0, for all the other
    it is a NOP. */
inline void ComlibDoneCreating() {
  if (CmiMyPe() != 0) return;
  CkpvAccess(conv_com_object).doneCreating();
}

void ConvComlibScheduleDoneInserting(int loc);

/** Converse send utilities for comlib. For all of them the message passed will
    be lost by the application. */
inline void ComlibSyncSendAndFree(unsigned int pe, unsigned int size, char *msg, ComlibInstanceHandle cinst) {
  CkpvAccess(conv_com_object).insertMessage(new MessageHolder(msg, size, pe), cinst);
}

inline void ComlibSyncBroadcastAllAndFree(unsigned int size, char *msg, ComlibInstanceHandle cinst) {
  CkpvAccess(conv_com_object).insertMessage(new MessageHolder(msg, size), cinst);
}

/// In this case the array "pes" is lost by the application together with "msg".
inline void ComlibSyncListSendAndFree(int npes, int *pes, unsigned int size, char *msg, ComlibInstanceHandle cinst) {
  CkpvAccess(conv_com_object).insertMessage(new MessageHolder(msg, size, npes, pes), cinst);
}

inline void ComlibBegin(ComlibInstanceHandle cinst, int iteration) {
  // Do nothing?
}

inline void ComlibEnd(ComlibInstanceHandle cinst, int iteration) {
  // flush strategy???
  CkpvAccess(conv_com_object).doneInserting(cinst);
}

CkpvExtern(int, comlib_handler);

/* DEPRECATED!!! gained the syntax above similar to Cmi
// Send a converse message to a remote strategy instance. On being
// received the handleMessage method will be invoked.
inline void ConvComlibSendMessage(int instance, int dest_pe, int size, char *msg) {
  CmiSetHandler(msg, CkpvAccess(strategy_handlerid));
  ((CmiMsgHeaderExt *) msg)->stratid = instance;
    
  CmiSyncSendAndFree(dest_pe, size, msg);
}*/

/*@}*/

#endif
