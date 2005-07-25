/* Converse ComlibManager 
   Enables communication library strategies to be called from converse code.
   Reused by the Charm Comlibmanager.
   
   Stores a strategy table. Strategies can be inserted and accessed
   from this table.

   Sameer Kumar 28/03/04.
*/

#ifndef CONVCOMLIBMANAGER
#define CONVCOMLIBMANAGER

#include <converse.h>
#include "convcomlib.h"
#include <convcomlibstrategy.h>

#define MAX_NUM_STRATS 32

class ConvComlibManager {
    
    StrategyTable strategyTable;
    CmiBool init_flag;

 public:
    int nstrats;

    ConvComlibManager();
    void setInitialized() {init_flag = CmiTrue;}
    CmiBool getInitialized() {return init_flag;}
    void insertStrategy(Strategy *s);
    void insertStrategy(Strategy *s, int loc);
    Strategy * getStrategy(int loc) {return strategyTable[loc].strategy;}
    StrategyTable *getStrategyTable() {return &strategyTable;}
};


void initComlibManager();
Strategy *ConvComlibGetStrategy(int loc);
void ConvComlibRegisterStrategy(Strategy *s);
void ConvComlibScheduleDoneInserting(int loc);

CkpvExtern(int, strategy_handlerid);

//Send a converse message to a remote strategy instance. On being
//received the handleMessage method will be invoked.
inline void ConvComlibSendMessage(int instance, int dest_pe, int size, char *msg) {
    CmiSetHandler(msg, CkpvAccess(strategy_handlerid));
    ((CmiMsgHeaderBasic *) msg)->stratid = instance;
    
    CmiSyncSendAndFree(dest_pe, size, msg);
}

#endif
