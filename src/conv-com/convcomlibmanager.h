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
#include "comlib.h"
#include <convcomlibstrategy.h>

#define MAX_NUM_STRATS 128

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

#endif
