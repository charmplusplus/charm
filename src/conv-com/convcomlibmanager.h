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

class ConvComlibManager {

    StrategyTable strategyTable;
    int nstrats;

 public:

    ConvComlibManager();
    void insertStrategy(Strategy *s);
    void insertStrategy(Strategy *s, int loc);
    Strategy * getStrategy(int loc) {return strategyTable[loc].strategy;}
    StrategyTable *getStrategyTable() {return &strategyTable;}
};


void initComlibManager();
Strategy *ConvComlibGetStrategy(int loc);
void ConvComlibRegisterStrategy(Strategy *s);

#endif
