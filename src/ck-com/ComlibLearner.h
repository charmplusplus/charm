#ifndef COMLIBLEARNER_H
#define COMLIBLEARNER_H

#include "convcomlibstrategy.h"


/* Communication library learner which takes a strategy or a list of
   strategies as input along with the communication pattern of the
   objects belonging to those strategies and returns new strategies to
   replace the input strategies. These new strategies optimize the
   communication pattern. */

class ComlibGlobalStats;
class ComlibLearner {
 public:
    virtual ~ComlibLearner() {}
    //Configures parameters of the learner. Will be called by the
    //communication library on every processor after the second
    //barrier of the communication library.
    virtual void init() {}
    
    //Optimizes a specific strategy. Returns a new optimized strategy
    virtual Strategy* optimizePattern(Strategy *strat, 
                                      ComlibGlobalStats &sdata){
        return NULL;
    }
    
    //Optimizes the communication pattern of a group of strategies
    //together
    virtual Strategy** optimizePattern(Strategy **strat, 
                                       ComlibGlobalStats &sdata){
        return NULL;
    }
};

#endif
