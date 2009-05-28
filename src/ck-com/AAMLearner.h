/**
   @addtogroup CharmComlib
   @{
*/



#ifndef AAMLEARNER_H
#define AAMLEARNER_H

#include "ComlibManager.h"
#include "ComlibLearner.h"
#include "AAPLearner.h"

#define GAMMA 2e-9

class AAMLearner : public ComlibLearner {
    //alpha network and cpu s/w overhead
    //beta network transmission time
    //gamma memory copy overhead
    double alpha, beta, gamma;   

    double computeDirect(double P, double m, double d);
    double computeMesh(double P, double m, double d);
    double computeHypercube(double P, double m, double d);
    double computeGrid(double P, double m, double d);

 public:
    AAMLearner();    

    void init();
    Strategy* optimizePattern(Strategy* , ComlibGlobalStats &);
    
    Strategy ** optimizePattern(Strategy** , ComlibGlobalStats &) {
        CkAbort("Not implemented\n");
        return NULL;
    }
};


#endif
/*@}*/
