
#ifndef AAPLEARNER_H
#define AAPLEARNER_H

#include "ComlibManager.h"
#include "ComlibLearner.h"

#define ALPHA 1e-5
#define BETA  7.8e-9

#define min(x,y) ((x < y) ? x : y)

inline double min4(double x, double y, double a, double b) {
    double x1 = min(x,y);
    double a1 = min(a,b);
    
    return min(x1,a1);
} 

class AAPLearner : public ComlibLearner {
    double alpha, beta;

    double computeDirect(double P, double m, double d);
    double computeMesh(double P, double m, double d);
    double computeHypercube(double P, double m, double d);
    double computeGrid(double P, double m, double d);

 public:
    AAPLearner();    

    void init();
    Strategy* optimizePattern(Strategy* , ComlibGlobalStats &);
    
    Strategy ** optimizePattern(Strategy** , ComlibGlobalStats &) {
        CkAbort("Not implemented\n");
    }
};


#endif
