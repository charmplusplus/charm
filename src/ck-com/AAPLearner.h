/**
   @addtogroup CharmComlib
   @{
*/

#ifndef AAPLEARNER_H
#define AAPLEARNER_H

#include "convcomlib.h"
#include "ComlibManager.h"
#include "ComlibLearner.h"

#define ALPHA       1e-5    //Total alpha overhead
#define ALPHA_NIC1  9e-6  //NIC alpha for short messages
#define ALPHA_NIC2  6e-6  //NIC alpha

#define ALPHA_CHARM 2e-6  //Charm++ processing after message has been received
                             //Includes malloc and scheduling overheads

#define GAMMA_NIC   2.6e-9   //DMA bandwidth to copy data into NIC memory
#define BETA        4e-9   //Network bandwidth
#define GAMMA_MEM       9e-10     //Memory bandwidth (copied twice)

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
        return NULL;
    }
};


#endif

/*@}*/
