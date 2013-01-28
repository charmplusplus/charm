/**
   @addtogroup ConvComlib
   @{
   @file 
   @brief Header for Hypercube topology discriptor 
*/

#ifndef __HYPERCUBE_TOPOLOGY_H
#define __HYPERCUBE_TOPOLOGY_H

#include "graphrouter.h"
#include "string.h"

#include <vector>

class HypercubeTopology: public TopologyDescriptor {
    int NumPes, MyPe, Dim;
    std::vector<int> penum;
    int **next;

    void CreateStageTable(int, int *);
    
 public:
    HypercubeTopology(int npes, int mype);
    //Entry methods which will define the specific graph.
    void getNeighbors(int &npes, int *pelist);
    int getNumStages();
    int getNumSteps(int stage);
    void getPesToSend(int step, int stage, int &npesToSend, 
                              int *pelist, int &nextpe);
    int getNumMessagesExpected(int stage);
};

#endif

/*@}*/
