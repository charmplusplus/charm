/**
   @addtogroup ComlibConverseStrategy
   @{
   @file 
   @brief Hypercube topology
*/

#include "hypercubetopology.h"

inline int neighbor(int pe, int dim)
{
    return(pe ^ (1<<dim));
}

inline int maxdim(int n)
{
    int maxpes=1, dim=0;

    while (maxpes< n) {
        maxpes *=2;
        dim++;
    }
    if (maxpes==n) return(dim);
    else return(dim-1);
}

inline int adjust(int dim, int pe)
{
    int mymax=1<<dim;
    if (pe >= mymax) return(neighbor(pe, dim));
    else return(pe);
}

HypercubeTopology::HypercubeTopology(int npes, int mype)
  : NumPes(npes), MyPe(mype), Dim(maxdim(npes)), penum(NumPes, 0)
{
    int i = 0;

    next = new int *[Dim];
    for (i=0;i<Dim;i++) {
        next[i]=new int[NumPes];
        for (int j=0;j<NumPes;j++) next[i][j]=-1;
    }
    
    //Create and initialize the indexes to the above table
    std::vector<int> dp(NumPes);
    for (i=0;i<NumPes;i++) {
        dp[i]=i;
    }
    
    CreateStageTable(NumPes, &dp[0]);
}

void HypercubeTopology::getNeighbors(int &np, int *pelist){
    np = Dim;

    for(int count = 0; count < Dim; count ++)
        pelist[count] = MyPe ^ (1 << (Dim - count - 1));
}

int HypercubeTopology::getNumStages(){
    return Dim;
}

int HypercubeTopology::getNumSteps(int stage) {
    return 1;
}

void HypercubeTopology::getPesToSend(int step, int stage, int &np, 
                                     int *pelist, int &nextpe) {
    if(step > 0) {
        np = 0;
        return;
    }

    np = penum[Dim - stage - 1];
    memcpy(pelist, next[Dim - stage - 1], np *sizeof(int));
        
    nextpe = neighbor(MyPe, Dim - stage - 1);
}

int HypercubeTopology::getNumMessagesExpected(int stage) {
    return 1;
}

void HypercubeTopology::CreateStageTable(int numpes, int *destpes)
{
    std::vector<int> dir(numpes);
    int nextdim, j, i;
    for (i=0;i<numpes;i++) {
        dir[i]=MyPe ^ adjust(Dim, destpes[i]);
    }
    
    for (nextdim=Dim-1; nextdim>=0; nextdim--) {
        int mask=1<<nextdim;
        for (i=0;i<numpes;i++) {
            if (dir[i] & mask) {
                dir[i]=0;
                for (j=0;(j<penum[nextdim]) && (destpes[i]!=next[nextdim][j]);j++);
                if (destpes[i]==next[nextdim][j]) { 
                    //CmiPrintf("EQUAL %d\n", destpes[i]);
                    continue;
                }
                next[nextdim][penum[nextdim]]=destpes[i];
                penum[nextdim]+=1;
                //CmiPrintf("%d next[%d][%d]=%d\n",MyPe, nextdim, penum[nextdim],destpes[i]);
            }
        }
    }
}
/*@}*/
