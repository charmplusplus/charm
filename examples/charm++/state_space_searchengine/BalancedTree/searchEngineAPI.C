#ifndef __SEARCHENGINEAPI__
#define __SEARCHENGINEAPI__

#include "searchEngine.h"

/*   framework for search engine */


extern int branchfactor;
extern int depth;
extern int initial_grainsize;
extern int target;

class BTreeStateBase : public StateBase
{
public:
   int depth;
   long long index;
};

void createInitialChildren(Solver *solver)
{
    BTreeStateBase *root = (BTreeStateBase*)solver->registerRootState(sizeof(BTreeStateBase), 0, 1);
    root->index = 0;
    root->depth = 0;
    solver->process(root);
}

inline void createChildren( StateBase *_base , Solver* solver, bool parallel)
{
    BTreeStateBase base = *((BTreeStateBase*)_base);
    long long t = 1;

    CkPrintf(" Processed :thisindex=%lld, depth=%d\n", base.index, base.depth);
    for(int i=0; i<target; i++)
         t = t << 1;
    for(int childIndex=0; childIndex<branchfactor; childIndex++)
    {
        long long thisindex = base.index * branchfactor + childIndex;
        if(base.depth == depth-1)
        {
            if( t-1 == thisindex)
                solver->reportSolution();
        }
        else{
            BTreeStateBase *child  = (BTreeStateBase*)solver->registerState(sizeof(BTreeStateBase), childIndex, branchfactor);
            child->depth = base.depth + 1;
            child->index = base.index * branchfactor + childIndex; 
            if(parallel) {
                solver->process(child);
            }
        }
    }
}

int parallelLevel()
{
    return initial_grainsize;
}

int searchDepthLimit()
{
    return 1;
}

SE_Register(BTreeStateBase, createInitialChildren, createChildren, parallelLevel, searchDepthLimit);

#endif
