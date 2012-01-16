/* searchEngine.h
 *
 *  Nov 3
 *
 * author: Yanhua Sun
 */
#ifndef __SEARCHENGINE__
#define __SEARCHENGINE__

#include "charm++.h"
#include "cmipool.h"

#define ADAPTIVE     1

class Solver;
class StateBase;
class SearchNodeMsg;
class CProxy_SearchConductor;

extern CProxy_SearchConductor searchEngineProxy;

typedef void (*SE_createInitialChildrenFn)(Solver *solver);
typedef void (*SE_createChildrenFn)( StateBase *_base , Solver* solver, bool parallel);
typedef int (*SE_parallelLevelFn)();
typedef int (*SE_searchDepthLimitFn)();
typedef double (*SE_lowerBoundFn)(StateBase *_base);

extern SE_lowerBoundFn   _lowerBoundFn;

void SE_register(SE_createInitialChildrenFn  f1,
                 SE_createChildrenFn  f2,
                 SE_parallelLevelFn   f3,
                 SE_searchDepthLimitFn  f4,
                 SE_lowerBoundFn f5 = NULL
                 );

class StateBase
{
};

class Solver {
protected:
    UShort parentBits;
    unsigned int* parentPtr;
    int searchLevel;
public:
    //Solver(): searchLevel(0) {}
    virtual StateBase *registerRootState(size_t size, unsigned int childnum, unsigned int totalNumChildren)=0;
    virtual StateBase *registerState(size_t size, unsigned int childnum, unsigned int totalNumChildren) = 0;
    virtual void process(StateBase *state)=0;
    virtual void deleteState(StateBase *state) = 0;
    virtual void setParentInfo(SearchNodeMsg *msg, int l) = 0;
    virtual inline void reportSolution();
    virtual void setPriority(StateBase *state, int p)=0;
#ifdef BRANCHBOUND
    inline void updateCost(double c);
#endif
};

#include "searchEngine_impl.h"

#define MASK  0XFF 
#define ENTRYGRAIN  0.002   //2ms
extern void registerSE();
extern int se_statesize;


#ifdef BRANCHBOUND
#if ! ADAPTIVE
#define SE_Register(state, f1, f2, f3, f4, f5)  \
    void registerSE() {    \
      SE_register(f1, f2, f3, f4, f5);   \
    }  \
     void createMultipleChildren(SearchGroup* myGroup, StateBase *parent, SequentialSolver* solver, bool parallel) {  \
       StateBase *state;  \
       double minCost = myGroup->getCost(); \
       double lb = f5(parent); \
        if(lb<minCost)  \
            f2(parent, solver, false);  \
        while((state=solver->dequeue()) != NULL) \
        {  \
            minCost = myGroup->getCost(); \
            lb = f5(parent); \
            if(lb>=minCost)  \
            continue;  \
            f2(state, solver, parallel); \
        }  \
    }
#else
#define SE_Register(state, f1, f2, f3, f4, f5)  \
    void registerSE() {    \
      SE_register(f1, f2, f3, f4, f5);   \
    }  \
    void createMultipleChildren(SearchGroup* myGroup, StateBase *parent, SequentialSolver* solver, bool parallel) {  \
       StateBase *_state;  \
       double avgentrytime = 0;  \
       int processed_nodes = 1; \
       double instrument_start; \
       double accumulate_time = 0; \
       double minCost = myGroup->getCost(); \
       double lb = f5(parent); \
       instrument_start = CkWallTimer();  \
        if(lb<minCost)  \
            f2(parent, solver, false);  \
        accumulate_time = avgentrytime  = CkWallTimer() - instrument_start;  \
        while((_state=solver->dequeue()) != NULL) \
        {  \
            minCost = myGroup->getCost(); \
            lb = f5(parent); \
            if(lb>=minCost)  \
            continue;  \
            if(processed_nodes  == 20)  \
            {  \
                avgentrytime  = (CkWallTimer() - instrument_start)/20;  \
            }  \
            f2(_state, solver, parallel); \
            accumulate_time += avgentrytime; \
            if(accumulate_time > ENTRYGRAIN)  \
            {  solver-> dequeue_multiple(avgentrytime, processed_nodes);} \
            processed_nodes++; \
        }  \
    }
#endif

#else
#define registerSE_DEF(state, f1, f2, f3, f4)    \
    void registerSE() {    \
      SE_register(f1, f2, f3, f4);   \
    }
#if ! ADAPTIVE
#define SE_Register(state, f1, f2, f3, f4)  \
    registerSE_DEF(state, f1, f2, f3, f4)  \
    void createMultipleChildren(StateBase *parent, SequentialSolver* solver, bool parallel) {  \
       f2(parent, solver, false);  \
       StateBase *state;  \
       while((state=solver->dequeue()) != NULL) \
       {  \
            f2(state, solver, parallel); \
       }  \
    } 
#else
#define SE_Register(state, f1, f2, f3, f4)  \
    registerSE_DEF(state, f1, f2, f3, f4)  \
    void createMultipleChildren(StateBase *parent, SequentialSolver* solver, bool parallel) {  \
       StateBase *state;  \
       double avgentrytime = 0;  \
       int processed_nodes = 1; \
       double instrument_start; \
       double accumulate_time = 0; \
       f2(parent, solver, false);  \
       instrument_start = CkWallTimer();\
       accumulate_time = avgentrytime  = CkWallTimer() - instrument_start;  \
       while((state=solver->dequeue()) != NULL) \
       {  \
            if(processed_nodes  == 20) \
            {  \
                avgentrytime  = (CkWallTimer() - instrument_start)/20;  \
            }  \
            f2(state, solver, parallel); \
            accumulate_time += avgentrytime; \
            if(accumulate_time > ENTRYGRAIN)  \
            {  solver-> dequeue_multiple(avgentrytime, processed_nodes);} \
            processed_nodes++; \
       } \
    } 
#endif

#endif

#endif
