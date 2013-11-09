/* searchEngine.C
 *
 *  Nov 3
 *
 * author: Yanhua Sun
 */
#include <vector>
#include <stack>
#include <deque>

using namespace std;

#ifdef USING_CONTROLPOINTS
#include <controlPoints.h>
#endif

#include "searchEngine.h"

static SE_createInitialChildrenFn createInitialChildren = NULL;
static SE_createChildrenFn createChildren = NULL;
static SE_parallelLevelFn     parallelLevel = NULL;
static SE_searchDepthLimitFn   searchDepthLimit = NULL;
SE_lowerBoundFn   _lowerBoundFn = NULL;

//#include "searchEngine_impl.h"

void printPriority(SearchNodeMsg *pm){
#if defined(USEBITPRIORITY) || defined(USEINTPRIORITY)
    UShort pw = UsrToEnv(pm)->getPrioWords();
    unsigned int *pr = (unsigned int *)(CkPriorityPtr(pm));
    CkPrintf("PE:%d ", CkMyPe());
    for(int i = 0; i < pw; i++){
        CkPrintf("[%d] 0x%x  ", i, pr[i]);
    }
    CkPrintf("\n");
#endif
}



#ifdef USING_CONTROLPOINTS
CProxy_BThreshold threshGroup;

int cp_grainsize;
int THRESH_MAX;
int THRESH_MIN;
void SearchConductor::controlChange(controlPointMsg* msg) {
    controlPointTimingStamp();
    cp_grainsize = controlPoint("grainsize", THRESH_MIN, THRESH_MAX);

    ThreshMsg *msg1 = new ThreshMsg(cp_grainsize);
    threshGroup.changeThreshold(msg1);
}
class BThreshold : public CBase_BThreshold {
public:
    BThreshold() {}

    void changeThreshold(ThreshMsg *msg) {
        cp_grainsize = msg->threshold;
    }
};

#endif


int se_statesize;     // readonly
CProxy_SearchConductor searchEngineProxy;
CProxy_SearchGroup groupProxy;


/****************************** search conductor  main chare */
SearchConductor::SearchConductor( CkArgMsg *m )
{
	searchEngineProxy = thisProxy;
	
	groupProxy = CProxy_SearchGroup::ckNew();
}
void SearchConductor::start()
{

    groupInitCount = 0;
    groupProxy.init();
}

void SearchConductor::groupInitComplete()
{
	groupInitCount ++;
	if( groupInitCount == CkNumPes() )
	{
#ifdef USING_CONTROLPOINTS
        ControlPoint::EffectIncrease::Concurrency("grainsize");
        cp_grainsize = controlPoint("grainsize", THRESH_MIN, THRESH_MAX);
        threshGroup = CProxy_BThreshold::ckNew();
        CkCallback cb(CkIndex_SearchConductor::controlChange(NULL), searchEngineProxy);
        registerCPChangeCallback(cb, true);
#endif
 
        groupInitCount = 0;
		fire();
	}
}

void SearchConductor::fire()
{
	
    int myStartDepth = 0;
    int mySearchDepth = 3;
	currentSearchDepth = 1;
	startTime = CkWallTimer();
  
    ParallelSolver parSolver;
    createInitialChildren(&parSolver);

    //set the QD call back function
    CkStartQD(CkIndex_SearchConductor::allSearchNodeDone((CkQdMsg *)0), &thishandle);
}

//QD call back function, it means all the searching is complete in current search depth
void SearchConductor::allSearchNodeDone( CkQdMsg *msg )
{

    long long numSolutions = groupProxy.ckLocalBranch()->getTotalSolutions();
    CkPrintf("All states are done in %lf sec \n",  CkWallTimer()-startTime);
#ifdef STATISTIC
    groupProxy.ckLocalBranch()->printfStatistic();
#endif

#ifdef BRANCHBOUND
    CkPrintf("Best solution is:%.4f\n Time cost:%lf on %d processors",groupProxy.ckLocalBranch()->getCost(), CkWallTimer()-startTime, CkNumPes());
    CkExit();
#endif
    if( numSolutions > 0 )
    {
        CkPrintf( "%lld solutions are found in %lf sec, with %d processors\n", numSolutions, CkWallTimer()-startTime, CkNumPes() );	
        CkExit(); 
    }
    else
    {
        currentSearchDepth ++;
        if( currentSearchDepth > searchDepthLimit() )
        {
            CkExit();
            return;
        }
        groupProxy.searchDepthChange( currentSearchDepth );
		
        //recreate the initail state, because the initial state may change as the search depth change
        ParallelSolver parSolver;
        createInitialChildren(&parSolver);

        //set the QD call back function
        CkStartQD(CkIndex_SearchConductor::allSearchNodeDone((CkQdMsg *)0), &thishandle);
        //delete parSolver;
    }

    delete msg; 
}

void SearchConductor::foundSolution()
{
    CkPrintf( "One solution is found in %lf sec, with %d processors\n",  CkWallTimer()-startTime, CkNumPes() );	
    CkExit(); 
}


/***************************************** Search Group   ***********************/

/* used to count the total number of solutions if all solutions are set */
SearchGroup::SearchGroup()
{ 
    // initialize the local count;
    mygrp = thisgroup;
    myCount = totalCount = 0;
#ifdef STATISTIC
    parallelnodes_generate = 0;
    parallelnodes_generate_sum = 0;
    parallelnodes_consume = 0;
    parallelnodes_consume_sum = 0;
    sequentialnodes_generate = 0;
    sequentialnodes_generate_sum = 0;
    sequentialnodes_consume = 0;
    sequentialnodes_consume_sum = 0;
#endif
    waitFor = CkNumPes(); // wait for all processors to report
    threadId = NULL;

#ifdef BRANCHBOUND
    minCost = 1000000;
#endif
}
#ifdef BRANCHBOUND
inline void SearchGroup::updateCost(double c)
{
    if(c<minCost) 
    {
        minCost = c;
    //   CkPrintf("min cost =%f\n", c);
    //    void **copyqueue;
    //    CqsEnumerateQueue((Queue)CpvAccess(CsdSchedQueue), &copyqueue);
    //    for(int i=0; i<CqsLength((Queue)CpvAccess(CsdSchedQueue)); i++)
    //    {
    //        void* msgchare = copyqueue[i];
    //        int prior = *(int*)CkPriorityPtr(msgchare);
    //        if(prior > minCost)
    //        {
    //              CqsRemoveSpecific((Queue)CpvAccess(CsdSchedQueue), msgchare);
    //              CkPrintf("node is removed, prior=%d, mincost=%f\n", prior, minCost);
    //        }
    //    }
    //    //CmiFree(copyqueue);
    }
}
#endif
// This method is invoked via a broadcast. Each branch then reports 
//  its count to the branch on 0 (or via a spanning tree.)
inline void SearchGroup::sendCounts()
{
    CProxy_SearchGroup grp(mygrp);
#ifdef STATISTIC
    grp[0].childCount(new countMsg(myCount, parallelnodes_generate, parallelnodes_consume, sequentialnodes_generate, sequentialnodes_consume));
#else
    grp[0].childCount(new countMsg(myCount));
#endif
}

inline void SearchGroup::childCount(countMsg *m)
{
    totalCount += m->count;
#ifdef STATISTIC
    parallelnodes_generate_sum  += m->pg;
    parallelnodes_consume_sum += m->pc;
    sequentialnodes_generate_sum += m->sg;
    sequentialnodes_consume_sum += m->sc;
#endif
    waitFor--;
    if (waitFor == 0) 
        if (threadId) { CthAwaken(threadId);}
    delete m;
}

long long  SearchGroup::getTotalSolutions() {
    CProxy_SearchGroup grp(mygrp);
    grp.sendCounts();//this is a broadcast, as no processor is mentioned
    threadId = CthSelf();
    while (waitFor != 0)  CthSuspend();
    return totalCount;
}

SearchGroup::~SearchGroup()
{
}

void SearchGroup::init()
{
    //get the factory from the user space
    ////this function is the bridge to pass the pointer to the search engine
    ////create new search core
    parallelize_level = parallelLevel();

#ifdef USING_CONTROLPOINTS
    THRESH_MIN = myCore->minimumLevel();
    THRESH_MAX = myCore->maximumLevel();
#endif
    //tell the mainchare that this branch is finished
    starttimer = CkWallTimer();
    searchEngineProxy.groupInitComplete();	
}


void SearchGroup::searchDepthChange( int depth) { /*searchDepthChangeNotify( depth );*/ }

void SearchGroup::killSearch()
{
#ifdef STATISTIC
    groupProxy.ckLocalBranch()->printfStatistic();
#endif
    CkExit();
}
//implementation of SearchNode
SearchNode::SearchNode( CkMigrateMessage *m ){}

SearchNode::~SearchNode() {}

#ifdef BRANCHBOUND
extern void createMultipleChildren(SearchGroup *s, StateBase *parent, SequentialSolver* solver, bool parallel);
#else
extern void createMultipleChildren(StateBase *parent, SequentialSolver* solver, bool parallel);
#endif

SearchNode::SearchNode( SearchNodeMsg *msg )
{

#ifdef STATISTIC
    groupProxy.ckLocalBranch()->inc_parallel_consume();     
#endif
    myGroup = groupProxy.ckLocalBranch();
    mySearchClass = (StateBase*) (msg->objectDump);

    myStartDepth = msg->startDepth;
    mySearchDepth = msg->searchDepth;

    if( mySearchDepth < myGroup->getParallelLevel() ){
        ParallelSolver solver;
#ifdef DEBUG
       CkPrintf("mySearchDepth: %d\n", mySearchDepth);
       printPriority(msg);
#endif
        solver.setParentInfo(msg, mySearchDepth);
#ifdef BRANCHBOUND
        double minCost = myGroup->getCost();
        double lb = _lowerBoundFn(mySearchClass);
        //CkPrintf("best solution is %f\n", minCost);
        if(lb<minCost)
            createChildren(mySearchClass, &solver, true);
        else
            CkPrintf("Node is pruned with lower bound:%f, best solution:%f\n", lb, minCost); 
#else
        createChildren(mySearchClass, &solver, true);
#endif
    }
    else
    {
        SequentialSolver solver;
        solver.setParentInfo(msg, mySearchDepth);
        StateBase *parent=NULL;// = mySearchClass;
        int processed_nodes = 0;
        if(msg->nodes == 1){
            solver.initialize();
            parent = mySearchClass;
        }
        else if(msg->nodes > 1){
            solver.initialize(msg);
            parent = solver.dequeue();
        }
#ifdef BRANCHBOUND
        createMultipleChildren(myGroup, parent, &solver, false);
#else
        createMultipleChildren(parent, &solver, false);
#endif
    }
    delete msg;
    delete this;
}

/* called in initproc */
void SE_register(SE_createInitialChildrenFn f1,
                 SE_createChildrenFn f2,
                 SE_parallelLevelFn f3,
                 SE_searchDepthLimitFn f4,
                 SE_lowerBoundFn f5
                 )
{
  createInitialChildren = f1;
  createChildren = f2;
  parallelLevel = f3;
  searchDepthLimit = f4;
  _lowerBoundFn = f5;

  CmiPoolAllocInit(30);
}


#ifdef BRANCHBOUND
#include "searchEngine_bound.def.h"
#else
#include "searchEngine.def.h"
#endif
