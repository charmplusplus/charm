/* searchEngine_impl.h
 *
 *  Nov 3
 *
 * author: Yanhua Sun
 */
#ifndef __SEARCHENGINE_IMPL_
#define __SEARCHENGINE_IMPL_



#ifdef BRANCHBOUND
#include "searchEngine_bound.decl.h"
#else
#include "searchEngine.decl.h"
#endif
#include "cmipool.h"

#define SIZEINT  sizeof(size_t)
#define MAXSTACKSIZE 409600
#define NODESINGROUP 50
#define ENTRYTHRESHOLD 100
extern CProxy_SearchConductor searchEngineProxy;
extern CProxy_SearchGroup groupProxy;
extern int se_statesize;

#ifdef USEBITPRIORITY
inline int __se_log(int n)
{
    int _mylog = 0;
    for(n=n-1;n>0;n=n>>1)
    {
        _mylog++;
    }
    return _mylog;
}

inline void setMsgPriority(SearchNodeMsg *msg, UShort childbits, unsigned int childnum, UShort newpw, UShort parentBits,unsigned int* parentPtr ){
    unsigned int *newPriority = (unsigned int *)(CkPriorityPtr(msg));
    UShort prioWords = CkPriobitsToInts(parentBits);
    for(int i=0; i<prioWords; i++)
    {
        newPriority[i] = parentPtr[i];
    }

    int shiftbits = 0;
    // Two cases 1: with new bits, we do not have to append the number of integers
    if(newpw == prioWords)
    {
        if((childbits+parentBits) % (8*sizeof(unsigned int)) != 0)
            shiftbits = 8*sizeof(unsigned int) - (childbits+parentBits)%(8*sizeof(unsigned int));
        newPriority[prioWords-1] = parentPtr[prioWords-1] | (childnum << shiftbits);
    }else if(newpw>prioWords)
    {
        /* have to append a new integer */
        if(parentBits % (8*sizeof(unsigned int)) == 0)
        {
            shiftbits = sizeof(unsigned int)*8 - childbits;
            newPriority[prioWords] = (childnum << shiftbits);
        }else /*higher bits are appended to the last integer and then use anothe new integer */
        {
            int inusebits = parentBits % ( 8*sizeof(unsigned int));
            unsigned int higherbits =  childnum >> (childbits - inusebits);
            newPriority[prioWords-1] = parentPtr[prioWords-1] | higherbits;
            /* lower bits are stored in new integer */
            newPriority[prioWords] = childnum << (8*sizeof(unsigned int) - childbits + inusebits);
        }
    }

}
 
#endif

class SearchNodeMsg : public CMessage_SearchNodeMsg
{
public:
	
    int startDepth, searchDepth;
//    int data_offset;
    int top;
    int nodes;
    char *msgptr;   // pointer to this message, make sure it is pointer aligned 
    char *objectDump;

    SearchNodeMsg( int sdepth, int depth, int t, int n):
                 startDepth(sdepth), searchDepth(depth), top(t), nodes(n)
    {
    }

    SearchNodeMsg(){}
};



/****************************************** search engine */
/* for count solutions */
class countMsg : public CMessage_countMsg {
public:
  long long count;
#ifdef STATISTIC
  int pg, pc, sg, sc;

  countMsg(long long c, int c1, int c2, int c3, int c4) : count(c), pg(c1), pc(c2), sg(c3), sc(c4){};
#endif
  countMsg(long long c) : count(c){};
};

class SearchConductor : public CBase_SearchConductor
{
public:	
    double startTime;
	
    int currentSearchDepth, mySearchLimit;
    int groupInitCount;
    int solutionFound;
    int parallelize_level;
	
    SearchConductor( CkArgMsg *m );

    void allSearchNodeDone( CkQdMsg *msg );
    void start();
    void groupInitComplete();
    void foundSolution();

    void fire();

#ifdef USING_CONTROLPOINTS
    void controlChange(controlPointMsg* msg);
#endif
};

class SearchGroup : public Group
{
private:
    CkGroupID mygrp;
    long long myCount;
    long long totalCount;

    double starttimer;
#ifdef BRANCHBOUND
    double minCost;
#endif
#ifdef STATISTIC
    int parallelnodes_generate;
    int parallelnodes_generate_sum;
    int parallelnodes_consume;
    int parallelnodes_consume_sum;
    int sequentialnodes_generate;
    int sequentialnodes_generate_sum;
    int sequentialnodes_consume;
    int sequentialnodes_consume_sum;
#endif
    int waitFor;
    CthThread threadId;
public:
    int parallelize_level;

    SearchGroup(CkMigrateMessage *m) {}
    SearchGroup();
	
    void killSearch();
    ~SearchGroup();
    double getStartTimer() {return starttimer;} 
    void childCount(countMsg *);
    void increment() {myCount++;}
#ifdef BRANCHBOUND
    double getCost() {return minCost;}
    //void  updateCostLocal(double c) { /*CkPrintf("before update best is:%f, new solution:%f\n", minCost, c); */if(c<minCost) minCost = c;}
    void  updateCost(double c);
#endif
#ifdef STATISTIC
    void inc_parallel_generate() {parallelnodes_generate++;}
    void inc_parallel_consume() {parallelnodes_consume++;}

    void inc_sequential_generate() {sequentialnodes_generate++;}
    void decrease_sequential_generate() {sequentialnodes_generate--;}
    void inc_sequential_consume() {sequentialnodes_consume++;}

    int printfStatistic()
    {
        CkPrintf(" Search Engine Statistic Information:\nParallel nodes produced:%d, processed:%d\n sequential nodes produced:%d, processed:%d\n", parallelnodes_generate_sum, parallelnodes_consume_sum, sequentialnodes_generate_sum, sequentialnodes_consume_sum);
    }
#endif
    void sendCounts();
    long long  getTotalSolutions();
	
    void init();
    inline void setParallelLevel( int level ) { parallelize_level = level;}
    inline void searchDepthChange( int depth);
	
    inline int getParallelLevel() {return parallelize_level;}

};

class SearchNode : public CBase_SearchNode
{
public:
	int myStartDepth;
	int mySearchDepth;

	SearchGroup *myGroup;
	StateBase *mySearchClass;
	
	SearchNode( CkMigrateMessage *m );
	SearchNode( SearchNodeMsg *msg );
	~SearchNode();
	
};

/**************************** parallel and sequential solver */

inline void Solver::reportSolution()
{
#ifdef ONESOLUTION
    double startime = groupProxy.ckLocalBranch()->getStartTimer();
    CkPrintf("First solution found within time %f second\n", CkWallTimer()-startime);
    groupProxy.ckLocalBranch()->increment();
    groupProxy.killSearch();
    CkExit();
#else
    groupProxy.ckLocalBranch()->increment();
#endif
 
}

#ifdef BRANCHBOUND
inline void Solver::updateCost(double c)
{
    //CkPrintf("updating best solution:%.4f\n", c);
    groupProxy.updateCost(c);
}
#endif

class ParallelSolver: public Solver {
public:
    inline void setParentInfo(SearchNodeMsg *msg, int l)
    {
#if defined(USEBITPRIORITY) || defined(USEINTPRIORITY)
        parentBits = UsrToEnv(msg)->getPriobits();
        parentPtr = (unsigned int *)(CkPriorityPtr(msg));
#endif
        searchLevel = l+1;
    }

    inline void setPriority(StateBase *s, int p)
    {
        SearchNodeMsg *msg = *(SearchNodeMsg**)((char*)s - 2*sizeof(void*));
#ifdef USEINTPRIORITY
        CkSetQueueing(msg, CK_QUEUEING_ILIFO);
        *(int*)CkPriorityPtr(msg) = p;
#endif
    }

    inline StateBase *registerRootState(size_t size, unsigned int childnum, unsigned int totalNumChildren)
    {
#ifdef STATISTIC
        groupProxy.ckLocalBranch()->inc_parallel_generate();     
#endif

#ifdef USEBITPRIORITY
        UShort rootBits = __se_log(totalNumChildren)+1;
        SearchNodeMsg *msg = new (size, rootBits) SearchNodeMsg(0, 0, size, 1);
        unsigned int *pbits = (unsigned int *)CkPriorityPtr(msg);
        *pbits = childnum << (SIZEINT*8 - rootBits);;
        CkSetQueueing(msg, CK_QUEUEING_BLIFO);
#elif USEINTPRIORITY
        SearchNodeMsg *msg = new (size , 8*sizeof(int)) SearchNodeMsg(0, 0, size, 1);
        CkSetQueueing(msg, CK_QUEUEING_ILIFO);
        *(int*)CkPriorityPtr(msg) = childnum;
#else
        SearchNodeMsg *msg = new (size , 0) SearchNodeMsg(0,0, size, 1);
#endif
        msg->searchDepth =  0;
        msg->msgptr = (char*)msg;
        return (StateBase *) (msg->objectDump);

    }

    inline StateBase *registerState(size_t size, unsigned int childnum, unsigned int totalNumChildren) 
    {

#ifdef STATISTIC
        groupProxy.ckLocalBranch()->inc_parallel_generate();     
#endif
#ifdef USEBITPRIORITY
        UShort extraBits = __se_log(totalNumChildren);
        CkAssert(extraBits <= CkIntbits);
        UShort newpbsize = extraBits + parentBits;
        SearchNodeMsg *msg = new (size, newpbsize) SearchNodeMsg(0, searchLevel, size, 1);
        CkSetQueueing(msg, CK_QUEUEING_BLIFO);
        setMsgPriority(msg, extraBits, childnum, CkPriobitsToInts(newpbsize), parentBits, parentPtr );
#elif USEINTPRIORITY
        SearchNodeMsg *msg = new (size , 8*sizeof(int)) SearchNodeMsg(0, searchLevel, size, 1);
        CkSetQueueing(msg, CK_QUEUEING_ILIFO);
        *(int*)CkPriorityPtr(msg) = childnum;
#else
        SearchNodeMsg *msg = new (size , 0) SearchNodeMsg(0, searchLevel, size, 1);
#endif
        msg->msgptr = (char*)msg;
        return (StateBase *) (msg->objectDump);
    }

    inline void deleteState(StateBase* s)
    {
        //SearchNodeMsg *msg = *(SearchNodeMsg**)((char*)s - 2*sizeof(void*));
        SearchNodeMsg *msg = *((SearchNodeMsg**)s - 2);
        delete msg;
    }

    /* recover the message from the state */
    inline void process(StateBase* s)
    {
        //SearchNodeMsg *msg = *(SearchNodeMsg**)((char*)s - 2*sizeof(void*));
#ifdef BRANCHBOUND
        if(_lowerBoundFn(s) >= groupProxy.ckLocalBranch()->getCost())
        {
            deleteState(s);
            return;
        }
#endif
        SearchNodeMsg *msg = *((SearchNodeMsg**)s - 2);
        CProxy_SearchNode::ckNew(msg, NULL, -1);
#ifdef STATISTIC
        groupProxy.ckLocalBranch()->inc_parallel_generate();     
#endif
    }
};

#ifdef USE_CMIPOOL
#define mymalloc  CmiPoolAlloc
#define myfree    CmiPoolFree
#else
#define mymalloc  malloc
#define myfree    free
#endif

#ifdef SE_FIXED_STATE
class StateStack {
   char* mystack;
   char* cur;
   int maxsize;
   bool msgstack;

   void expand(int size) {
       while(cur - mystack + size >maxsize)
           maxsize *= 2;
       //CkPrintf("maxsize=%d, top=%d, top=%d\n", maxsize, cursize() + size + SIZEINT,  cursize());
       char* newstack = (char*)mymalloc(maxsize);
       memcpy(newstack, mystack, cursize());
       cur = newstack + cursize();
       free_stack();
       if(msgstack) msgstack = false;
       mystack = newstack;
   }
public:
   StateStack(): mystack(NULL), cur(NULL), maxsize(0), msgstack(false) {}
   ~StateStack(){
       free_stack();
   }                        
   inline void set(int size) {
       maxsize = size;
       mystack = cur = (char*)mymalloc(maxsize);
   }
   inline void set(SearchNodeMsg *m) {
       mystack = m->objectDump;
       cur = mystack + m->top;
       maxsize = m->top;    // ?
       msgstack = true;
   }
   inline void freeMsg() {
       SearchNodeMsg *msg = *((SearchNodeMsg**)mystack - 2);
       delete msg;
   }
   inline void free_stack() {
       if(!msgstack) myfree(mystack);
   }
   inline int empty() const {  return cur == mystack; }
   inline char *top() const { return cur; }
   inline int cursize() const { return cur - mystack; }
   inline void clear() { cur = mystack; }
   inline char* push(int size)
   {
       if(cursize() + size >maxsize) expand(size);
       char * const oldcur = cur;
       cur += size;
       return oldcur;
   }
   inline StateBase *pop(){
        if(empty()) return NULL;
        cur -= se_statesize;
        return (StateBase*)cur;
    }
    inline int popN(int n) {
        int i;
        for( i=0; !empty() && i<n; i++)
        {
            cur -= se_statesize;
        }
        return i;
    }
};
#else
class StateStack {
   char* mystack;
   char* cur;
   int maxsize;
   bool msgstack;

   void expand(int size) {
       while(cur - mystack + size + SIZEINT >maxsize)
           maxsize *= 2;
       //CkPrintf("State stack expand: maxsize=%d, top=%d, top=%d\n", maxsize, cursize() + size + SIZEINT,  cursize());
       char* newstack = (char*)mymalloc(maxsize);
       memcpy(newstack, mystack, cursize());
       cur = newstack + cursize();
       free_stack();
       if(msgstack) msgstack = false;
       mystack = newstack;
       //CkPrintf("maxsize=%d, top=%d, stack=%p %p\n", maxsize, cursize() + size + SIZEINT,  mystack, cur);
   }
public:
   StateStack(): mystack(NULL), cur(NULL), maxsize(0), msgstack(false) {}
   ~StateStack(){
       free_stack();
    }                        
   inline void set(int size) {
     maxsize = size;
     mystack = cur = (char*)mymalloc(maxsize);
   }
   inline void set(SearchNodeMsg *m) {
     mystack = m->objectDump;
     cur = mystack + m->top;
     maxsize = m->top;    // ?
     msgstack = true;
   }
   inline void freeMsg() {
        SearchNodeMsg *msg = *((SearchNodeMsg**)mystack - 2);
        delete msg;
   }
   inline void free_stack() {
       if(!msgstack) myfree(mystack);
   }
   inline char* push(int size)
   {
        if(cursize() + size + SIZEINT >maxsize) expand(size);
        char * const oldcur = cur;
        *((size_t*)(cur+size)) = size;
        cur += (size + SIZEINT);
        return oldcur;
   }
   inline StateBase *pop(){
        if(empty()) return NULL;
        const size_t size = *((size_t*)(cur-SIZEINT));
        cur -= (SIZEINT + size);
        return (StateBase*)cur;
    }
    inline int popN(int n) {
        int i=0;
        for(i=0 ; !empty() && i<n; i++)
            {
                size_t size = *((size_t*)(cur-SIZEINT));
                cur -= (SIZEINT + size);
            }
         return i;
    }
    inline int empty() const {  return cur == mystack; }
    inline char *top() const { return cur; }
    inline int cursize() const { return cur - mystack; }
    inline void clear() { cur = mystack; }
};
#endif

class SequentialSolver : public Solver { 

private:
    int nodenum;
    StateStack stack;
public:                                                            
    SequentialSolver() {}
    ~SequentialSolver() {}                        
    inline void initialize()
    {
        stack.set(MAXSTACKSIZE);
        nodenum = 0;
    }

    inline void initialize(SearchNodeMsg *m)
    {
        stack.set(m);
        nodenum = m->nodes;
    }
    
    inline StateBase *dequeue(){
        StateBase *state = stack.pop();
        if (state != NULL)  {
          nodenum--;
#ifdef STATISTIC
          groupProxy.ckLocalBranch()->inc_sequential_consume();     
#endif
        }
        return state;
    }
    
    inline void dequeue_multiple(double avgentrytime, int maxexecnodes)
    {
        int num_dequeued = 0;
        maxexecnodes = 1;
        int groups = nodenum/maxexecnodes;
        if(nodenum % maxexecnodes> 0)
            groups ++;
#ifdef USEBITPRIORITY
        UShort extraBits = __se_log(groups);
#endif        
        int groupIndex = 0;
        //CkPrintf("Fine grain decomposition\n");
        while(!stack.empty())
        {
            /* top ---------- top_top is a chunk */
            const char *old_top = stack.top();
            int c = stack.popN(maxexecnodes);
            const char *currenttop = stack.top(); 
            const int chunk_size = old_top - currenttop;
            //CkPrintf("[%d]fire new tasks with %d  nodes, chunksize=%d, old_top=%p, new top=%p\n", groupIndex, c, chunk_size, old_top, currenttop);
#ifdef USEBITPRIORITY
            UShort newpbsize = extraBits + parentBits;
            SearchNodeMsg *msg = new (chunk_size, newpbsize)SearchNodeMsg(0, searchLevel, chunk_size, c);
            CkSetQueueing(msg, CK_QUEUEING_BLIFO);
            setMsgPriority(msg, extraBits, groupIndex, CkPriobitsToInts(newpbsize), parentBits, parentPtr);
#elif USEINTPRIORITY
            SearchNodeMsg *msg = new (chunk_size, 8*sizeof(int)) SearchNodeMsg(0, searchLevel, chunk_size, c);
            CkSetQueueing(msg, CK_QUEUEING_ILIFO);
            *(int*)CkPriorityPtr(msg) = *parentPtr;
#else
            SearchNodeMsg *msg = new (chunk_size, 0)SearchNodeMsg(0, searchLevel, chunk_size, c);
#endif
            memcpy(msg->objectDump, stack.top(), chunk_size);
            nodenum-=c; 
            CProxy_SearchNode::ckNew(msg, NULL, -1);
            groupIndex++;
        }
    }

    inline void setParentInfo(SearchNodeMsg *msg, int l)
    {
#if defined(USEBITPRIORITY) || defined(USEINTPRIORITY)
        parentBits = UsrToEnv(msg)->getPriobits();
        parentPtr = (unsigned int *)(CkPriorityPtr(msg));
#endif
        searchLevel = l+1;
 
    }

    inline void setPriority(StateBase *s, int p){}

    inline StateBase *registerRootState(size_t size, unsigned int childnum, unsigned int totalNumChildren)
    {
#ifdef STATISTIC
        groupProxy.ckLocalBranch()->inc_sequential_generate();     
#endif
        nodenum++;
        return (StateBase*)stack.push(size);
    }

    inline StateBase *registerState(size_t size, unsigned int childnum, unsigned int totalNumChildren)
    {
#ifdef STATISTIC
        groupProxy.ckLocalBranch()->inc_sequential_generate();     
#endif
        nodenum++;
        return (StateBase *)stack.push(size);
    }

       /* pop up the top one since it is always a stack */
    inline void deleteState(StateBase* s)
    {
        StateBase *state = stack.pop();
        if (state) {
          nodenum--;
#ifdef STATISTIC
          if (state)
            groupProxy.ckLocalBranch()->decrease_sequential_generate();
#endif
        }
    }
    inline void process(StateBase *s)
    {
#ifdef BRANCHBOUND
        if(_lowerBoundFn(s) >= groupProxy.ckLocalBranch()->getCost())
            deleteState(s);
        // sort again 
#endif
    }

    inline void reportSolution()
    {
        Solver::reportSolution();
#ifdef ONESOLUTION
        stack.clear();
#endif
    }

};
#endif
