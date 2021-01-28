/*
 * exampleTsp.C
 *
 *  Created on: Mar 4, 2010
 */

#include "searchEngine.h"
#include "exampleTsp.h"

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
using namespace std;

extern int se_statesize;

/*readonly*/ extern int initial_grainsize;
#define MAX 100000
extern CkVec< double > graph; 
class TspBase: public StateBase
{
public:
	int length; // length
    double cost;
    double lowerBoundValue;
    int *tour;
    int *intour; //whether a node is in tour or not

    void initialize()
    {
        tour = (int*) ( (char*)this+sizeof(TspBase));
        intour = (int*) ((char*) this+sizeof(TspBase)+sizeof(int)*N);
    }
    void copy(TspBase *parent)
    {
        length = parent->length;
        cost = parent->cost;
        lowerBoundValue = parent->lowerBoundValue;
        for(int i=0;i<length; i++)
        {
            tour[i] = parent->tour[i];
        }
        for(int i=0; i<N; i++)
        {
            intour[i] = parent->intour[i];
        }
    }

    double lowerbound()
    {
        
        double lower_sum = 0;
        for(int i=0; i<N; i++)
        {
            if(intour[i] == 2)
                continue;
            if(intour[i] == 1 && i!=0)
            {
                double m_min=MAX;
                for(int j=0; j<N; j++)
                {
                    if((intour[j] ==0 && graph[j]<m_min) || (length==N && intour[j]==1))
                        m_min= graph[j];
                }
                lower_sum += m_min;
            }else
            {
                double m_min, m2_min;
                m_min = m2_min=graph[i*N];
                for(int j=i*N+1; j<i*N+N; j++)
                {
                    if(graph[j]<m_min)
                    {
                        m2_min=m_min;
                        m_min=graph[j];
                    }
                }
                lower_sum += m_min+m2_min;
            }
        }
        lowerBoundValue = (cost + 0.5*lower_sum);
        return lowerBoundValue; 
    }

    double getLowerBound()
    {
        return lowerBoundValue;
    }

    void printInfo()
    {
        CkPrintf("\n length=%d, lowerBound=%f", length, lowerBoundValue);
        for(int i=0; i<length; i++)
        {
            CkPrintf("(%d, value=%f", tour[i], graph[tour[i]*N+tour[(i+1)%N]]);
            CkPrintf("%d)", intour[tour[i]]);
        }
    }
};


inline double cost( StateBase *s )
{
    return ( ((TspBase*)s)->cost);					// cost of given node
}

inline double lowerBound( StateBase *s)
{
    return ((TspBase*)s)->getLowerBound();
}

inline void createInitialChildren(Solver *solver)
{

//    se_statesize = sizeof(TspBase)+2*sizeof(int)*N;
    TspBase *state = (TspBase*)solver->registerRootState(sizeof(TspBase)+2*sizeof(int)*N, 0, 1);
    state->initialize();
    state->tour[0]=0;
    state->length=1;
    state->cost = 0;
    for(int i=0; i<N; i++)
        state->intour[i]=0;
    state->lowerbound();
#ifdef USEINTPRIORITY
    solver->setPriority(state, (int)lowerBound(state));
#endif
    CkAssert(state->length>0);
    solver->process(state);
}

inline void createChildren( StateBase *_base , Solver* solver, bool parallel)
{

    TspBase *s = (TspBase*)alloca(sizeof(TspBase)+2*sizeof(int)*N);
    s->initialize();
    ((TspBase*)_base)->initialize();
    s->copy((TspBase*)_base);
    int childIndex = 0;
    CkAssert(((TspBase*)_base)->length>0);
    CkAssert(s->length>0);
    int last = s->tour[s->length-1];
    int childNum = 0;
    //CkPrintf("lower bound=%f\n", s->getLowerBound());
    /* construct the current graph, remove all the edges already in the path */
    //s->printInfo();
    for(int j=0; j<N; j++)
    {
        //CkPrintf("last=%d,j=%d, intour=%d\n", last, j, s->intour[j]);
        if(last == j || s->intour[j]==2 || (j==0&&s->length<N))
            continue;
        if(s->length == N)
        {
            if(j == 0)
            {
                //s->printInfo();
                //CkPrintf(" cost=%f\n", s->cost+graph[last*N+j]);
                solver->updateCost(s->cost+graph[last*N+j]);
                //solver->reportSolution(); 
            }
        }else
        {
            childNum++;
            TspBase *NewTour  = (TspBase*)solver->registerState(sizeof(TspBase)+2*sizeof(int)*N, childIndex, N);
            NewTour->initialize();
            NewTour->copy(s);
            NewTour->tour[s->length] = j;
            NewTour->intour[j]=1;
            NewTour->intour[last] = NewTour->intour[last]+1;
            NewTour->length = s->length + 1;
            NewTour->cost = s->cost + graph[last*N+j];
            NewTour->lowerbound();
            //NewTour->printInfo();
#ifdef USEINTPRIORITY
            solver->setPriority(NewTour, (int)lowerBound(NewTour));
#endif
            CkAssert(NewTour->length>0);
            solver->process(NewTour);
        }
    }
    //CkPrintf(" %d children created\n", childNum);
}
int parallelLevel()
{
    return initial_grainsize;
}

int searchDepthLimit()
{
    return 1;
}

void set_statesize(int N)
{
        se_statesize = sizeof(TspBase)+2*sizeof(int)*N;
}

    SE_Register(TspBase, createInitialChildren, createChildren, parallelLevel, searchDepthLimit, lowerBound);


