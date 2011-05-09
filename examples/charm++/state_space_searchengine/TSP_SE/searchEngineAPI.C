/*
 * exampleTsp.C
 *
 *  Created on: Mar 4, 2010
 *      Author: gagan
 */

#include "searchEngine.h"
#include "exampleTsp.h"

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
using namespace std;


/*readonly*/ extern int initial_grainsize;

extern CkVec< int > graph; 
class TspBase: public StateBase
{
public:
	int length; // length
	double cost;
    int *tour;
    int *intour;

    void initialize()
    {
        tour = (int*) ( (char*)this+sizeof(TspBase));
        intour = (int*) ((char*) this+sizeof(TspBase)+sizeof(int)*N);
    }
    void copy(TspBase *parent)
    {
        length = parent->length;
        cost = parent->cost;

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
        vector< int > temp_graph;
        temp_graph.resize(N*N);
        for(int i=0; i<N*N; i++)
            temp_graph[i] = graph[i];
        
        double lower_sum = 0;
        for(int i=0; i<N; i++)
        {
            if(intour[i] == 2)
                continue;
            if(intour[i] == 1)
            {
                int m_min=maxD;
                for(int j=0; j<N; j++)
                {
                    if(intour[j] ==0 && graph[j]<m_min)
                        m_min= graph[j];
                }
                lower_sum += m_min;
            }else
            {
                int m_min, m2_min;
                m_min = m2_min=temp_graph[i*N];
                for(int j=1; j<N; j++)
                {
                    if(graph[j]<m_min)
                    {
                        m2_min=m_min;
                        m_min=temp_graph[j];
                    }
                }
                lower_sum += m_min+m2_min;
            }
        }
        return (cost + 0.5*lower_sum); 
    }

    void printInfo()
    {
        CkPrintf(" length=%d, cost=%.4f\n[", length, cost);
        for(int i=0; i<length; i++)
        {
            CkPrintf("%d  ", tour[i]);
        }
        CkPrintf("]\n");
    }
};


inline double cost( StateBase *s )
{
    return ( ((TspBase*)s)->cost);					// cost of given node
}

inline double lowerBound( StateBase *s)
{
    return ((TspBase*)s)->lowerbound();
}

inline void createInitialChildren(Solver *solver)
{

    TspBase *state = (TspBase*)solver->registerRootState(sizeof(TspBase)+2*sizeof(int)*N, 0, 1);
    state->initialize();
    state->tour[0]=0;
    state->length=1;
    state->cost = 0;
    for(int i=1; i<N; i++)
        state->intour[i]=0;
    state->intour[0] = 1;
#ifdef USEINTPRIORITY
    solver->setPriority(state, (int)lowerBound(state));
#endif
    solver->process(state);
}

inline void createChildren( StateBase *_base , Solver* solver, bool parallel)
{

    TspBase *s = (TspBase*)alloca(sizeof(TspBase)+2*sizeof(int)*N);
    s->initialize();
    s->copy((TspBase*)_base);
    int childIndex = 0;
    int last = s->tour[s->length-1];
    /* construct the current graph, remove all the edges already in the path */
    for(int j=0; j<N; j++)
    {
        if(last == j || s->intour[j]==2)
            continue;
        if(s->length == N)
        {
            if(j == 0)
            {
                //s->printInfo();
                solver->updateCost(s->cost+graph[last*N+j]);
                //solver->reportSolution(); 
            }
        }else
        {
            TspBase *NewTour  = (TspBase*)solver->registerState(sizeof(TspBase)+sizeof(int)*N, childIndex, N);
            NewTour->initialize();
            NewTour->copy(s);
            NewTour->tour[s->length] = j;
            NewTour->intour[j]=1;
            NewTour->intour[last]=2;
            NewTour->length = s->length + 1;
            NewTour->cost = s->cost + graph[last*N+j];
#ifdef USEINTPRIORITY
            solver->setPriority(NewTour, (int)lowerBound(NewTour));
#endif
            if(parallel)
                solver->process(NewTour);
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

    SE_Register(TspBase, createInitialChildren, createChildren, parallelLevel, searchDepthLimit, lowerBound);


