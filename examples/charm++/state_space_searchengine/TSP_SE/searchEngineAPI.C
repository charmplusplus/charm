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

    void initialize()
    {
        tour = (int*) ( (char*)this+sizeof(TspBase));
    }
    void copy(TspBase *parent)
    {
        length = parent->length;
        cost = parent->cost;

        for(int i=0;i<length; i++)
        {
            tour[i] = parent->tour[i];
        }

    }

    double lowerbound()
    {
        vector< int > temp_graph;
        temp_graph.resize(N*N);
        for(int i=0; i<N*N; i++)
            temp_graph[i] = graph[i];

        if(length >= 2)
        {
            int v1 = tour[0];
            int v2 = tour[1];
            temp_graph[v1*N+v2] = maxD;
            temp_graph[v2*N+v1] = maxD;
            for(int i=1; i<length-1; i++)
            {
                v1=tour[i];
                v2 = tour[i+1];

                for(int j=0; j<N; j++)
                    temp_graph[v1*N+j] = maxD;
                temp_graph[v2*N+v1]= maxD;
            }
        }
        double lower_sum = 0;
        for(int i=0; i<N; i++)
        {
            if(i==0 || i== tour[length-1])
            {
                lower_sum += *min_element(temp_graph.begin()+i*N, temp_graph.begin()+(i+1)*N);
            }else
            {
                int j;
                for( j=1; j<length-1; j++)
                {
                    if(i == tour[j])
                        break;
                }
                if(j < length-1)
                    continue;
                sort(temp_graph.begin()+i*N, temp_graph.begin()+(i+1)*N-1);

                lower_sum += temp_graph[i*N] + temp_graph[i*N+1];
            }
        }
        return (cost + lower_sum); 
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

    TspBase *state = (TspBase*)solver->registerRootState(sizeof(TspBase)+sizeof(int)*N, 0, 1);
    state->initialize();
    state->tour[0]=0;
    state->length=1;
    state->cost = 0;
#ifdef USEINTPRIORITY
    solver->setPriority(state, (int)lowerBound(state));
#endif
    solver->process(state);
}

inline void createChildren( StateBase *_base , Solver* solver, bool parallel)
{

    TspBase *s = (TspBase*)alloca(sizeof(TspBase)+sizeof(int)*N);
    s->initialize();
    s->copy((TspBase*)_base);
    int childIndex = 0;
    int last = s->tour[s->length-1];
    /* construct the current graph, remove all the edges already in the path */
    vector< int > temp_graph;
    temp_graph.resize(N*N);
    for(int i=0; i<N*N; i++)
        temp_graph[i] = graph[i];

    if(s->length >= 2)
    {
        int v1 = s->tour[0];
        int v2 = s->tour[1];
        temp_graph[v1*N+v2] = maxD;
        temp_graph[v2*N+v1] = maxD;
        for(int i=1; i<s->length-1; i++)
        {
            v1=s->tour[i];
            v2 = s->tour[i+1];

            for(int j=0; j<N; j++)
                temp_graph[v1*N+j] = maxD;
            temp_graph[v2*N+v1]= maxD;
        }
    }
    
    for(int j=0; j<N; j++)
    {
        if(last == j || temp_graph[last*N+j] == maxD)
            continue;
        if(s->length == N)
        {
            if(j == 0)
            {
                //s->printInfo();
                solver->updateCost(s->cost+temp_graph[last*N+j]);
                //solver->reportSolution(); 
            }
        }else
        {
            TspBase *NewTour  = (TspBase*)solver->registerState(sizeof(TspBase)+sizeof(int)*N, childIndex, N);
            NewTour->initialize();
            NewTour->copy(s);
            NewTour->tour[s->length] = j;
            NewTour->length = s->length + 1;
            
            NewTour->cost = s->cost + temp_graph[last*N+j];
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


