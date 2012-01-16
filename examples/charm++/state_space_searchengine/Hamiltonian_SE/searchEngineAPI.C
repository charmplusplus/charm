#include <stdio.h>
#include <stdlib.h>
#include "searchEngine.h"
#include "searchEngineAPI.h"
#include "hamiltonian.h"
extern CkVec<int> inputGraph; 


void createInitialChildren(Solver *solver)
{
    HamiltonianStateBase *root = (HamiltonianStateBase*)solver->registerRootState(sizeof(HamiltonianStateBase)+3*verticesNum * sizeof(int), 0, 1);
    root->initialize();
    root->resetUnionId();
    solver->process(root);
}

void createChildren( StateBase *_base , Solver* solver, bool parallel)
{

#ifdef USERSOLVER
    if(!parallel)
    {
        CkPrintf("------------------------\nStarting sequential solver\n");
        SeqHamiltonianState *seqstate = new SeqHamiltonianState((HamiltonianStateBase*)_base);
        seqstate->recursiveSolver(solver);
        delete seqstate;
        return;
    }
#endif
    int state_size = sizeof(HamiltonianStateBase) + 3*verticesNum* sizeof(int); 
    HamiltonianStateBase* parent = (HamiltonianStateBase*)alloca(state_size);
    memcpy((char*)parent, (char*)_base, state_size);
    parent->initialize();

    /* copy to aliveEdges */
    vector< list<int> > aliveEdges;
    aliveEdges.resize(verticesNum);
    
    for(int i=0; i<inputGraph.size(); )
    {
        int nodesrc = inputGraph[i];
        i++;
        int count = inputGraph[i];
        i++;
        for(int j=0; j<count; j++)
        {
            int dest = inputGraph[i];
            i++;
            aliveEdges[nodesrc].push_back(dest);
            aliveEdges[dest].push_back(nodesrc);
        }
    }

    int *parentNodeUnionId = parent->UnionId;
    /* construct the alive edges */
    for(int i=0; i<verticesNum; i++)
    {
        int left_nb = parentNodeUnionId[3*i+1];
        int right_nb = parentNodeUnionId[3*i+2];
        /* only one edge is in the path set */
        if(left_nb >-1 && right_nb == -1)
        {
            aliveEdges[i].remove(left_nb);
        }else if(right_nb > -1) //this vertice already has two edges
        {
            //delete all its neighbors
            list<int>::iterator iter;
            for(iter=aliveEdges[i].begin(); iter != aliveEdges[i].end(); iter++)
            {   /* remove edge  (i, *iter) */
                aliveEdges[*iter].remove(i);
            }
            aliveEdges[i].clear();
        }
    }  /* finish construct the alive edges */
    /* variable ordering, variable with fewest alive edges */
    /* find the vertice with fewest edges */
    int fewest_vertice = -1;
    int fewest_edges = verticesNum;
    for(int i=0; i<verticesNum; i++)
    {
        if((parent->UnionId)[3*i+2] >-1)
            continue;
        else if(fewest_edges > aliveEdges[i].size())
        {
            fewest_vertice = i;
            fewest_edges = aliveEdges[i].size();
        }
    }

    if( fewest_vertice == -1 || fewest_edges == 0 || (fewest_edges == 1 && parentNodeUnionId[3*fewest_vertice] == -1))
    {
        return;
        /* Not solution */
    }
    /* sort all the neighbor vertices with fewest vertice */
    vector<mypair> neighborVertices;
    list<int>::iterator iter; 
    for(iter=aliveEdges[fewest_vertice].begin(); iter != aliveEdges[fewest_vertice].end(); iter++)
    {
        mypair pair_1(*iter, aliveEdges[*iter].size(), !parallel);
        neighborVertices.push_back(pair_1);
    }
    /* from lower to higher */
#ifdef VALUE_HEURISTIC
        sort(neighborVertices.begin(), neighborVertices.end());
#endif
    if(parentNodeUnionId[3*fewest_vertice+1] != -1) //has left neighbor, one edge in the selected path
    {
        /* choose another vertice */
        /* fire aliveEdgeNum[fewest_vertice] new childrens */
        int totalChildren = aliveEdges[fewest_vertice].size();
        int childIndex = 0;
        //CkPrintf(" each state size is %d, number %d\n", state_size, totalChildren);
        for(vector<mypair>::iterator v_iter=neighborVertices.begin(); v_iter!=neighborVertices.end(); v_iter++, childIndex++)
        {
            mypair top_pair = *v_iter;
            int branchVertice = top_pair.getVertice();
            HamiltonianStateBase *child = (HamiltonianStateBase*)solver->registerState(state_size, childIndex, totalChildren);
            memcpy((char*)child, (char*)parent, state_size);
            child->initialize();
                        
            int ret = child->addOneEdgetoPath(fewest_vertice, branchVertice);
            if(ret == -1)
            {
                solver->deleteState(child);
                continue;
            }else if(ret == 0)
            {
                solver->deleteState(child);
                //child->printSolution();
                solver->reportSolution();
            }else
            {
                child->incEdgesInPath(1);
                if(parallel)
                    solver->process(child);
            }
        }  // cases that this branch variable has left neighbor end while
    }else //this vertice is not in the selected path, unionId == -1
    {
        int nb_edges = aliveEdges[fewest_vertice].size();
        vector<triple> allchoices;//[nb_edges * nb_edges / 2];
        for(vector<mypair>::iterator v_iter_outer=neighborVertices.begin(); v_iter_outer!=neighborVertices.end(); v_iter_outer++)
        {
            mypair pair_outer = *v_iter_outer;
            for(vector<mypair>::iterator v_iter_inner= v_iter_outer+1; v_iter_inner!=neighborVertices.end(); v_iter_inner++)
            {
                mypair pair_inner = *v_iter_inner;
                triple onetriple(pair_outer.verticeIndex, pair_inner.verticeIndex, pair_outer.aliveEdges +  pair_inner.aliveEdges, !parallel);
                allchoices.push_back( onetriple);
            }
        }
#ifdef VALUE_HEURISTIC
        sort(allchoices.begin(), allchoices.end());
        /* from lower to higher */
#endif
        int totalChildren = nb_edges*(nb_edges-1)/2;
        int childIndex = 0;
        //CkPrintf(" each state size is %d, number %d\n", state_size, totalChildren);
        /* All possible solutions */
        for( vector<triple>::iterator c_iter = allchoices.begin(); c_iter != allchoices.end(); c_iter++, childIndex++)
        {
            
            triple t_triple = *c_iter;
            int v_1 = t_triple.v1;
            int v_2 = t_triple.v2;
            HamiltonianStateBase *child = (HamiltonianStateBase*)solver->registerState(state_size, childIndex, totalChildren);
            memcpy((char*)child, (char*)parent, state_size);
            child->initialize();
            child->copy(parent);
            int ret = child->addTwoEdgestoPath(fewest_vertice, v_1, v_2);
            if(ret == -1)
            {
                solver->deleteState(child);
                continue;
            }else if(ret == 0)
            {
                //child->printSolution();
                solver->deleteState(child);
                solver->reportSolution();
            }else
            {
                child->incEdgesInPath(2);
                if(parallel)
                    solver->process(child);
            }
        } //end for
    } //end else
}

int parallelLevel()
{
    return initial_grainsize;
}

int searchDepthLimit()
{
    return 1;
}

#ifdef USERSOLVER

int SeqHamiltonianState::recursiveSolver(Solver *solver)
{
    SeqHamiltonianState *parent = this;
    /* variable ordering, variable with fewest alive edges */
    /* find the vertice with fewest edges */

    /* variable ordering, variable with fewest alive edges */
    /* find the vertice with fewest edges */
    int fewest_vertice = parent->findFewestVertice();
    if(fewest_vertice == -1 || parent->conflict(fewest_vertice))
    {
        return -1;
    }
    vector<mypair> neighborVertices;
    neighborVertices.clear();
    parent->sortNeighbor(fewest_vertice, neighborVertices);
#ifdef VALUE_HEURISTIC
    sort(neighborVertices.begin(), neighborVertices.end());
#endif
    //SeqHamiltonianState *child = new SeqHamiltonianState(parent);
    int childIndex = 0;
    int totalChildren = neighborVertices.size();
    //CkPrintf("output the fewest %d\n", fewest_vertice);
    //parent->printInfo();
    if( parent->OneEdgeInPath(fewest_vertice)) //one edge in the selected path
    {
        for(vector<mypair>::iterator v_iter=neighborVertices.begin(); v_iter!=neighborVertices.end(); v_iter++)
        {
            mypair top_pair = *v_iter;
            int branchVertice = top_pair.getVertice();
            //parent->printInfo();
            //child->printInfo();
            if(parent->OneEdgeInPath(branchVertice))
            {
                if( parent->detectImpossible(fewest_vertice, branchVertice))
                    continue;
            }
            SeqHamiltonianState *child = new SeqHamiltonianState(parent);
            int ret = child->addOneEdgetoPath(fewest_vertice, branchVertice);
            //CkPrintf("This child return %d (%d:%d), index=%d, total=%d\n", ret, fewest_vertice, branchVertice, childIndex++, totalChildren);
            if(ret == -1)
            {
                delete child;
                continue;
            }else if(ret == 0)
            {
                delete child;
                //child->printSolution();
                solver->reportSolution();
#ifdef ONESOLUTION
                return 0;
#else
                continue;
#endif
            }else
            {
                if(child->ImpossibleDectection())
                {
                    delete child;
                   // child->resetToParent(parent);
                    continue;
                }
                child->incEdgesInPath(1);
                ret = child->recursiveSolver(solver);
#ifdef ONESOLUTION
                if(ret == 0)
                {
                    delete child;
                    return 0;
                }
#endif
                delete child;
                //child->resetToParent(parent);
            }
        }
    }  // cases that this branch variable has left neighbor end while
    else
    {
        int nb_edges = neighborVertices.size();
        vector<triple> allchoices;//[nb_edges * nb_edges / 2];
        allchoices.clear();
        for(vector<mypair>::iterator v_iter_outer=neighborVertices.begin(); v_iter_outer!=neighborVertices.end(); v_iter_outer++)
        {
            mypair pair_outer = *v_iter_outer;
            int v_1 = pair_outer.verticeIndex;
            if(parent->OneEdgeInPath(v_1))
            {
                if( parent->detectImpossible(fewest_vertice, v_1))
                    continue;
            }
            for(vector<mypair>::iterator v_iter_inner= v_iter_outer+1; v_iter_inner!=neighborVertices.end(); v_iter_inner++)
            {
                mypair pair_inner = *v_iter_inner;
                int v_2 = pair_inner.verticeIndex;
                if(parent->OneEdgeInPath(v_2))
                {
                    if( parent->detectImpossible(fewest_vertice, v_2))
                        continue;
                }

                triple onetriple(pair_outer.verticeIndex, pair_inner.verticeIndex, pair_outer.aliveEdges +  pair_inner.aliveEdges);
                allchoices.push_back( onetriple);
            }
        }
#ifdef VALUE_HEURISTIC
        sort(allchoices.begin(), allchoices.end());
#endif
        for( vector<triple>::iterator c_iter = allchoices.begin(); c_iter != allchoices.end(); c_iter++)
        {
            triple t_triple = *c_iter;
            int v_1 = t_triple.v1;
            int v_2 = t_triple.v2;
            //parent->printInfo();
            //child->printInfo();
            SeqHamiltonianState *child = new SeqHamiltonianState(parent);
            int ret = child->addTwoEdgestoPath(fewest_vertice, v_1, v_2);
            //CkPrintf("This child return %d, (%d:%d:%d)\n", ret, fewest_vertice, v_1, v_2);
            if(ret == -1)
            {
                delete child;
                continue;
            }else if(ret == 0)
            {
                delete child;
                //child->printSolution();
                solver->reportSolution();
#ifdef ONESOLUTION
                return 0;
#else
                continue;
#endif
            }else
            {
                if(child->ImpossibleDectection())
                {
                    delete child;
                    //child->resetToParent(parent);
                    continue;
                }
                child->incEdgesInPath(2);
                ret = child->recursiveSolver(solver);
#ifdef ONESOLUTION
                if(ret == 0)
                {
                    delete child;
                    return 0;
                }
#endif
                delete child;
                //child->resetToParent(parent);
            }

        }
    } //end else
    //delete this;
    return 1;
} 
#endif
SE_Register(HamiltonianStateBase, createInitialChildren, createChildren, parallelLevel, searchDepthLimit);
