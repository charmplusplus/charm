#include <vector>
#include <list>

using namespace std;

extern CkVec<int> inputGraph; 
class mypair{

public:
    int verticeIndex;
    int aliveEdges;
    bool reverse; 
    mypair(int v, int e, bool b)
    {
        verticeIndex = v;
        aliveEdges = e;
        reverse = b;
    }
    mypair(int v, int e)
    {
        verticeIndex = v;
        aliveEdges = e;
        reverse = false;
    }

    bool operator<(const mypair& right) const
    {
        if(reverse)
            return aliveEdges > right.aliveEdges;
        else
            return aliveEdges < right.aliveEdges;
    }

    int getVertice()
    {
        return verticeIndex;
    }
};

bool reverseComp(mypair& left, mypair& right)
{
    return left.aliveEdges > right.aliveEdges;
}
class triple{
public:
    int v1;
    int v2;
    int edgesSum;
    bool reverse; 
    
    triple(int i1, int i2, int s1, bool b)
    {
        v1 = i1;
        v2 = i2;
        edgesSum = s1;
        reverse = b;
    }
    triple(int i1, int i2, int s1)
    {
        v1 = i1;
        v2 = i2;
        edgesSum = s1;
        reverse = false;
    }

    bool operator<(const triple& right) const
    {
        if(reverse)
            return edgesSum > right.edgesSum;
        else
            return edgesSum < right.edgesSum;
    }

};

bool reverseComp(const triple& left, const triple& right)
{
    return left.edgesSum > right.edgesSum;
}
#ifdef USERSOLVER 
class SeqHamiltonianState {
public:
    int maxUnionId;
    int edgesInPath;
    vector<int> UnionId;
    vector< list<int> > aliveEdges;
    SeqHamiltonianState(HamiltonianStateBase* parallelstate )
    {
        maxUnionId = parallelstate->maxUnionId;
        edgesInPath = parallelstate->edgesInPath;
        aliveEdges.resize(verticesNum);
        UnionId.resize(3*verticesNum);
        for(int i=0; i<3*verticesNum; i++)
        {
            UnionId[i] = (parallelstate->UnionId)[i];
        }
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

        /* construct the alive edges */
        for(int i=0; i<verticesNum; i++)
        {
            int left_nb = UnionId[3*i+1];
            int right_nb =UnionId[3*i+2];
        /* only one edge is in the path set */
            if(left_nb >-1 && right_nb == -1)
            {
                aliveEdges[i].remove(left_nb);
            }else if(right_nb > -1) //this vertice already has two edges
            {
                list<int>::iterator iter;
                for(iter=aliveEdges[i].begin(); iter != aliveEdges[i].end(); iter++)
                {   /* remove edge  (i, *iter) */
                    aliveEdges[*iter].remove(i);
                }
                aliveEdges[i].clear();
            }
        }  /* finish construct the alive edges */
    }
    SeqHamiltonianState(SeqHamiltonianState *parent)
    {
        maxUnionId = parent->maxUnionId;
        edgesInPath = parent->edgesInPath;
        aliveEdges = parent->aliveEdges;
        UnionId = parent->UnionId;
    }
    void resetToParent(SeqHamiltonianState *parent)
    {
        maxUnionId = parent->maxUnionId;
        edgesInPath = parent->edgesInPath;
        aliveEdges = parent->aliveEdges;
        UnionId = parent->UnionId;
    }
    bool ImpossibleDectection()
    {
        for(int i=0; i<verticesNum; i++)
        {
            if(UnionId[3*i+1] == -1 && aliveEdges[i].size()<2 || UnionId[3*i+2] == -1 && aliveEdges[i].size() == 0)
                return true;
        }
        return false;
    }
    void incEdgesInPath(int m)
    {
        edgesInPath += m;
    }
    int recursiveSolver(Solver *solver);

    void setUnionId(int vertex, int id)
    {
        UnionId[3*vertex] = id;
    }
    int getUnionId(int vertex)
    {
        return UnionId[3*vertex];
    }
    bool NotInPath(int vertex)
    {
        return UnionId[3*vertex] == -1;
    }
    bool OneEdgeInPath(int vertex)
    {
        return UnionId[3*vertex+1] > -1 && UnionId[3*vertex+2]==-1;
    }
    bool TwoEdgesInPath(int vertex)
    {
            return UnionId[3*vertex+2] > -1;
        }
        void resetUnionId()
        {
            for(int i=0; i<3*verticesNum; i++)
            {
               UnionId[i] =  -1;
            }
        }
        void addOneEdge(int s, int d)
        {
            aliveEdges[s].push_back(d);
            aliveEdges[d].push_back(s);

        }
        /* s has left neighbor */
        void removeOneEdge(int s)
        {
           int d = UnionId[3*s+1]; 
           aliveEdges[s].remove(d);
           aliveEdges[d].remove(s);
        }


        void removeOneEdge(int s, int d)
        {
            aliveEdges[s].remove(d);
            aliveEdges[d].remove(s);
        }

        void removeAllNeighborEdges(int b)
        {
            /* delete all edges associated with branchVertice */
            for(list<int>::iterator l_iter = (aliveEdges[b]).begin(); l_iter != aliveEdges[b].end(); l_iter++)
            {
                aliveEdges[*l_iter].remove(b);
            }
            aliveEdges[b].clear();
        }
   
        bool detectImpossible(int a, int b)
        {
            /* delete all edges associated with branchVertice */
            for(list<int>::iterator l_iter = (aliveEdges[b]).begin(); l_iter != aliveEdges[b].end(); l_iter++)
            {
                if(*l_iter == a)
                    continue;
                int newsize = aliveEdges[*l_iter].size()-1;
                if(UnionId[*l_iter *3 +1] == -1 && newsize<2 || UnionId[*l_iter *3 +2] == -1 && newsize==0){
                    return true;
                }
            }
            return false;
        }
        void removeAllEdges(int b)
        {
            aliveEdges[b].clear();
        }

        int findFewestVertice()
        {
            int fewest_vertice = -1;
            int fewest_edges = verticesNum;
            for(int i=0; i<verticesNum; i++)
            {
                int edgeNum = aliveEdges[i].size();
                if( UnionId[3*i+2] > -1)
                    continue;
                else if(fewest_edges > edgeNum)
                {
                    fewest_vertice = i;
                    fewest_edges = edgeNum;
                }
            }
            return fewest_vertice;
        }

        bool conflict(int vertice)
        {
            int edgeNum = aliveEdges[vertice].size(); 
            if( (edgeNum == 0 && UnionId[3*vertice+2] == -1 ) || ( edgeNum == 1 && UnionId[3*vertice] == -1 ))
            {
                return true;
            }else
                return false;
        }

        void sortNeighbor(int fewest_vertice, vector<mypair>& neighborVertices)
        {
            list<int>::iterator iter; 
            for(iter=aliveEdges[fewest_vertice].begin(); iter != aliveEdges[fewest_vertice].end(); iter++)
            {
                mypair pair_1(*iter, aliveEdges[*iter].size());
                neighborVertices.push_back(pair_1);
            }
            /* from  lower to higher*/
        }

        /* one edge already in the path */
        int addOneEdgetoPath(int fewest_vertice, int branchVertice)
        {
            int selectId = UnionId[fewest_vertice *3];
            UnionId[3*fewest_vertice + 2] = branchVertice;
            /* add this edge  */
            if(UnionId[3*branchVertice] == -1) 
            {
                UnionId[3*branchVertice] = selectId;
                UnionId[3*branchVertice+1] = fewest_vertice;
            }else  //branch node has left neighbor
            {
                int branchId = UnionId[3*branchVertice];
                UnionId[3*branchVertice+2] = fewest_vertice;
                if(selectId < branchId)
                {
                    /* merge two sets */
                    for(int k=0; k<3*verticesNum; k+=3)
                    {
                        if(UnionId[k] == branchId)
                            UnionId[k] = selectId;
                    }
                }else if(selectId > branchId)
                {
                    /* merge two sets */
                    for(int k=0; k<3*verticesNum; k+=3)
                    {
                        if(UnionId[k] == selectId)
                            UnionId[k] = branchId;
                    }
                }else 
                {
                    for(int m=2; m< verticesNum*3; m+=3)
                    {
                        if(UnionId[m] == -1)
                        {
                            //undo the changes
                            UnionId[3*fewest_vertice + 2] = -1;
                            UnionId[3*branchVertice+2] = -1;
                            return -1;
                        }
                    }
                    return 0;
                }
                removeAllNeighborEdges(branchVertice);
            }
            removeAllNeighborEdges(fewest_vertice);
            return 1;
        }

        int addTwoEdgestoPath(int fewest_vertice, int v_1, int v_2)
        {

            int uionIdSet;
            int changeId = -1; 
            int v1_unionid = UnionId[3*v_1];
            int v2_unionid = UnionId[3*v_2];
            UnionId[3*fewest_vertice+1] = v_1;
            UnionId[3*fewest_vertice+2] = v_2;
            if(v1_unionid == -1 && v2_unionid == -1)
            {
                maxUnionId++;
                uionIdSet = maxUnionId;
                UnionId[v_1*3+1] = fewest_vertice;    
                UnionId[v_2*3+1] = fewest_vertice;    
                UnionId[v_1*3] = maxUnionId;    
                UnionId[v_2*3] = maxUnionId;    
            }else if(v1_unionid == -1 && v2_unionid > -1)
            {
                uionIdSet = v2_unionid;
                UnionId[v_1*3+1] = fewest_vertice;    
                UnionId[v_2*3+2] = fewest_vertice;    
                removeAllNeighborEdges(v_2);
                UnionId[v_1*3] = v2_unionid;    
            }else if(v2_unionid == -1 && v1_unionid > -1)
            {
                uionIdSet = v1_unionid;
                UnionId[v_1*3+2] = fewest_vertice;    
                removeAllNeighborEdges(v_1);
                UnionId[v_2*3+1] = fewest_vertice;    
                UnionId[v_2*3] = v1_unionid;    
            }else /*both the vertices already have one edge in the path*/
            {
                UnionId[v_1*3+2] = fewest_vertice;    
                UnionId[v_2*3+2] = fewest_vertice;    
                if(v1_unionid < v2_unionid)
                {
                    uionIdSet = v1_unionid;
                    changeId = v2_unionid;
                }else if(v1_unionid > v2_unionid)
                {
                    uionIdSet = v2_unionid;
                    changeId = v1_unionid;
                }else
                {
                    /* v1 and v2 and in the same path add fewest will connect them and form a cycle*/
                    /* test whether this is a solution or not */
                    /* only if all the vertices except the fewest are in the same path */
                    for(int m=2; m< verticesNum*3; m+=3)
                    {
                        if(UnionId[m] == -1)
                        {
                            UnionId[v_1*3+2] = -1;    
                            UnionId[v_2*3+2] = -1;    
                            UnionId[3*fewest_vertice+1] = -1;
                            UnionId[3*fewest_vertice+2] = -1;
                            return -1;
                        }
                    }
                    UnionId[3*fewest_vertice] = v1_unionid;
                    return 0;
                }
                removeAllNeighborEdges(v_1);
                removeAllNeighborEdges(v_2);
                /* merge the two paths involving v_1 v_2 */
                if(changeId > -1)
                {
                    for(int m=0; m< verticesNum*3; m+=3)
                    {
                        if(UnionId[m] == changeId)
                        {
                            UnionId[m] = uionIdSet;
                        }
                    }
                }
            }
            UnionId[3*fewest_vertice] = uionIdSet;
            removeAllNeighborEdges(fewest_vertice); 
            return 1;
        }

        void printSolution()
        {
            int previous = 0;
            int next = UnionId[1];
            int current = next;
            int numInPath = 1;

            CkPrintf(" Sequential Solution:\n");
            CkPrintf("============================\n");
            CkPrintf("0 ");
            while (next != 0)
            {
                numInPath++;
                CkPrintf(" %d ", next);
                current = next;
                if( UnionId[next*3+1] == previous)
                {
                    next = UnionId[next*3+2];
                }else
                    next = UnionId[next*3+1];
                previous = current;
            }
            CkPrintf(" 0 \n Total vertices:%d; vertices in path:%d\n", verticesNum, numInPath );
        }

        void printInfo()
        {
            int verticesInPath  = 0;
            printf("\n");
            for(int i=0; i<verticesNum; i++)
            {
                CkPrintf("%d: [%d:%d:%d] (alive edges: ", i, UnionId[3*i], UnionId[3*i+1], UnionId[3*i+2]);
                list<int>::iterator j;
                for(j=aliveEdges[i].begin(); j!= aliveEdges[i].end(); j++)
                    CkPrintf(" %d ", *j);
                CkPrintf("\n");
            }

            printf("\n");
        }
};

#endif
