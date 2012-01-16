/* searchEngineAPI.h
 *
 *  Nov 3
 *
 * author: Yanhua Sun
 */
#ifndef __SEARCHENGINEAPI__
#define __SEARCHENGINEAPI__
#include "cmipool.h"
/*   framework for search engine */

#include <vector>
#include <list>
#include <algorithm>

using namespace std;

extern int initial_grainsize;
extern int verticesNum;

class HamiltonianStateBase : public StateBase{
public:
    int maxUnionId;
    int  *UnionId;
    int edgesInPath;

    HamiltonianStateBase(int maxId)
    {
        maxUnionId = maxId;
        edgesInPath = 0;
    }

    HamiltonianStateBase(int maxId, int level)
    {
        maxUnionId = maxId;
        edgesInPath = 0;
    }
    void initialize()
    {
        int offset = sizeof(HamiltonianStateBase);
        UnionId =  (int*)((char*)this + offset); 
    }
    void resetUnionId()
    {
        for(int i=0; i<3*verticesNum; i++)
        {
            UnionId[i] =  -1;
        }
    }
    void copy(HamiltonianStateBase* hm)
    {
        maxUnionId = hm->maxUnionId;
        for(int i=0; i<3*verticesNum; i++)
        {
            UnionId[i] =  hm->UnionId[i];
        }
        edgesInPath = hm->edgesInPath;
    }

    void incEdgesInPath(int m)
    {
        edgesInPath += m;
    }
    void printInfo()
    {
        for(int i=0; i<verticesNum; i++)
        {
            CkPrintf(" %d:%d %d %d\n", i, UnionId[i*3], UnionId[i*3+1], UnionId[i*3+2]);
        }
    }

    /* one edge already in the path */
    int addOneEdgetoPath(int fewest_vertice, int branchVertice)
    {
        int selectId = UnionId[fewest_vertice *3];
        /* add this edge  */
        UnionId[3*fewest_vertice + 2] = branchVertice;
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
                        return -1;
                    }
                }
                return 0;
            }
        }
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
            UnionId[v_1*3] = v2_unionid;   
        }else if(v2_unionid == -1 && v1_unionid > -1)
        {
            uionIdSet = v1_unionid;
            UnionId[v_1*3+2] = fewest_vertice;    
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
                        return -1;
                    }
                }
                UnionId[3*fewest_vertice] = v1_unionid;
                return 0;
            }
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
        return 1;
    }


    void printSolution()
    {
        int previous = 0;
        int next = UnionId[1];
        int current = next;
        int numInPath = 1;

        CkPrintf("Parallel Solution:\n");
        CkPrintf("============================\n");
        CkPrintf("1 ");
        while (next != 0)
        {
            numInPath++;
            CkPrintf(" %d ", next+1);
            current = next;
            if( UnionId[next*3+1] == previous)
            {
                next = UnionId[next*3+2];
            }else
                next = UnionId[next*3+1];
            previous = current;
        }
        CkPrintf(" 1 \n\nTotal vertices:%d; vertices in path:%d\n", verticesNum, numInPath );
    }
};

#endif
