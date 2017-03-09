/*
 *  Charm++ support for collective operations
 *
 *  written by Vipul Harsh,   vharsh2@illinois.edu
 *  on 1/2017
 *
 *  features:
            Scatterv - node level point to point
 * */

#include "charm++.h"
#include "envelope.h"
#include "register.h"

#include "ckcollectives.h"
#include "spanningTreeStrategy.h"

#include <map>
#include <vector>

#define DEBUGF(x)  // CkPrintf x;

typedef unsigned char byte;

void _ckCollectivesInit(void){}

void CkCollectiveMgr::ckScatter(void *msg, CkArrayID aid){
    CProxy_CkCollectiveMgr collMgr(thisgroup);
    CkScatterWrapper w;
    getScatterInfo(msg, &w);
    CkArrayIndex *dest = (CkArrayIndex *) w.dest;
    w.aid = aid;
    CkArray *array = CProxy_ArrayBase(aid).ckLocalBranch();
    std::map<int, std::vector<int> > elemBins;
    //group into node bins
    for (int i=0; i<w.ndest; i++) {
        int pe = array->lastKnown(dest[i]);
        int node = CkNodeOf(pe);
        elemBins[node].push_back(i);
    }
    //create msg for each bin
    for(auto it: elemBins){
       void *newMsg = createScatterMsg(msg, w, it.second);
       int node = it.first;
       collMgr[node].scatterSendAllEntry((CkMessage *)newMsg);
    }
}


void CkCollectiveMgr::ckScatterTree(void *msg, CkArrayID aid){
    CProxy_CkCollectiveMgr collMgr(thisgroup);
    CkScatterWrapper w;
    getScatterInfo(msg, &w);
    w.aid = aid;
    DEBUGF(("[%d] CkCollectiveMgr::ckScatter, msgsize: %d \n", CkMyPe(), UsrToEnv(msg)->getTotalsize()));
    PUP::toMem p((void *)(((CkMarshallMsg *)msg)->msgBuf));
    p|w;
    ckScatterSpanningTree((CkMessage *)msg, w, false);
    CkFreeMsg(msg);
}


/*
 * strips a message to exclude the scatterv buffers at the end
 *
 **** Assumptions ****
 * w should be unpacked
 */
void CkCollectiveMgr::stripScatterMsg(CkMessage *msg, CkScatterWrapper &w){
       envelope *env = UsrToEnv(msg);
       void* msgBuf = ((CkMarshallMsg *)msg)->msgBuf;
       int *cnt = (int *) w.cnt;
       //Strip msg to exclude scatter buffers
       int tailsize = ((char *)&cnt[w.ndest]) - ((char *)w.disp);
       int msgsize = env->getTotalsize();
       env->setTotalsize(msgsize - tailsize);
}


void CkCollectiveMgr::ckScatterSpanningTreeEntry(CkMessage *msg){
    CkScatterWrapper w;
    getScatterInfo(msg, &w);
    w.unpackInfo(((CkMarshallMsg *)msg)->msgBuf);
    stripScatterMsg(msg, w);
    ckScatterSpanningTree(msg, w, false);
    CkFreeMsg(msg);
}


/*
 * Invocation on a spanning tree node for scatterv
 *
 ***** Assumptions *****
 * - msg is not tied to the buffer
 * - w is unpacked
*/
void CkCollectiveMgr::ckScatterSpanningTree(CkMessage *msg, CkScatterWrapper &w, bool free){
    CProxy_CkCollectiveMgr collMgr(thisgroup);
    CkArrayIndex *dest = (CkArrayIndex *) w.dest;
    std::map<int, std::vector<int> > elemBins;
    elemBins[CkMyNode()];
    CkArray *array = CProxy_ArrayBase(w.aid).ckLocalBranch();
    //node bins
    for (int i=0; i<w.ndest; i++) {
        int pe = array->lastKnown(dest[i]);
        int node = CkNodeOf(pe);
        elemBins[node].push_back(i);
    }
    
    DEBUGF(("[%d/%d] ckScatterSpanningTree, w.ndest: %d, w.buf: %p, w.disp: %p, w.dest: %p, w.cnt: %p\n", CkMyNode(), CkMyPe(), w.ndest, w.buf, w.disp, w.dest, w.cnt));
   
    if(elemBins.size() == 1){ //terminal msg
        int node = elemBins.begin()->first;
        CkAssert(node == CkMyNode());
        //send to all elements
        scatterSendAll(msg, w, false, false);
    }
    else if(elemBins[CkMyNode()].size() > 0){
        //send to local elements
        scatterSendAll(msg, w, true, false); //no need to create another messsage
    }

    CkVec<int> mySubTreePEs;
    mySubTreePEs.reserve(elemBins.size());
    //First PE in my subtree should be me, the tree root(required by the spanning tree builder)
    mySubTreePEs.push_back(CkNodeFirst(CkMyNode()));
    // Identify the child PEs in the tree, i.e. PEs participating in scatterv
    for (std::map<int, std::vector<int> >::iterator itr = elemBins.begin();
       itr != elemBins.end(); ++itr){
         if(itr->first != CkMyNode()){
            //CkPrintf("Pushing mySubTreePEs: %d \n", CkNodeFirst(itr->first)); 
            mySubTreePEs.push_back(CkNodeFirst(itr->first));
         }
    }

    int num = mySubTreePEs.size() - 1, numchild = 0;
    int factor = 10; //branching factor
    numchild = num<factor?num:factor;
 
    //If there are any children, build a spanning tree
    if (numchild) 
    {
        //Build the next generation of the spanning tree rooted at my PE
        int *peListPtr = mySubTreePEs.getVec();
        topo::SpanningTreeVertex *nextGenInfo;
        nextGenInfo = topo::buildSpanningTreeGeneration(peListPtr,peListPtr + mySubTreePEs.size(),numchild);
        numchild = nextGenInfo->childIndex.size();

        //Send message to each direct child to setup its subtree
        for (int i=0; i < numchild; i++)
        {
            //Determine the indices of the first and last PEs in this branch of my sub-tree
            int childStartIndex = nextGenInfo->childIndex[i], childEndIndex;
            if (i < numchild-1)
                childEndIndex = nextGenInfo->childIndex[i+1];
            else
                childEndIndex = mySubTreePEs.size();

            //Find the total number of chare array elements on this subtree
            int ndest = 0;
            std::vector<int> indices; 
            for (int j = childStartIndex; j < childEndIndex; j++){
                int node = CkNodeOf(mySubTreePEs[j]);
                indices.insert(indices.end(), elemBins[node].begin(), elemBins[node].end());
                ndest += elemBins[node].size();
            }
            DEBUGF(("[%d/%d] child: %d, ndest: %d\n", CkMyNode(), CkMyPe(), i, ndest));
            //create msg for each child
            void *newMsg = createScatterMsg(msg, w, indices);
            int destnode = CkNodeOf(mySubTreePEs[childStartIndex]);
            collMgr[destnode].ckScatterSpanningTreeEntry((CkMessage *)newMsg);
        }
        delete nextGenInfo;
    }
    if(free)
        CkFreeMsg(msg);
}



void CkCollectiveMgr::scatterSendAllEntry(CkMessage *msg){
       //!Send to all elements
       envelope *env = UsrToEnv(msg);
       DEBUGF(("[%d/%d]scatterSendAllEntry scatter-msg size: %d, \n", CkMyNode(), CkMyPe(), UsrToEnv(msg)->getTotalsize()));
       CkScatterWrapper w;
       getScatterInfo(msg, &w);
       void* msgBuf = ((CkMarshallMsg *)msg)->msgBuf;
       w.unpackInfo(msgBuf);
       stripScatterMsg(msg, w);
       scatterSendAll(msg, w, false, true);
}


/*
 * Delivers scatterv msg to all destinations or all local destinations
 *
 * local: send only to chares that are local to this node
 ***** Assumptions *****
 * - msg is not tied to the buffer
 * - w is unpacked
*/

void CkCollectiveMgr::scatterSendAll(CkMessage *msg, CkScatterWrapper &w, bool local, bool free){
       //!Send to all elements
       envelope *env = UsrToEnv(msg);
       void *newMsg;
       CkArrayIndex *dest = (CkArrayIndex *) w.dest;
       int ep = ((CkArrayMessage *)msg)->array_ep_bcast();       
       DEBUGF(("[%d/%d] scatterSendAll: scatter-msg  size: %d, ep: %d, ndest: %d send only to local elems?: %d\n", CkMyNode(), CkMyPe(), UsrToEnv(msg)->getTotalsize(), w.ndest, ep, local));
       CkArray *array = CProxy_ArrayBase(w.aid).ckLocalBranch();
       for (int i=0; i<w.ndest; i++) {
            int pe = array->lastKnown(dest[i]);
            int node = CkNodeOf(pe);
            if(node == CkMyNode() || !local){
                CProxyElement_ArrayBase ap(w.aid, dest[i]);
                newMsg = createScatterMsg(msg, w, i);
                DEBUGF(("[%d/%d] CkCollectiveMgr::recvScatterMsg, sending msg# %d \n", CkMyNode(), CkMyPe(), i));
                ap.ckSend((CkArrayMessage *)newMsg,ep,0);
            }
       }
       if(free)
            CkFreeMsg(msg);
}



#include "CkCollectives.def.h"

