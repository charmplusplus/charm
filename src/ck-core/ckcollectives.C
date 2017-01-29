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

typedef unsigned char            byte;

extern void CkPackMessage(envelope **pEnv);
extern void CkUnpackMessage(envelope **pEnv);

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
       collMgr[node].scatterSendAll((CkMessage *)newMsg);
    }
}

void CkCollectiveMgr::scatterSendAll(CkMessage *msg){
       //!Send to all elements
       envelope *env = UsrToEnv(msg);
       void *newMsg;
       CkScatterWrapper w;
       getScatterInfo(msg, &w);

       void* msgBuf = ((CkMarshallMsg *)msg)->msgBuf;
       w.unpackInfo(msgBuf);
       int *disp = (int *) w.disp;
       CkArrayIndex *dest = (CkArrayIndex *) w.dest;
       int *cnt = (int *) w.cnt;

       int tailsize = ((char *)&cnt[w.ndest]) - ((char *)disp);
       int msgsize = env->getTotalsize();
       env->setTotalsize(msgsize - tailsize);

       int ep = ((CkArrayMessage *)msg)->array_ep_bcast();       
       CkPrintf("[%d/%d] Received Scatter msg, ndest: %d, ep: %d \n", CkMyNode(), CkMyPe(), w.ndest, ep);
       CkArray *array = CProxy_ArrayBase(w.aid).ckLocalBranch();
       for (int i=0; i<w.ndest; i++) {
            CProxyElement_ArrayBase ap(w.aid, dest[i]);
            newMsg = createScatterMsg(msg, w, i);
            //CkPrintf("[%d/%d] CkMulticastMgr::recvScatterMsg, sending msg# %d \n", CkMyNode(), CkMyPe(), i);
            ap.ckSend((CkArrayMessage *)newMsg,ep,0);
       }
       env->setTotalsize(msgsize);
       CkFreeMsg(msg);
}



#include "CkCollectives.def.h"

