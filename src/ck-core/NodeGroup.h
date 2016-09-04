#ifndef _NODEGROUP_H
#define _NODEGROUP_H


#include "CkNodeReductionMgr.h"


//A NodeGroup that contribute to reductions
class NodeGroup : public CkNodeReductionMgr {
  protected:
    contributorInfo reductionInfo;//My reduction information
  public:
    CmiNodeLock __nodelock;
    const int thisIndex;
    NodeGroup();
    NodeGroup(CkMigrateMessage* m):CkNodeReductionMgr(m),thisIndex(CkMyNode()) { __nodelock=CmiCreateLock(); }
    
    ~NodeGroup();
    inline const CkGroupID &ckGetGroupID(void) const {return thisgroup;}
    inline CkGroupID CkGetNodeGroupID(void) const {return thisgroup;}
    virtual bool isNodeGroup() { return true; }

    virtual void pup(PUP::er &p);
    virtual void flushStates() {
    	CkNodeReductionMgr::flushStates();
        reductionInfo.redNo = 0;
    }

    CK_REDUCTION_CONTRIBUTE_METHODS_DECL
    void contributeWithCounter(CkReductionMsg *msg,int count);
};


#endif
