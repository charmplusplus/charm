#ifndef _GROUP_H_
#define _GROUP_H_

#include "pup.h"
#include "CkReductionMgr.h"



//A group that can contribute to reductions
class Group : public CkReductionMgr
{
	contributorInfo reductionInfo;//My reduction information
 public:
    const int thisIndex;
	Group();
	Group(CkMigrateMessage *msg);
	virtual bool isNodeGroup() { return false; }
	virtual void pup(PUP::er &p);
	virtual void flushStates() {
		CkReductionMgr::flushStates();
		reductionInfo.redNo = 0;
 	}
	virtual void CkAddThreadListeners(CthThread tid, void *msg);

	CK_REDUCTION_CONTRIBUTE_METHODS_DECL
        CK_BARRIER_CONTRIBUTE_METHODS_DECL
};

#endif
