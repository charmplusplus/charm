#ifndef _REDUCTION_H
#define _REDUCTION_H

#include "reduction.decl.h"
#include "megatest.h"

class reductionArray : public CBase_reductionArray {
 public:
	reductionArray() {}
	reductionArray(CkMigrateMessage *msg) {}
	void start(void);
};

class reductionGroup : public CBase_reductionGroup {
 public:
	reductionGroup() {}
	reductionGroup(CkMigrateMessage *msg) {}
	void start(void);
};

#endif
