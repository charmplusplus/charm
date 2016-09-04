#ifndef _GROUP_CALLBACK_H
#define _GROUP_CALLBACK_H

#include "pup.h"
#include "IrrGroup.h"

class CkGroupCallbackMsg;
class CkGroupInitCallback : public IrrGroup {
public:
	CkGroupInitCallback(void);
	CkGroupInitCallback(CkMigrateMessage *m):IrrGroup(m) {}
	void callMeBack(CkGroupCallbackMsg *m);
	void pup(PUP::er& p){ IrrGroup::pup(p); }
};

#endif
