/*
  Data types, function prototypes,  etc. used within liveViz.
 */
#ifndef __UIUC_CHARM_LIVEVIZ_IMPL_H
#define __UIUC_CHARM_LIVEVIZ_IMPL_H
#include "liveViz.h"

//Silly globals declared in liveViz.C
extern liveVizConfig lv_config;
extern CProxy_liveVizGroup lvG;


// Moved here from liveViz.C so that liveVizPoll.C can see it too.
class imageHeader {
public:
	liveVizRequest req;
	CkRect r;
	imageHeader(const liveVizRequest &req_,const CkRect &r_)
		:req(req_), r(r_) {}
};

extern void vizReductionHandler(CkReductionMsg * msg);
extern CkCallback clientGetImageCallback;

//The liveVizGroup is only used to set lv_config on every processor.
class liveVizGroup : public CBase_liveVizGroup {
public:
	liveVizGroup(const liveVizConfig &cfg, CkCallback c) {
		lv_config=cfg;
		clientGetImageCallback=c;
		contribute(CkCallback(CkReductionTarget(liveVizGroup, initComplete), thisProxy[0]));
	}
	liveVizGroup(CkMigrateMessage *m): CBase_liveVizGroup(m) {}
	void initComplete() {
		liveViz0Init(lv_config);
	}
	void combine(CkReductionMsg * msg) {
		vizReductionHandler(msg);
	}
	void pup(PUP::er &p) {
		lv_config.pupNetwork(p);
		p | clientGetImageCallback;

#ifdef CMK_MEM_CHECKPOINT
		if (p.isUnpacking() && p.isRestarting() && CmiMyPe() == 0)
			liveViz0Init(lv_config);
#endif
	}
};

#endif
