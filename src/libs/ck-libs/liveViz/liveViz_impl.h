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

extern void vizReductionHandler(void *r_msg);
void liveVizInitComplete(void *rednMessage);
extern CkCallback clientGetImageCallback;

//The liveVizGroup is only used to set lv_config on every processor.
class liveVizGroup : public CBase_liveVizGroup {
public:
	liveVizGroup(const liveVizConfig &cfg) {
		lv_config=cfg;
		contribute(0,0,CkReduction::sum_int,CkCallback(liveVizInitComplete));
	}
	liveVizGroup(CkMigrateMessage *m): CBase_liveVizGroup(m) {}
	void pup(PUP::er &p) {
		lv_config.pupNetwork(p);
	}
};

#endif
