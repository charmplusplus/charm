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

//Image combining reduction type: defined in liveViz.C.
extern CkReductionMsg *imageCombine(int nMsg,CkReductionMsg **msgs);
extern CkReduction::reducerType imageCombineReducer;
extern void vizReductionHandler(void *r_msg);
extern CkReductionMsg *allocateImageMsg(const liveVizRequest &req,const CkRect &r, byte **imgDest);
void liveVizInitComplete(void *rednMessage);
extern CkCallback clientGetImageCallback;

//The liveVizGroup is only used to set lv_config on every processor.
class liveVizGroup : public Group {
public:
	liveVizGroup(const liveVizConfig &cfg) {
		lv_config=cfg;
		contribute(0,0,CkReduction::sum_int,CkCallback(liveVizInitComplete));
	}
};

#endif
