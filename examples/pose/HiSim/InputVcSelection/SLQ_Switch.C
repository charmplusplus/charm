#include "SLQ_Switch.h"
int SLQ_Switch::selectInputVc(map<int,int> & Bufsize,map<int,int> & requested,map<int, vector <Header> > & inBuffer,const int outVcId) {

	vector <Header>::iterator headBuf;

	int i=0,longestQ=NO_VC_AVAILABLE,longestLen=-1;
	for(i=0;i<(config.switchVc*config.fanout*2);i++) {
		if(inBuffer[i].size()) {
		headBuf = inBuffer[i].begin();
                if((!requested[i]) && (headBuf->portId == (outVcId/config.switchVc))) {
                        if(longestLen < Bufsize[i]) {
                                longestLen = Bufsize[i];
                                longestQ = i;
                        }
                }
		}
        }
	return longestQ;
}
