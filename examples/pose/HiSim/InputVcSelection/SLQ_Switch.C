#include "SLQ_Switch.h"
int SLQ_Switch::selectInputVc(map<int,int> & Bufsize,map<int,int> & requested,map<int, vector <Header> > & inBuffer,const int outVcId) {

	vector <Header>::iterator headBuf;

	int i=0,longestQ=NO_VC_AVAILABLE,shortestLen=config.switchBufsize;
	for(i=0;i<(config.switchVc*config.numP);i++) {
		if(inBuffer[i].size()) {
		headBuf = inBuffer[i].begin();
                if((!requested[i]) && (headBuf->portId == (outVcId/config.switchVc))) {
                        if(shortestLen >= Bufsize[i]) {
                                shortestLen = Bufsize[i];
                                longestQ = i;
                        }
                }
		}
        }
	return longestQ;
}
