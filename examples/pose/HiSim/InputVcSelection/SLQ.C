#include "SLQ.h"
int SLQ::selectInputVc(map<int,int> & Bufsize,map<int,int> &requested,map<int, vector <Header> > &inBuffer,const int globalVc) {
	int i=0,longestQ=NO_VC_AVAILABLE,longestLen=config.switchBufsize,start;
	start = (globalVc/config.switchVc)*config.switchVc;

	for(i=start;i<(start+config.switchVc);i++) {
                if(inBuffer[i].size() && (!(requested[i]))) {
                        if(longestLen >= Bufsize[i]) {
                                longestLen = Bufsize[i];
                                longestQ = i;
                        }
                }
        }
	return longestQ;
}
