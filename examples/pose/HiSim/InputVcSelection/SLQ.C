#include "SLQ.h"
int SLQ::selectInputVc(map<int,int> & Bufsize,map<int,int> &requested,map<int, vector <Header> > &inBuffer,const int ignore) {
	int i=0,longestQ=NO_VC_AVAILABLE,longestLen=-1;
	for(i=0;i<config.switchVc;i++) {
                if(inBuffer[i].size()   &&  (!(requested[i]))) {
                        if(longestLen < Bufsize[i]) {
                                longestLen = Bufsize[i];
                                longestQ = i;
                        }
                }
        }
	return longestQ;
}
