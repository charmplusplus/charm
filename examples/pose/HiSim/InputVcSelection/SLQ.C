#include "SLQ.h"
int SLQ::selectInputVc(map<int,int> & Bufsize,map<int,int> &requested,map<int, vector <Header> > &inBuffer,const int globalVc) {
	int i=0,longestQ=NO_VC_AVAILABLE,longestLen=config.switchBufsize,start;
	vector <Header>::iterator headOfBuf; 
	start = (globalVc/config.switchVc);

	// Should have another strategy for selecting among single port. So I would have a iterator going through each port
	// and calling this additional function which does similar thing for the vc in a port

	for(i=0;i<config.switchVc*6;i++) {
                if((inBuffer[i].size()) && (!(requested[i]))) {
			headOfBuf = inBuffer[i].begin();
			if(headOfBuf->portId == start) {	
                        if(longestLen >= Bufsize[i]) {
                                longestLen = Bufsize[i];
                                longestQ = i;
                        }
			}
                }
        }
	return longestQ;
}
