#include "SLQ.h"
int SLQ::selectInputVc(map<int,int> & Bufsize,map<int,int> &requested,map<int, vector <Header> > &inBuffer,const int globalVc) {
	int i=0,longestQ=NO_VC_AVAILABLE,shortestLen=config.switchBufsize,start;
	vector <Header>::iterator headOfBuf; 
	start = (globalVc/config.switchVc);
	int occupied[6],myvc;
	// Should have another strategy for selecting among single port. So I would have a iterator going through each port
	// and calling this additional function which does similar thing for the vc in a port

	for(i=0;i<6;i++) {
		occupied[i] = 0;
		for(int j=0;j<config.switchVc;j++) {
			myvc = i*config.switchVc+j;
			if(requested[myvc]) occupied[i]++;
		}
	}
	

	for(i=0;i<config.switchVc*6;i++) {
                if((inBuffer[i].size()) && (!(requested[i])) && (occupied[i/config.switchVc] < config.inputSpeedup)) {
			headOfBuf = inBuffer[i].begin();
			if(headOfBuf->portId == start) {	
                        if(shortestLen >= Bufsize[i]) {
                                shortestLen = Bufsize[i];
                                longestQ = i;
                        }
			}
                }
        }
	return longestQ;
}
