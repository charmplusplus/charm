#include "RoundRobin.h"
int RoundRobin::selectInputVc(map<int,int> & Bufsize,map<int,int> & requested,map<int, vector <Header> > &inBuffer, const int ignore) {
	int i=0;
	for(i=0;i<config.switchVc;i++) {
		roundRobin = (roundRobin+1)%config.switchVc;
                if(inBuffer[roundRobin].size()   &&  (!requested[roundRobin])) {
			return roundRobin;
                }
        }
	return NO_VC_AVAILABLE;
}
