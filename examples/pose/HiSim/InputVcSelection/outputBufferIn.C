#include "outputBufferIn.h"
int outputBufferIn::selectInputVc(map<int,int> & Bufsize,map<int,int> & requested,map<int, vector <Header> > &inBuffer, const int lastOutputVc) {

	int maxsize = 0,vc = NO_VC_AVAILABLE,outPort = lastOutputVc/config.switchVc;	

// We can provide some sort of QOS for particular virtual channels here
	if(requested[outPort])
		return NO_VC_AVAILABLE;

	for(int i=outPort*config.switchVc;i<((outPort+1)*config.switchVc);i++) {
                if(inBuffer[i].size()) {
		if(maxsize < inBuffer[i].size()) {
			maxsize = inBuffer[i].size();
			vc = i;
		}
                }
        }

	return vc;
}
