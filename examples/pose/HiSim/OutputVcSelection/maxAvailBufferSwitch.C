#include "maxAvailBufferSwitch.h"

// Return vcid within a port ( not like selectInputVc which returns within a switch )
int maxAvailBufferSwitch::selectOutputVc(map<int,int> & Bufsize,const Packet *h) {
        int avail=0,vc=NO_VC_AVAILABLE,nextVcId,longestLen = 0;

	for(int i=0;i<config.switchVc;i++) {
		nextVcId = h->hdr.portId*config.switchVc+i;

		if(Bufsize[nextVcId] >= longestLen) {
			longestLen = Bufsize[nextVcId];
			vc = i;	
		}
	}			
	
		if(vc == NO_VC_AVAILABLE) 
         	        return NO_VC_AVAILABLE;
		else if(Bufsize[h->hdr.portId*config.switchVc+vc] >= h->hdr.routeInfo.datalen) 
			return vc;
		else
         	        return NO_VC_AVAILABLE;
}
