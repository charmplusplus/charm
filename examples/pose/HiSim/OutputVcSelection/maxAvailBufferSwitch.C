#include "maxAvailBufferSwitch.h"

// Return vcid within a port ( not like selectInputVc which returns within a switch )
int maxAvailBufferSwitch::selectOutputVc(map<int,int> & Bufsize,const Packet *h) {
        int avail=0,vc=NO_VC_AVAILABLE,nextVcId;

	for(int i=0;i<config.switchVc;i++) {
		nextVcId = h->hdr.portId*config.switchVc+i;
		if(Bufsize[nextVcId] >= h->hdr.routeInfo.datalen) {
			return i;
		}
	}				
                return NO_VC_AVAILABLE;
}
