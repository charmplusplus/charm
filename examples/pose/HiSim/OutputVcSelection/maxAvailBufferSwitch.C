#include "maxAvailBufferSwitch.h"
int maxAvailBufferSwitch::selectOutputVc(map<int,int> & Bufsize,map<int,int> & mapVc,const Packet *h) {
        int avail=0,vc=NO_VC_AVAILABLE,nextVcId;

	for(int i=0;i<config.switchVc;i++) {
		nextVcId = h->hdr.portId*config.switchVc+i;
		if((mapVc[nextVcId] == IDLE) && (Bufsize[nextVcId] >= h->hdr.routeInfo.datalen)) {
			return i;
		}
	}				
                return NO_VC_AVAILABLE;
}
