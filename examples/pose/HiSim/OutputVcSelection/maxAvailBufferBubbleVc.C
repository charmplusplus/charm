#include "maxAvailBufferBubbleVc.h"
int maxAvailBufferBubbleVc::selectOutputVc(map<int,int> & Bufsize,const Packet *h) {

       int avail=0,vc=NO_VC_AVAILABLE,start;
	start = h->hdr.portId*config.switchVc;

	if(h->hdr.portId == 6) return 0; // Don't care for the sink

        for(int i=start;i<(start+config.switchVc-1);i++) { 
		if(avail <= Bufsize[i]) {
			 vc = i; avail = Bufsize[i] ; 
		} 
	}

	vc -= start;

        if(h->hdr.routeInfo.datalen <= avail)
                return vc;
        else {
                //Check for Bubble VC
                if(Bufsize[start+config.switchVc-1] < h->hdr.routeInfo.datalen)
                return NO_VC_AVAILABLE;
                else
                return (config.switchVc-1);
        }
}
