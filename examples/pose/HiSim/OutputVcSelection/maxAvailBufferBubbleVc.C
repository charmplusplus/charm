#include "maxAvailBufferBubbleVc.h"
int maxAvailBufferBubbleVc::selectOutputVc(map<int,int> & Bufsize,const Packet *h) {

       int avail=0,vc=NO_VC_AVAILABLE;
        for(int i=0;i<config.switchVc-1;i++) { 
		if(avail <= Bufsize[i]) {
			 vc = i; avail = Bufsize[i] ; 
		} 
	}

        if(h->hdr.routeInfo.datalen <= avail)
                return vc;
        else {
                //Check for Bubble VC
                if(Bufsize[config.switchVc-1] < h->hdr.routeInfo.datalen)
                return NO_VC_AVAILABLE;
                else
                return (config.switchVc-1);
        }
}
