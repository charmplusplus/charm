#include "maxAvailBuffer.h"
int maxAvailBuffer::selectOutputVc(map<int,int> & Bufsize,const Packet *h) {
        int avail=0,vc=NO_VC_AVAILABLE;

        for(int i=0;i<config.switchVc;i++) { 
	    if(avail <= Bufsize[i]) { 
		vc = i; 
		avail = Bufsize[i] ; 
	    } 
	}

        if(h->hdr.routeInfo.datalen <= avail)
                return vc;
        else
                return NO_VC_AVAILABLE;
}

