#include "TorusRouting.h"

// Look at bluegene paper for more details

int TorusRouting::selectRoute(int c,int d,int numP,Topology *top,Packet *p,map<int,int> & Bufsize) {
	int *next; next = top->next;
	int longestSize = -1,offset;
        int xdiff,ydiff,zdiff,xdiff2,ydiff2,zdiff2,nextPort=-1,select_torus=0,bubble=-1,rand[3];
        Position dst,pos; dst.init(d); pos.init(c);

	if(c == d) return numP;

        xdiff = dst.x - pos.x; ydiff = dst.y - pos.y ; zdiff = dst.z - pos.z;   
	xdiff2 = (netLength-(int)abs(xdiff)); ydiff2 = (netHeight-(int)abs(ydiff)); zdiff2 = (netWidth-(int)abs(zdiff));


	if(config.adaptiveRouting) {
		if(xdiff) {  
			//  Given source and destination, which direction ( + or - ) should we chose ?
			if(xdiff > 0) offset = (X_POS*config.switchVc); else offset = (X_NEG*config.switchVc);
			// Since we are simulating  a torus, we can use wraparound if it's shorter
        		if(xdiff2 < (int)abs(xdiff)) { offset = (((offset/config.switchVc)+3)%6)*config.switchVc; }
			// Among 3 of the  4 VC in the given direction, select max avail buffer.
			for(int i=0;i<(config.switchVc-1);i++)  
				if(longestSize < Bufsize[offset+i]) {
					longestSize = Bufsize[offset+i];
					nextPort = offset/config.switchVc;
				}
			// bubble should contain the vc number of BUBBLE_VC ( 4th vc) among
			// the ports, so that it has max avail buffer among BUBBLE_VC
			// bubble vc is used ONLY if the other 3 vc are full
			// Also, even after using bubble vc, space for one full packet should
			// be available. This is to avoid deadlock
			bubble = (Bufsize[config.switchVc-1+offset] > longestSize)? (config.switchVc-1+offset):-1;	
			rand[0] = offset/config.switchVc;// To be used when any vc is unable to fulfill the request.
		} else rand[0] = (ydiff)?ydiff:zdiff;

	// Comments for y and z direction  are the same as x
		// Advantage of duplicating code is that topology can lie between mesh and torus
		if(ydiff) {  
			if(ydiff > 0) offset = (Y_POS*config.switchVc); else offset = (Y_NEG*config.switchVc);
        		if(ydiff2 < (int)abs(ydiff)) { offset = (((offset/config.switchVc)+3)%6)*config.switchVc; }
			for(int i=0;i<(config.switchVc-1);i++)  
				if(longestSize < Bufsize[offset+i]) {
					longestSize = Bufsize[offset+i];
					nextPort = offset/config.switchVc;
				}
			if(bubble != -1)
			if(Bufsize[bubble*config.switchVc+config.switchVc-1] < Bufsize[config.switchVc-1+offset])
				bubble = (Bufsize[config.switchVc-1+offset] > longestSize)? (config.switchVc-1+offset) : -1;	

			rand[1] = offset/config.switchVc;
		} else rand[1] = (zdiff)?zdiff:xdiff;

		if(zdiff) {  
			if(zdiff > 0) offset = (Z_POS*config.switchVc); else offset = (Z_NEG*config.switchVc);
        		if(zdiff2 < (int)abs(zdiff)) { offset = (((offset/config.switchVc)+3)%6)*config.switchVc; }
			for(int i=0;i<(config.switchVc-1);i++)  
				if(longestSize < Bufsize[offset+i]) {
					longestSize = Bufsize[offset+i];
					nextPort = offset/config.switchVc;
				}
			if(bubble != -1)
			if(Bufsize[bubble*config.switchVc+config.switchVc-1] < Bufsize[config.switchVc-1+offset])
				bubble = (Bufsize[config.switchVc-1+offset] > longestSize) ? (config.switchVc-1+offset) : -1;	
			rand[2] = offset/config.switchVc;
		} else rand[2] = (xdiff)?xdiff:ydiff;

		if(longestSize == -1) {  // else nextPort is valid, return port with highest available buffer
			if(bubble == -1) {
				nextPort = rand[p->hdr.routeInfo.dst % 3];  // this is a dummy case
			} else
				nextPort = bubble/config.switchVc;  // return highest available buffer among bubble vc's
		}
		CkAssert(nextPort < 6);	
	} else {
        if(xdiff) { if(xdiff > 0) nextPort = X_POS; else nextPort = X_NEG; 
        if(xdiff2 < (int)abs(xdiff)) { nextPort = (nextPort+3)%6; }
	}
        else if(ydiff) { if(ydiff > 0) nextPort = Y_POS; else nextPort = Y_NEG; 
        if(ydiff2 < (int)abs(ydiff)) { nextPort = (nextPort+3)%6; }
	}
        else if(zdiff) { if(zdiff > 0) nextPort = Z_POS; else nextPort = Z_NEG; 
        if(zdiff2 < (int)abs(zdiff)) { nextPort = (nextPort+3)%6; }
	}   // Assume that self node won't be called
	}
	CkAssert(nextPort < 6);	
        return nextPort;
}

int TorusRouting::expectedTime(int s,int d,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops) {
        Position src,pos;  src.init(s);pos.init(d);
        POSE_TimeType extra,expected,xdist,ydist,zdist;
	xdist = netLength-abs(pos.x-src.x);
	ydist = netHeight-abs(pos.y-src.y);
	zdist = netWidth-abs(pos.z-src.z);

	if(xdist > abs(pos.x-src.x))  xdist = abs(pos.x-src.x);
	if(ydist > abs(pos.y-src.y))  ydist = abs(pos.y-src.y);
	if(zdist > abs(pos.z-src.z))  zdist = abs(pos.z-src.z);
	
        *hops = xdist+ydist+zdist;
        expected = *hops * config.switchC_Delay + (POSE_TimeType)(len/config.switchC_BW) + START_LATENCY + 2*CPU_OVERHEAD;
        extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}

int TorusRouting::convertOutputToInputPort(int id,Packet *p,int numP) {
	int port = p->hdr.portId;
	if(port == 6) return 6;
	return((port+3)%6);
}
