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
			if(xdiff > 0) offset = (X_POS*config.switchVc); else offset = (X_NEG*config.switchVc);
        		if(xdiff2 < (int)abs(xdiff)) { offset = (((offset/config.switchVc)+3)%6)*config.switchVc; }
			for(int i=0;i<(config.switchVc-1);i++)  
				if(longestSize < Bufsize[offset+i]) {
					longestSize = Bufsize[offset+i];
					nextPort = offset/config.switchVc;
				}
			bubble = (Bufsize[config.switchVc-1+offset] > longestSize)? (config.switchVc-1+offset):-1;	
			rand[0] = offset/config.switchVc;
		} else rand[0] = (ydiff)?ydiff:zdiff;

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
				nextPort = bubble;  // return highest available buffer among bubble vc's
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
        return nextPort;
}

int TorusRouting::expectedTime(int s,int d,int ovt,int origovt,int len,int *hops) {
        Position src,pos;  src.init(s);pos.init(d);
        int extra,expected,xdist,ydist,zdist;
	xdist = netLength-abs(pos.x-src.x);
	ydist = netHeight-abs(pos.y-src.y);
	zdist = netWidth-abs(pos.z-src.z);

	if(xdist > abs(pos.x-src.x))  xdist = abs(pos.x-src.x);
	if(ydist > abs(pos.y-src.y))  ydist = abs(pos.y-src.y);
	if(zdist > abs(pos.z-src.z))  zdist = abs(pos.z-src.z);
	
        *hops = xdist+ydist+zdist;
        expected = *hops * config.switchC_Delay + (int)(len/config.switchC_BW) + START_LATENCY;
        extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}

int TorusRouting::convertOutputToInputPort(int port) {
	if(port == 6) return 6;
	return((port+3)%6);
}
