#include "TorusRouting.h"

int TorusRouting::selectRoute(int c,int d,int numP,int *next) {
        int xdiff,ydiff,zdiff,xdiff2,ydiff2,zdiff2,nextPort,select_torus=0;
        Position dst,pos; dst.init(d); pos.init(c);

	if(c == d) return numP;

        xdiff = dst.x - pos.x; ydiff = dst.y - pos.y ; zdiff = dst.z - pos.z;   
	xdiff2 = (netLength-(int)abs(xdiff)); ydiff2 = (netHeight-(int)abs(ydiff)); zdiff2 = (netWidth-(int)abs(zdiff));

        if(xdiff) { if(xdiff > 0) nextPort = X_POS; else nextPort = X_NEG; 
        if(xdiff2 < (int)abs(xdiff)) { nextPort = (nextPort+3)%6; }
	}
        else if(ydiff) { if(ydiff > 0) nextPort = Y_POS; else nextPort = Y_NEG; 
        if(ydiff2 < (int)abs(ydiff)) { nextPort = (nextPort+3)%6; }
	}
        else if(zdiff) { if(zdiff > 0) nextPort = Z_POS; else nextPort = Z_NEG; 
        if(zdiff2 < (int)abs(zdiff)) { nextPort = (nextPort+3)%6; }
	}   // Assume that self node won't be called
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
        expected = *hops * config.switchC_Delay + len/config.switchC_BW + START_LATENCY;
        extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}

int TorusRouting::convertOutputToInputPort(int port) {
	if(port == 6) return 6;
	return((port+3)%6);
}
