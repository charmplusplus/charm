#include "DirectionOrdered3D.h"

int DirectionOrdered3D::selectRoute(int c,int d,int numP,int *next) {
        int xdiff,ydiff,zdiff,zdiff2,nextPort,select_torus=0;
        Position dst,pos; dst.init(d); pos.init(c);

        xdiff = dst.x - pos.x; ydiff = dst.y - pos.y ; zdiff = dst.z - pos.z;   zdiff2 = (netWidth-(int)abs(zdiff));
        if(xdiff) { if(xdiff > 0) nextPort = X_POS; else nextPort = X_NEG; }
        else if(ydiff) { if(ydiff > 0) nextPort = Y_POS; else nextPort = Y_NEG; }
        else if(zdiff) { if(zdiff > 0) nextPort = Z_POS; else nextPort = Z_NEG; }   // Assume that self node won't be called
        if((zdiff2 < (int)abs(zdiff)) && ((nextPort == Z_POS) || (nextPort == Z_NEG))) { select_torus = 3; }
//      if(select_torus) { nextPort = ((nextPort+3)%6); }
        return nextPort;
}

int DirectionOrdered3D::expectedTime(int s,int d,int ovt,int origovt,int len,int *hops) {
        Position src,pos;  src.init(s);pos.init(d);
        int extra,expected;
        *hops = abs(pos.x-src.x)+abs(pos.y-src.y)+abs(pos.z-src.y);
        expected = *hops * config.switchC_Delay + len/config.switchC_BW + START_LATENCY;
        extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}
