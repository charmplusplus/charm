#include "UpDown.h"

int isDirectionChanged(int & src,int & dst,int & nodeRangeStart,int & nodeRangeEnd)
{
        if( (src >= nodeRangeStart) && (src < nodeRangeEnd) && (dst < nodeRangeEnd) && (dst >= nodeRangeStart) )
                return 1;
        else
                return 0;
}

int UpDown::selectRoute(int nodeRangeStart,int nodeRangeEnd,Packet *p) {
//Do Static Routing for now
	int goDown,nextP,dstOffset,fanout=config.fanout;
	goDown = isDirectionChanged(p->hdr.src,p->hdr.routeInfo.dst,nodeRangeStart,nodeRangeEnd);
		
	if(!goDown) 
		return ((p->hdr.portId%fanout)+fanout);
	else {
		dstOffset = p->hdr.routeInfo.dst-nodeRangeStart;
		nextP = ((dstOffset*fanout)/(nodeRangeEnd-nodeRangeStart));
		return nextP;
	}	
}

int UpDown::expectedTime(int s,int d,int ovt,int origovt,int len,int *hops) {
      	int fanout = config.fanout; 
        int dimSize = config.numNodes/fanout,tmp=config.numNodes;
	*hops = 0;

        while(tmp > 1) { hops+=2; tmp /= fanout; }
        (*hops) --;

        while(dimSize > 1) {
                if((s/dimSize) != (d/dimSize)) {
                        break;
                }
                dimSize /= config.fanout;
                (*hops) -= 2;
        }

        int expected = *hops * config.switchC_Delay + len/config.switchC_BW + START_LATENCY;
        int extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}
