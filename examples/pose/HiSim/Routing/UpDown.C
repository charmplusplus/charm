#include "UpDown.h"

int isDirectionChanged(int & src,int & dst,int & nodeRangeStart,int & nodeRangeEnd)
{
        if( (src >= nodeRangeStart) && (src < nodeRangeEnd) && (dst < nodeRangeEnd) && (dst >= nodeRangeStart) )
                return 1;
        else
                return 0;
}

int UpDown::selectRoute(int c,int d,int numP,Topology *top,Packet *p,map<int,int> & Bufsize) {
//Do Static Routing for now
	int goDown,nextP,dstOffset,fanout=(config.numP/2),portId;
	portId = p->hdr.portId;

	goDown = isDirectionChanged(p->hdr.src,p->hdr.routeInfo.dst,top->nodeRangeStart,top->nodeRangeEnd);
		
	if((!goDown) && (portId < fanout)) 
		nextP = (portId%fanout)+fanout;
	else {
		dstOffset = p->hdr.routeInfo.dst-top->nodeRangeStart;
		nextP = ((dstOffset*fanout)/(top->nodeRangeEnd-top->nodeRangeStart));
	}	
		return nextP;
}

int UpDown::expectedTime(int s,int d,int ovt,int origovt,int len,int *hops) {
      	int fanout = (config.numP/2),numhops=0; 
        int dimSize = config.numNodes/fanout,tmp=config.numNodes;
	*hops = 0; 

        while(tmp > 1) { numhops+=2; tmp /= fanout; }
        numhops --;

        while(dimSize > 1) {
                if((s/dimSize) != (d/dimSize)) {
                        break;
                }
                dimSize /= fanout;
                numhops -= 2;
        }

        int expected = numhops * config.switchC_Delay + (int)(len/config.switchC_BW) + START_LATENCY;
        int extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
	*hops = numhops;
        return extra;
}

int UpDown::convertOutputToInputPort(int portid) {
	int fanout = (config.numP/2);
	return (((portid+fanout)%config.numP));    // Hehehe
}
