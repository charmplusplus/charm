#include "UpDown.h"

// Check if direction we need to go from up to down
int isDirectionChanged(int & src,int & dst,int & nodeRangeStart,int & nodeRangeEnd)
{
        if( (src >= nodeRangeStart) && (src < nodeRangeEnd) && (dst < nodeRangeEnd) && (dst >= nodeRangeStart) )
                return 1;
        else
                return 0;
}

void UpDown::populateRoutes(Packet *p,int numP) {
	int fanout = numP/2,n,src,dst,i=0,pktid;
	n = fanout; src = p->hdr.src; dst = p->hdr.routeInfo.dst; pktid = p->hdr.pktId;

	while(n <= config.numNodes)  { // Has to be perfect fat tree
		if((src/n) == (dst/n)) break;
		p->hdr.nextPort[i++] = (src+pktid)%fanout+fanout;
		n *= fanout;
	}

	while(n >= fanout) {
		p->hdr.nextPort[i++] = ((dst%n) * fanout)/n;
		dst = dst % (n/fanout);  
		n /= fanout;
	}
}
			
// Select by adaptive ( based on load ) or static routing
int UpDown::selectRoute(int c,int d,int numP,Topology *top,Packet *p,map<int,int> & Bufsize) {
	int goDown,nextP=-1,dstOffset,fanout=(config.numP/2),portId;
	int longestLen,shortestQ = -1,sum;

	portId = p->hdr.portId;

	goDown = isDirectionChanged(p->hdr.src,p->hdr.routeInfo.dst,top->nodeRangeStart,top->nodeRangeEnd);
		
	if((!goDown) && (portId < fanout)) {
		if(config.adaptive_routing) {

		if(!config.inputBuffering) {
			nextP = fanout; longestLen = 0;
			for(int j=0;j<config.switchVc;j++)
			longestLen += Bufsize[fanout*config.switchVc+j];

			for(int i=fanout;i<config.numP;i++) {
				sum = 0; 
				for(int j=0;j<config.switchVc;j++) sum += Bufsize[i*config.switchVc+j];
				if(longestLen < sum) {
					longestLen = sum;
					nextP = i;
				} 
			}	
//			CkPrintf("Switch %d nextPort is %d longestAvail is %d\n",c,nextP,longestLen);	
		}  else {
			nextP = p->hdr.routeInfo.dst%fanout+fanout;
			longestLen = Bufsize[nextP*config.switchVc];
			for(int i=fanout;i<config.numP;i++) {
				for(int j=0;j<config.switchVc;j++) {
					sum = Bufsize[i*config.switchVc+j];
					if(longestLen < sum) {
						longestLen = sum;
						nextP = i;
					}
				}
			}	
		}	
//			CkPrintf("Switch %d nextPort is %d longestAvail is %d\n",c,nextP,longestLen);	

               }
		else {  // You can have multiple static routing schemes
		nextP = (portId%fanout)+fanout;
		//nextP = (p->hdr.src%fanout)+fanout;
		}
	}
	else {
		dstOffset = p->hdr.routeInfo.dst-top->nodeRangeStart;
		nextP = ((dstOffset*fanout)/(top->nodeRangeEnd-top->nodeRangeStart));
	}
		CkAssert(nextP != -1);	
		return nextP;
}


int UpDown::expectedTime(int s,int d,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops) {
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

        POSE_TimeType expected = numhops * config.switchC_Delay + (POSE_TimeType)(len/config.switchC_BW) + START_LATENCY + 2*CPU_OVERHEAD;
        POSE_TimeType extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
	// The actual might be slightly lower in no-latency case than "expected" as
	//  round-off during integer/fp division occurs in transit (i.e POSE_invoke or elapse delays are rounded off)
	*hops = numhops;
        return extra;
}

int UpDown::convertOutputToInputPort(int id,Packet *p,int numP) {
	int portid = p->hdr.portId;
	int fanout = (config.numP/2);
	return (((portid+fanout)%config.numP));    // Hehehe
}
