#include "HammingDistance.h"
// Simple algorithm based on hamming distance

int HammingDistance::selectRoute(int c,int d,int numP,Topology *top,Packet *p,map<int,int> & Bufsize) {
	int *next; next = top->next;
        unsigned int xorCurAndDst,portid=0,checkBit=1,nextNode;
        xorCurAndDst = c ^ d;
	if(c == d) return numP;

//	CkPrintf("Routing Current %d prev %d portid %d numP %d xorCurAndDst %d checkBit %d\n",
//		c,p->hdr.prev_src,p->hdr.portId,numP,xorCurAndDst,checkBit);
        while(!(xorCurAndDst & checkBit)) checkBit *=2;
                                                                                                                                                             
        nextNode = c ^ checkBit;
        while((portid < numP) && (nextNode != next[portid++]));
        if((portid == numP) && (next[numP-1] != nextNode))
                CkAssert("Hypercube routing algorithm received incorrect packet\n");
        portid--;

        return portid;
}

int HammingDistance::loadTable(Packet *p,int numP) {
	int current = p->hdr.routeInfo.dst;
	int mask,xorResult,port;

	for(int dst=0;dst<config.numNodes;dst++) {
        mask = 0x01; 
        xorResult = current ^ dst;
        port = 0;
       	if(!xorResult) port = numP;
	else
        while(!(xorResult & mask)) { mask *= 2; port ++; }

	p->hdr.nextPort[dst] = port;
	}
	if(p->hdr.routeInfo.dst == p->hdr.src) return -1;
	return 1;
}

int HammingDistance::getNextSwitch(int switchIndex) {
	return switchIndex; 
	// Beautiful property, verify it by drawing figure
}
	
void HammingDistance::populateRoutes(Packet *p,int numP) {
	int current,dst,mask,i,n,xorResult,port;

	current = p->hdr.src;
	dst = p->hdr.routeInfo.dst;
	i = 0; mask = 0x01;  n = numP; port = 0;

        xorResult = current ^ dst;
	while(n--) {
        if(xorResult & mask) {
                p->hdr.nextPort[i++] = port;
        }
         mask *= 2; port ++;
	}
		p->hdr.nextPort[i] = numP; // At the destination, just put into the processor reception FIFO
}

int HammingDistance::expectedTime(int s,int d,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops) {
        unsigned int hammingCode,i=0;
        POSE_TimeType extra,expected;
        hammingCode = s ^ d;  *hops = 0;
        while(i++ <  (sizeof(int)*8)) {
                if(hammingCode & 0x01) (*hops)++;  hammingCode = hammingCode >> 1;
        }
        expected = *hops * config.switchC_Delay + (POSE_TimeType)(len/config.switchC_BW) + START_LATENCY + 2*CPU_OVERHEAD;
        extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}

int HammingDistance::convertOutputToInputPort(int c,Packet *p,int numP) {
	int portid = p->hdr.portId;
	if(portid == numP) return numP;
        unsigned int xorCurAndDst,checkBit=1,nextNode;
	portid = 0;
        xorCurAndDst = c ^ (p->hdr.prev_src);
//	CkPrintf("Current %d prev %d portid %d numP %d xorCurAndDst %d checkBit %d\n",
//		c,p->hdr.prev_src,p->hdr.portId,numP,xorCurAndDst,checkBit);
        while(!(xorCurAndDst & checkBit)) { checkBit *=2; portid++; }

        if((portid == numP))
                CkAssert("Hypercube convertOutputToInputPort algorithm received incorrect packet\n");
        return portid;
}	
