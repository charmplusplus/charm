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

int HammingDistance::expectedTime(int s,int d,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops) {
        unsigned int hammingCode,i=0;
        POSE_TimeType extra,expected;
        hammingCode = s ^ d;  *hops = 0;
        while(i++ <  (sizeof(int)*8)) {
                if(hammingCode & 0x01) (*hops)++;  hammingCode = hammingCode >> 1;
        }
        expected = *hops * config.switchC_Delay + (POSE_TimeType)(len/config.switchC_BW) + START_LATENCY + CPU_OVERHEAD;
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
