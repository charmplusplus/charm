#include "HammingDistance.h"

int HammingDistance::selectRoute(int c,int d,int numP,int *next) {
        unsigned int xorCurAndDst,portid=0,checkBit=1,nextNode;
        xorCurAndDst = c ^ d;
        while(!(xorCurAndDst & checkBit)) checkBit *=2;
                                                                                                                                                             
        nextNode = c ^ checkBit;
        while((portid < numP) && (nextNode != next[portid++]));
        if((portid == numP) && (next[numP-1] != nextNode))
                CkAssert("Hypercube routing algorithm received incorrect packet\n");
        portid--;
        return portid;
}

int HammingDistance::expectedTime(int s,int d,int ovt,int origovt,int len,int *hops) {
        unsigned int hammingCode,i=0;
        int extra,expected;
        hammingCode = s ^ d;  *hops = 0;
        while(i++ <  (sizeof(int)*8)) {
                if(hammingCode & 0x01) (*hops)++;  hammingCode = hammingCode >> 1;
        }
        expected = *hops * config.switchC_Delay + (int)(len/config.switchC_BW) + START_LATENCY;
        extra = (ovt-origovt) - expected;
        if(extra < 0) extra = 0;
        return extra;
}
