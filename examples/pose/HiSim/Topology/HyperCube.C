#include "HyperCube.h"
#include "InitNetwork.h"
#include ROUTING_FILE
#include OUTPUT_VC_FILE
#include INPUT_VC_FILE

HyperCube::HyperCube() {
	routingAlgorithm = new ROUTING_ALGORITHM;
	outputVcSelect = new OUTPUT_VC_SELECTION;
	inputVcSelect = new INPUT_VC_SELECTION;	
}

void HyperCube::getNeighbours(int nodeid,int numP) {
        unsigned int flipPosition=1,checkNode,portid=0;
        next = (int *)malloc(sizeof(int)*numP);
        for(int i=0;(i<(sizeof(int)*8)) && (portid < numP);i++) {
                checkNode = nodeid ^ flipPosition;
                flipPosition *= 2;
                if(checkNode < config.numNodes) { next[portid++] = checkNode;
        //CkPrintf("Node %d port %d next %d\n",nodeid,portid,checkNode);
        }
        }
}
                                                                                                                                                             
int HyperCube::getNext(int portid,int nodeid,int numP) {
unsigned int p,pow2,y;
int r,id,c,i,nextPort,checkBit;
float con = log(2);
                                                                                                                                                             
nextPort = 0;
checkBit = nodeid ^ next[portid];
while(checkBit = (checkBit >> 1)) nextPort ++;
                                                                                                                                                             
p=0;c=config.numNodes; id = next[portid]; y = 0;
i = (int)((log(c+0.9))/con);
                                                                                                                                                             
while(i > 0) {
pow2 = (int)pow(2,i);
r = c - pow2;
                                                                                                                                                             
if(id <= 0) break;
                                                                                                                                                             
if(r >= 0) {
if(id <= r)
        p += (id * (i+1+y));
else
        p += (r * (i+1+y));
                                                                                                                                                             
if(id > r) {
if(id < pow2 )
        p += ((id-r) * (i+y));
else
        p += ((pow2-r)* (i+y));
}
                                                                                                                                                             
}
                                                                                                                                                             
id -= pow2; c  -= pow2;
y++; i--;
}
                                                                                                                                                             
        CkAssert(portid < numP);
        return(config.InputBufferStart + next[portid] + p + nextPort); // Add next[portid] to account for injection port
}
