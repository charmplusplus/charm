#include "HyperCube.h"
#include "InitNetwork.h"

HyperCube::HyperCube() {
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
        

// If routing is modified, we can support incomplete hypercubes
// The topology recognizes incomplete hypercubes for now. So the code is slightly complex                                                                                                                                                     
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
                                                                                                                                                             
        CkAssert(portid <= numP);
  //      return(config.InputBufferStart + next[portid] + p + nextPort); // Add next[portid] to account for injection port
        return(config.switchStart + next[portid] ); // Add next[portid] to account for injection port
}

int HyperCube::getNextChannel(int portid,int switchid,int numP) {
// Don't support incomplete hypercubes for now
        if(portid < numP)
                return(config.ChannelStart + (numP+1)*next[portid]+portid);
        else
                return(config.ChannelStart + (switchid-config.switchStart)*(numP+1)+numP);
}

int HyperCube::getStartPort(int id,int numP) {
        return(numP);  // There are 7 input ports
}

int HyperCube::getStartVc() { // Assume no end to end flow control
        return 0;
}

int HyperCube::getStartSwitch(int id) {
        return(config.switchStart +  id);
}
