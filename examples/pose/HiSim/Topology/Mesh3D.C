#include "Mesh3D.h"
#include "InitNetwork.h"

Mesh3D::Mesh3D() {
}

void Mesh3D::getNeighbours(int nodeid,int numP) {
        Position pos;
        pos.init(nodeid);
        next = (int *) malloc(sizeof(int)*(numP+1));
        pos.getNeighbours(next); // Should generate a 3d torus. But the torus capability is just not used.
	next[numP]= nodeid;
//	CkPrintf("Node %d  ",nodeid);
//	for(int i=0;i<6;i++) CkPrintf("next[%d] = %d ",i,next[i]); 
//	CkPrintf("\n");
}

int Mesh3D::getNext(int portid,int switchid,int numP) {
//	CkPrintf("For port %d node %d next is %d \n",portid,switchid-config.switchStart,next[portid]);
	if(portid == numP)
		return(config.nicStart + next[portid]);
	else
        	return(config.switchStart + next[portid]);
}

int Mesh3D::getNextChannel(int portid,int switchid) {
	int numP = 6; 

	if(portid < numP)
		return(config.ChannelStart + (numP+1)*next[portid]+portid);
	else
		return(config.ChannelStart + (switchid-config.switchStart)*(numP+1)+numP);
}

int Mesh3D::getStartPort(int id) {
        return(6);  // There are 7 input ports
}

int Mesh3D::getStartVc() { // Assume no end to end flow control
        return 0;
}

int Mesh3D::getStartSwitch(int id) {
        return(config.switchStart +  id);
}
