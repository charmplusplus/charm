#include "Mesh3D.h"
#include "InitNetwork.h"
#include ROUTING_FILE
#include OUTPUT_VC_FILE
#include INPUT_VC_FILE

Mesh3D::Mesh3D() {
routingAlgorithm = new ROUTING_ALGORITHM;
outputVcSelect = new OUTPUT_VC_SELECTION;
inputVcSelect = new INPUT_VC_SELECTION;
}

void Mesh3D::getNeighbours(int nodeid,int numP) {
        Position pos;
        pos.init(nodeid);
        next = (int *) malloc(sizeof(int)*numP);
        pos.getNeighbours(next); // Should generate a 3d torus. But the torus capability is just not used.
}

int Mesh3D::getNext(int portid,int nodeid,int numP) {
        return(config.InputBufferStart + next[portid]*(numP+1) + (portid+numP/2)%numP);
}
