#ifndef __MESH3D_H
#define __MESH3D_H

#include "MainTopology.h"
class Mesh3D : public Topology {
	public:
	Mesh3D();
        void getNeighbours(int nodeid,int numP);
        int getNext(int portid,int nodeid,int numP);
};
#endif
