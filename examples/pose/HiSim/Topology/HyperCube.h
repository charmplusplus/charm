#ifndef __HYPERCUBE_H
#define  __HYPERCUBE_H
#include "MainTopology.h"

class HyperCube : public Topology {
	public:
	HyperCube();
        void getNeighbours(int nodeid,int numP);
        int getNext(int portid,int nodeid,int numP);
        int getNextChannel(int,int,int);
        int getStartPort(int,int);
        int getStartVc();
        int getStartSwitch(int);
};

#endif
