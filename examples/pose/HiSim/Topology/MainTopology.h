#ifndef __TOPOLOGY_H
#define __TOPOLOGY_H
#include "../Main/BgSim_sim.h"

class Topology {
        public:
        int *next;
	int nodeRangeStart,nodeRangeEnd;
        virtual void getNeighbours(int nodeid,int numP)=0;
        virtual int getNext(int portid,int nodeid,int numP) = 0;
	virtual int getNextChannel(int,int) = 0;
	virtual int getStartPort(int id) = 0;
	virtual int getStartVc() = 0;
	virtual int getStartSwitch(int id) = 0;
	virtual int getStartNode(){}
	virtual int getEndNode(){}
        int initialize();
};
#endif
