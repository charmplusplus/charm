#ifndef __FATTREE_H
#define __FATTREE_H

#include "MainTopology.h"
class FatTree : public Topology {
public:
int level,curLevelId,curLevelDimOffset;
int outNextPortUp,outNextPortDown;

	FatTree();
        void getNeighbours(int nodeid,int numP);
        int getNext(int portid,int nodeid,int numP);
	int getNextChannel(int,int);
	int getStartPort(int);
	int getStartVc();
	int getStartSwitch(int);
	int getStartNode();
	int getEndNode();
};
#endif
