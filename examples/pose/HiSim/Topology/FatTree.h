#ifndef __FATTREE_H
#define __FATTREE_H

#include "MainTopology.h"
class FatTree : public Topology {
public:
int level,curLevelId,curLevelDimOffset,nodeRangeStart;
int nodeRangeEnd,outNextPortUp,outNextPortDown;

	FatTree();
        void getNeighbours(int nodeid,int numP);
        int getNext(int portid,int nodeid,int numP);
	int getStartNode();
	int getEndNode();
};
#endif
