#ifndef __ROUTINGALGORITHM_H
#define __ROUTINGALGORITHM_H
#include "../Main/BgSim_sim.h"
#define SELECT_ANY_PORT -1

class RoutingAlgorithm {
        public:
        virtual int selectRoute(int current,int dst,int numP,int *next){}
        virtual int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops){}
	virtual int selectRoute(int,int,const Packet *){}
	virtual int convertOutputToInputPort(int)=0;
};
#endif
