#include "MainRouting.h"

class TorusRouting : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP,Topology *top,Packet *p,map<int,int> & Bufsize);
        int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops);
	int convertOutputToInputPort(int);
};
