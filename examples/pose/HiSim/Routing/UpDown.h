#include "MainRouting.h"

class UpDown : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP, Topology *top,Packet *,map<int,int> & Bufsize);
        int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops);
	int convertOutputToInputPort(int);
};
