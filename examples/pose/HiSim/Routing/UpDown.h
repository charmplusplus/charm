#include "MainRouting.h"

class UpDown : public RoutingAlgorithm{
	public:
        int selectRoute(int start,int end,Packet *);
        int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops);
};
