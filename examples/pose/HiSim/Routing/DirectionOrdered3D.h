#include "MainRouting.h"

class DirectionOrdered3D : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP,int *next);
        int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops);
};
