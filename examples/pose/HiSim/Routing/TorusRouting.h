#include "MainRouting.h"

class TorusRouting : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP,int *next);
        int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops);
	int convertOutputToInputPort(int);
};
