#include "MainRouting.h"

class HammingDistance : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP,int *);
        int expectedTime(int src,int dst,int ovt,int origovt,int len,int *hops);
};
                                                                                                                                                             

