#include "MainRouting.h"

class DirectionOrdered3D : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP,int *next);
        int expectedTime(int src,int dst,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops);
};
