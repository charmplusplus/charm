#include "MainRouting.h"

class UpDown : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP, Topology *top,Packet *,map<int,int> & Bufsize);
        int expectedTime(int src,int dst,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops);
	int convertOutputToInputPort(int id,Packet *,int);
};
