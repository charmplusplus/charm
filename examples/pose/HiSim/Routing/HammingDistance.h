#include "MainRouting.h"

class HammingDistance : public RoutingAlgorithm{
	public:
        int selectRoute(int current,int dst,int numP,Topology *,Packet *,map<int,int> &);
        int expectedTime(int src,int dst,POSE_TimeType ovt,POSE_TimeType origovt,int len,int *hops);
	int convertOutputToInputPort(int,Packet *,int);
	void populateRoutes(Packet *,int);
	int loadTable(Packet *,int);
	int getNextSwitch(int);
};
                                                                                                                                                             

