//#include "InitNetwork.h"
#include "BgSim_sim.h"
#include "../Topology/HyperCube.h"
#include "../Routing/HammingDistance.h"
#include "../OutputVcSelection/maxAvailBuffer.h"  
#include "../InputVcSelection/RoundRobin.h"

void InitNetwork(MachineParams *mp) {

        unsigned int diff,counter,firstBitPos,totalP,*ports,*portindex,nnodes,nextNode,k,i,mapPE,portid;
	int nodeid,nicStart,switchStart,chanStart;

	NetInterfaceMsg *nic;
	SwitchMsg *sbuf;
	ChannelMsg *chan;
	Topology **topology;

        totalP = 0;
	nicStart = mp->config->nicStart;
        switchStart = mp->config->switchStart = mp->config->nicStart+mp->BGnodes;
        ports = (unsigned int *)malloc(sizeof(int)*mp->BGnodes);
        portindex = (unsigned int *)malloc(sizeof(int)*mp->BGnodes);

        diff = mp->BGnodes; counter = 0; nnodes = 1; portindex[0] = 0;
        while(diff = diff >> 1)  { counter ++; nnodes *= 2; }

	// The code here is mostly for the future incomplete hypercube. 
	// The implementation is based on the fact that an incomplete hypercube
	// is composed of two parts. one complete sub-hypercube and one incomplete one
	// The complete one has number of ports corresponding to the log2(sub-hypercube)
	// and some nodes have an extra port as they connect to nodes in incomplete sub-hypercube
	// Now takes this logic recursively for the incomplete sub-hypercube ( now sub-sub complete and incomplete)
	// and so on.

        for(i=0;i<mp->BGnodes;i++) {
        if(i < nnodes) {
                ports[i] = counter;
                if((i+nnodes) <= (mp->BGnodes-1)) ports[i]++;
        }
        else {
                diff = mp->BGnodes-i;  firstBitPos = 0;
                while(diff) { diff = diff >> 1; firstBitPos++; }
                ports[i] = firstBitPos;
        }

        totalP += (ports[i]+1);
        if(i!=0) portindex[i] = portindex[i-1]+ports[i-1]+1;
        }

        topology = (Topology **)malloc(sizeof(Topology *)* (mp->BGnodes));
        for(k=0;k<mp->BGnodes;k++)
                topology[k] = new HyperCube;

        chanStart = mp->config->ChannelStart = mp->config->switchStart + mp->BGnodes  ;

        for(i=nicStart;i< mp->config->switchStart;i++) {
                topology[i-nicStart]->getNeighbours(i-nicStart,ports[i-nicStart]);
                nic = new NetInterfaceMsg(i,switchStart+(i-nicStart),ports[i-nicStart]);
                nic->Timestamp(0);
                mapPE = mp->procs*(i-nicStart)/mp->BGnodes;

                (*(CProxy_NetInterface *) &POSE_Objects)[i].insert(nic,mapPE);
        }

        counter = 0; nodeid = -1;
        int counter2 = mp->config->ChannelStart;

        for(i=switchStart;i< mp->config->ChannelStart;i++) {
                if(!counter) {
                nodeid ++; counter = ports[nodeid]+1; portid = 0;
                if(nodeid)
                counter2 += (ports[nodeid-1]+1);
                }  // Adding 1 to include injection port

                sbuf = new SwitchMsg(i,ports[nodeid]);
                sbuf->Timestamp(0);
                mapPE = mp->procs*(nodeid-1)/mp->BGnodes;
                (*(CProxy_Switch *) &POSE_Objects)[i].insert(sbuf,mapPE);
                portid ++; counter--;
        }

        counter = 0; nodeid = -1;
        for(i=mp->config->ChannelStart;i< mp->config->ChannelStart+totalP;i++) {
                if(!counter) {  nodeid ++; counter = ports[nodeid]+1; portid = 0; }
                chan = new ChannelMsg(i,portid,nodeid,ports[nodeid]);
                chan->Timestamp(0);
                mapPE = mp->procs*(nodeid-1)/mp->BGnodes;
                (*(CProxy_Channel *) &POSE_Objects)[i].insert(chan,mapPE);
                portid ++; counter --;
        }


        free(ports); free(portindex);
        for(k=0;k<mp->BGnodes;k++) free(topology[k]);
        free(topology);
}

void initializeNetwork(Topology **topology,RoutingAlgorithm **routing,InputVcSelection ** invc,OutputVcSelection **outvc)
{
   *topology = new HyperCube;
   *routing = new HammingDistance;
   *invc = new RoundRobin;
   *outvc = new maxAvailBuffer; 

}

void initializeNetwork(Topology **topology,RoutingAlgorithm **routing)
{
   *topology = new HyperCube;
   *routing = new HammingDistance;
}
