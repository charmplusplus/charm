//#include "InitNetwork.h"
#include "BgSim_sim.h"
#include "../Topology/FatTree.h"
#include "../Routing/UpDown.h"
#include "../InputVcSelection/SLQ_Switch.h"
#include "../OutputVcSelection/maxAvailBufferSwitch.h"

void InitNetwork(MachineParams *mp) {

        unsigned int mapPE,portid,switchP,counter,numNodes,nnodes,numLevels;
	int nodeid,i,nicStart=config.nicStart,switchStart,chanStart,fanout,numSwitches,numChans;

	NetInterfaceMsg *nic;
	SwitchMsg *switchbuf;
	ChannelMsg *chan;
        fanout = (mp->config->numP)/2;
	nnodes = 1; numLevels = 0;
	numNodes = mp->config->numNodes;

        while(nnodes < numNodes) {numLevels++;nnodes*=fanout;}
        if(numNodes > nnodes) {numNodes = nnodes*fanout; numLevels++;}

	CkPrintf("fanout %d numP %d \n",fanout,mp->config->numP);	
		
	mp->config->numNodes = numNodes = nnodes ;	
       	switchP = mp->config->numP;
	numSwitches = (numNodes/fanout)*numLevels;
	// Basically each switch except the toplevel have numP channels
	numChans = switchP*numSwitches ;  
	// Should actually subtract numNodes above, but then getNextChannel interface becomes complicated 
	 
	switchStart = mp->config->switchStart= mp->config->nicStart+numNodes; 
        chanStart = mp->config->ChannelStart = mp->config->switchStart + numSwitches;
	
        for(i=nicStart;i< nicStart + numNodes ;i++) {
                nic = new NetInterfaceMsg(i,switchStart+ ((i-nicStart)/fanout),switchP);
                nic->Timestamp(0);
                mapPE = mp->procs*(i-nicStart)/numNodes;
                (*(CProxy_NetInterface *) &POSE_Objects)[i].insert(nic,mapPE);
        }

        for(i=switchStart;i< switchStart+numSwitches;i++) {
                switchbuf = new SwitchMsg(i,switchP);
                switchbuf->Timestamp(0);
                mapPE = mp->procs*(i-switchStart)/numSwitches;
                (*(CProxy_Switch *) &POSE_Objects)[i].insert(switchbuf,mapPE);
        }

	int chanid;
        for(i=chanStart;i< chanStart+numChans;i++) {
		chanid = i-chanStart;
		portid = chanid%switchP;
		nodeid = chanid/switchP;

                chan = new ChannelMsg(i,portid,nodeid,switchP);
                chan->Timestamp(0);
                mapPE = mp->procs*(i-chanStart)/numChans;
                (*(CProxy_Channel *) &POSE_Objects)[i].insert(chan,mapPE);
        }

}

void initializeNetwork(Topology **topology,RoutingAlgorithm **routing,InputVcSelection ** invc,OutputVcSelection **outvc)
{
   *topology = new FatTree;
   *routing = new UpDown;
   *invc = new SLQ_Switch;
   *outvc = new maxAvailBufferSwitch;
}

void initializeNetwork(Topology **topology,RoutingAlgorithm **routing)
{
   *topology = new FatTree;
   *routing = new UpDown;
}
