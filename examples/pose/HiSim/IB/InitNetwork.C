//#include "InitNetwork.h"
#include "BgSim_sim.h"
#include "../Topology/FatTree.h"

void InitNetwork(MachineParams *mp) {

        unsigned int mapPE,portid,switchP,counter,numNodes,nnodes,numLevels;
	int nodeid,i,nicStart=config.nicStart,switchStart,chanStart,fanout,numSwitches,numChans;

	NetInterfaceMsg *nic;
	SwitchMsg *switchbuf;
	ChannelMsg *chan;
        fanout = mp->config->fanout = 4;
	nnodes = 1; numLevels = 0;
	numNodes = mp->config->numNodes;

        while(nnodes < numNodes) {numLevels++;nnodes*=fanout;}
        if(numNodes > nnodes) {numNodes = nnodes*fanout; numLevels++;}
		
	mp->config->numNodes = numNodes;	
       	switchP = 2*fanout;
	numSwitches = (numNodes/fanout)*numLevels;
	numChans = switchP*numSwitches - (switchP/2)*(numNodes/fanout);  // Have to account for top level switches and bot nodes
 
	switchStart = mp->config->switchStart= mp->config->nicStart+numNodes; 
        chanStart = mp->config->ChannelStart = mp->config->switchStart + numSwitches;
	

        for(i=nicStart;i< nicStart + numNodes ;i++) {
		// Up is even number. Down is odd numbered channel
		// The channels go from switch to switch , covering each switch fully
                nic = new NetInterfaceMsg(i,chanStart+ ((i-nicStart)/fanout)*switchP*2 + 2*((i-nicStart)%fanout),switchP);
                nic->Timestamp(0);
                mapPE = mp->procs*(i-nicStart)/numNodes;
                (*(CProxy_NetInterface *) &POSE_Objects)[i].insert(nic,mapPE);
        }

        for(i=switchStart;i< switchStart+numSwitches;i++) {
                switchbuf = new SwitchMsg(i);
                switchbuf->Timestamp(0);
                mapPE = mp->procs*(i-switchStart)/numSwitches;
                (*(CProxy_Switch *) &POSE_Objects)[i].insert(switchbuf,mapPE);
        }

        for(i=chanStart;i< chanStart+numChans;i++) {
		portid = ((i-chanStart)%(2*switchP))/2;
		nodeid = (i-chanStart)/(2*switchP);
                chan = new ChannelMsg(i,portid,nodeid,switchP);
                chan->Timestamp(0);
                mapPE = mp->procs*(i-chanStart)/numChans;
                (*(CProxy_Channel *) &POSE_Objects)[i].insert(chan,mapPE);
        }

}

void getMyTopology(Topology **topology) 
{
   *topology = new FatTree;
}
