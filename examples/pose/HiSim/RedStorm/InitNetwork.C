//#include "InitNetwork.h"
#include "BgSim_sim.h"
#include "../Topology/Mesh3D.h"

void InitNetwork(MachineParams *mp) {

        unsigned int mapPE,portid,switchP,counter;
	int nodeid,i;

	NetInterfaceMsg *nic;
	InputBufferMsg *inpbuf;
	ChannelMsg *chan;
        switchP = 6;
	
        mp->config->InputBufferStart= mp->config->nicStart+mp->BGnodes; 
        mp->config->ChannelStart = mp->config->InputBufferStart + (switchP+1)*(mp->BGnodes);
	

        for(i=mp->config->nicStart;i< mp->config->nicStart + mp->BGnodes ;i++) {
                nic = new NetInterfaceMsg(i,mp->config->InputBufferStart+(i-mp->config->nicStart)*(switchP+1),switchP);
                nic->Timestamp(0);
                mapPE = mp->procs*(i-mp->config->nicStart)/mp->BGnodes;
                (*(CProxy_NetInterface *) &POSE_Objects)[i].insert(nic,mapPE);
        }

	counter = mp->config->ChannelStart;
        for(i=mp->config->InputBufferStart;i< mp->config->InputBufferStart+(switchP+1)*mp->BGnodes;i++) {
                nodeid = (i-mp->config->InputBufferStart)/(switchP+1); portid = (i-mp->config->InputBufferStart)%(switchP+1);
		if(((i-mp->config->InputBufferStart)%(switchP+1)) == 0) {
			if(i != mp->config->InputBufferStart)
			counter += (switchP+1);
		}
                inpbuf = new InputBufferMsg(i,portid,nodeid,switchP,counter);
                inpbuf->Timestamp(0);
                mapPE = mp->procs*(i-mp->config->InputBufferStart)/((switchP+1)*mp->BGnodes);
                (*(CProxy_InputBuffer *) &POSE_Objects)[i].insert(inpbuf,mapPE);
        }

        for(i=mp->config->ChannelStart;i< mp->config->ChannelStart+(switchP+1)*mp->BGnodes;i++) {
                nodeid = (i-mp->config->ChannelStart)/(switchP+1); portid = (i-mp->config->ChannelStart)%(switchP+1);
                chan = new ChannelMsg(i,portid,nodeid,switchP);
                chan->Timestamp(0);
                mapPE = mp->procs*(i-mp->config->ChannelStart)/((switchP+1) * mp->BGnodes);
                (*(CProxy_Channel *) &POSE_Objects)[i].insert(chan,mapPE);
        }

}

void getMyTopology(Topology **topology) 
{
   *topology = new Mesh3D;
}
