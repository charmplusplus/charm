//#include "InitNetwork.h"
#include "BgSim_sim.h"
#include "../Topology/Mesh3D.h"

void InitNetwork(MachineParams *mp) {

        unsigned int mapPE,portid,switchP;
	int nodeid,i;

	NetInterfaceMsg *nic;
	SwitchMsg *sbuf;
	ChannelMsg *chan;
        switchP = 6;
	
        mp->config->switchStart= mp->config->nicStart+mp->BGnodes; 
        mp->config->ChannelStart = mp->config->switchStart + mp->BGnodes;
	

        for(i=mp->config->nicStart;i< mp->config->nicStart + mp->BGnodes ;i++) {
                nic = new NetInterfaceMsg(i,mp->config->switchStart+(i-mp->config->nicStart),switchP);
                nic->Timestamp(0);
                mapPE = mp->procs*(i-mp->config->nicStart)/mp->BGnodes;
                (*(CProxy_NetInterface *) &POSE_Objects)[i].insert(nic,mapPE);
        }

        for(i=mp->config->switchStart;i< mp->config->switchStart+mp->BGnodes;i++) {
                sbuf = new SwitchMsg(i,switchP);
                sbuf->Timestamp(0);
                mapPE = mp->procs*(i-mp->config->switchStart)/(mp->BGnodes);
                (*(CProxy_Switch *) &POSE_Objects)[i].insert(sbuf,mapPE);
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
