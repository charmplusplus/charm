//#include "InitNetwork.h"
#include "BgSim_sim.h"
#include "../Topology/HyperCube.h"

void InitNetwork(MachineParams *mp) {

        unsigned int diff,counter,firstBitPos,totalP,*ports,*portindex,nnodes,nextNode,k,i,mapPE,portid;
	int nodeid;

        Topology **topology;
	NetInterfaceMsg *nic;
	InputBufferMsg *inpbuf;
	ChannelMsg *chan;

        totalP = 0;
        mp->config->InputBufferStart = mp->config->nicStart+mp->BGnodes;
        ports = (unsigned int *)malloc(sizeof(int)*mp->BGnodes);
        portindex = (unsigned int *)malloc(sizeof(int)*mp->BGnodes);

        diff = mp->BGnodes; counter = 0; nnodes = 1; portindex[0] = 0;
        while(diff = diff >> 1)  { counter ++; nnodes *= 2; }

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


        mp->config->ChannelStart = mp->config->InputBufferStart + totalP ;

        counter = mp->config->InputBufferStart;
        for(i=mp->config->nicStart;i< mp->config->InputBufferStart;i++) {
                topology[i-mp->config->nicStart]->getNeighbours(i-mp->config->nicStart,ports[i
-mp->config->nicStart]);
                nic = new NetInterfaceMsg(i,counter,ports[i-mp->config->nicStart]);
                counter += (ports[i-mp->config->nicStart]+1);
                nic->Timestamp(0);
                mapPE = mp->procs*(i-mp->config->nicStart)/mp->BGnodes;

                (*(CProxy_NetInterface *) &POSE_Objects)[i].insert(nic,mapPE);
        }

        counter = 0; nodeid = -1;
        int counter2 = mp->config->ChannelStart;

        for(i=mp->config->InputBufferStart;i< mp->config->ChannelStart;i++) {
                if(!counter) {
                nodeid ++; counter = ports[nodeid]+1; portid = 0;
                if(nodeid)
                counter2 += (ports[nodeid-1]+1);
                }  // Adding 1 to include injection port

                inpbuf = new InputBufferMsg(i,portid,nodeid,ports[nodeid],counter2);
                inpbuf->Timestamp(0);
                mapPE = mp->procs*(nodeid-1)/mp->BGnodes;
                (*(CProxy_InputBuffer *) &POSE_Objects)[i].insert(inpbuf,mapPE);
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

void getMyTopology(Topology **topology) 
{
   *topology = new HyperCube;
}
