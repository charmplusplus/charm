#include "BgSim_sim.h"
#include "TCsim_sim.h"

#include<math.h>


Channel::Channel(ChannelMsg *m) {
   id = m->id;  portid = m->portid;  nodeid = m->nodeid; numP = m->numP;
   prevIntervalStart = 0; counter = 0;
}

void Channel::recvPacket(Packet *copyP) {
        Packet *p; p = new Packet; *p = *copyP;

        CkAssert(p->hdr.src >= 0);
        if(portid == numP) { // For direct networks only
        POSE_invoke(recvPacket(p),NetInterface,config.nicStart+nodeid,(int)(p->hdr.routeInfo.datalen/config.switchC_BW));
	if(!config.receptionSerial)
        elapse((int)(p->hdr.routeInfo.datalen/config.switchC_BW));
        return;
        }

       int curInterval,tmp=prevIntervalStart,nextId;
        if(config.linkstats_on) {
        curInterval = ovt/config.collection_interval;
        if(curInterval > prevIntervalStart) {
        prevIntervalStart = curInterval;
        parent->CommitPrintf("%d->%d %d %f\n",nodeid,portid,prevIntervalStart,(float)counter/config.collection_interval);
        counter=0;
        }
        }


        if((p->hdr.nextId >= config.nicStart)  && (p->hdr.nextId < (config.nicStart+config.numNodes)))
                POSE_invoke(recvPacket(p),NetInterface,p->hdr.nextId,(int)(p->hdr.routeInfo.datalen/config.switchC_BW));
        else
                POSE_invoke(recvPacket(p),Switch,p->hdr.nextId,config.switchC_Delay);

        counter+= ((int)(config.switchC_Delay+copyP->hdr.routeInfo.datalen/config.switchC_BW));
        elapse((int)(copyP->hdr.routeInfo.datalen/config.switchC_BW));  // Same buffer , we have already elapsed, so no need.
}
