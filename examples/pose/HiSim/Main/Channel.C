#include "BgSim_sim.h"
#include "TCsim_sim.h"

#include<math.h>


Channel::Channel(ChannelMsg *m) {
   id = m->id;  portid = m->portid;  nodeid = m->nodeid; numP = m->numP;
   prevIntervalStart = 0; counter = 0;
}

// Send packet to switch or node. Also doubles as the reception channel in direct network
void Channel::recvPacket(Packet *copyP) {
        Packet *p; p = new Packet; *p = *copyP;

        CkAssert(p->hdr.src >= 0);
        if(portid == numP) { // For direct networks only
        POSE_invoke(recvPacket(p),NetInterface,config.nicStart+nodeid,(POSE_TimeType)(p->hdr.routeInfo.datalen/config.switchC_BW));
	if(!config.receptionSerial)
        elapse((POSE_TimeType)(p->hdr.routeInfo.datalen/config.switchC_BW));
        return;
        }

       POSE_TimeType curInterval,tmp=prevIntervalStart,nextId;
        if(config.linkstats_on) {
        curInterval = ovt/config.collection_interval;
        if(curInterval > prevIntervalStart) {
        prevIntervalStart = curInterval;
        parent->CommitPrintf("%d->%d %ld %f\n",nodeid,portid,prevIntervalStart,(float)counter/config.collection_interval);
        counter=0;
        }
        }


        if((p->hdr.nextId >= config.nicStart)  && (p->hdr.nextId < (config.nicStart+config.numNodes)))
                POSE_invoke(recvPacket(p),NetInterface,p->hdr.nextId,(POSE_TimeType)(p->hdr.routeInfo.datalen/config.switchC_BW));
        else {
		CkAssert(p->hdr.nextId >= config.switchStart);
                POSE_invoke(recvPacket(p),Switch,p->hdr.nextId,config.switchC_Delay);
	}
        counter+= ((POSE_TimeType)(config.switchC_Delay+copyP->hdr.routeInfo.datalen/config.switchC_BW));
        elapse((POSE_TimeType)(copyP->hdr.routeInfo.datalen/config.switchC_BW));  // Same buffer , we have already elapsed, so no need.
}
