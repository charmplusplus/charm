#include "BgSim_sim.h"
#include "TCsim_sim.h"
#include "math.h"

extern void initializeNetwork(Topology **,RoutingAlgorithm **,InputVcSelection **,OutputVcSelection **);
extern void initializeNetwork(Topology **,RoutingAlgorithm **);

NetInterface::NetInterface(NetInterfaceMsg *niMsg) {
   nicConsts = new NicConsts;
   nicConsts->id = niMsg->id;  nicConsts->numP = niMsg->numP;
   nicConsts->startId = niMsg->startId;

   numRecvd = 0; prevIntervalStart = 0; counter = 0; roundRobin = 0;
   initializeNetwork(&topology,&routingAlgorithm);
}

// Packetize message and pump it out
void NetInterface::recvMsg(NicMsg *nic) {
        POSE_TimeType delay=0,inputPort,msgLenRest = nic->totalLen,packetnum  = 0,curlen,initPort,initVc;
        Packet *p;
        NicMsg *newNic = new NicMsg; *newNic = *nic;
        POSE_invoke(storeMsgInAdvance(newNic),NetInterface,nic->routeInfo.dst+config.nicStart,0);

//	parent->CommitPrintf("-%d %d %d %d\n",nic->src,nic->msgId,nic->routeInfo.dst,nic->totalLen);
        while(msgLenRest > 0) {
                p = new Packet;
                p->hdr = *nic;
                curlen  = minP(config.maxpacksize,msgLenRest);
                p->hdr.routeInfo.datalen = curlen;
                CkAssert(delay >= 0);
		p->hdr.pktId = packetnum ++;
                p->hdr.portId = topology->getStartPort(nicConsts->id-config.nicStart,nicConsts->numP);
                p->hdr.vcid = roundRobin; roundRobin = (roundRobin+1)%config.switchVc;
                p->hdr.prevId = nicConsts->id;
                p->hdr.prev_vcid = -1; p->hdr.prev_src = p->hdr.src;
                p->hdr.nextId = topology->getStartSwitch(nicConsts->id-config.nicStart);

                POSE_invoke(recvPacket(p),Switch,nicConsts->startId,delay);
                delay += ((POSE_TimeType)(curlen/config.switchC_BW));  msgLenRest -= curlen; 
        }
elapse((POSE_TimeType)(nic->totalLen/config.switchC_BW));
}


// Store part of message having higher level protocol directly to destination
void NetInterface::storeMsgInAdvance(NicMsg *m) {
        MsgStore ms; ms = *m;
        remoteMsgId rmid(m->msgId,m->src);
        storeBuf[rmid] = ms;
//      CkPrintf("%d Stored src %d msgid %d\n",ovt,m->src,m->msgId);
}

// Receive packet by packet and finally send message to node

void NetInterface::recvPacket(Packet *p) {
        POSE_TimeType tmp,expected,extra,hops,remlen; TaskMsg *tm; TransMsg *tr;Position src; MsgStore ms;
        remoteMsgId rmid(p->hdr.msgId,p->hdr.src);
        map<remoteMsgId,int>::iterator it2 = pktMap.find(rmid);

        if(p->hdr.routeInfo.dst != (nicConsts->id-config.nicStart)) {
//              CkPrintf("Current node %d \n",nicConsts->id-config.nicStart); p->hdr.dump();
        }

//      CkPrintf("NIC id is %d dst is %d \n",nicConsts->id,p->hdr.routeInfo.dst);
        if(p->hdr.routeInfo.dst != (nicConsts->id-config.nicStart)) {
                parent->CommitPrintf("%d actual dst was %d src %d dst %d\n",
                ovt,nicConsts->id-config.nicStart,p->hdr.src,p->hdr.routeInfo.dst);
                parent->CommitError("Packet misrouted to destination\n");
                return;
        }

	numRecvd+= p->hdr.routeInfo.datalen;
        if(it2 == pktMap.end())  {
                remlen = p->hdr.totalLen - (p->hdr.routeInfo.datalen); pktMap[rmid] = remlen;
        } else {
                pktMap[rmid] -= p->hdr.routeInfo.datalen;
	}

        if(!pktMap[rmid])  {
        map<remoteMsgId,MsgStore>::iterator it1 = storeBuf.find(rmid);
        if(it1 == storeBuf.end()) {CkPrintf("%d Something wrong src %d dst %d msgid %d \n",
        ovt,p->hdr.src,p->hdr.routeInfo.dst,p->hdr.msgId);
        parent->CommitError("message was not stored in advance");
        return;}

        ms = storeBuf[rmid];
//      CkPrintf("id %d size %d time %d src %d dst %d msgid %d index %d recvTime %d totalLen %d
//      destNodecode %d destTID %d\n", nicConsts->id-config.nicStart,storeBuf.size(),ovt,
//      ms.src,p->hdr.routeInfo.dst,ms.msgId,ms.index,ms.recvTime,ms.totalLen,ms.destNodeCode,ms.destTID);

        if(config.msgstats_on) {
        extra =
        routingAlgorithm->expectedTime(p->hdr.src,nicConsts->id-config.nicStart,ovt,ms.origovt,p->hdr.totalLen,&hops);
        POSE_TimeType curInterval,tmp=prevIntervalStart;
        if(config.collection_interval != 0) {
        curInterval = ovt/config.collection_interval;
        if((curInterval > prevIntervalStart) && (numRecvd)) {
        prevIntervalStart = curInterval;
        parent->CommitPrintf("%d*%d %d %ld %.2f\n",nicConsts->id-config.nicStart,numRecvd,
        hops,prevIntervalStart,(float)counter/numRecvd);counter=0; numRecvd = 0;
        }
        }
        counter += ((POSE_TimeType)(100.0 * extra/(ovt-ms.origovt))) * (p->hdr.totalLen); 
        }

        pktMap.erase(rmid);
        storeBuf.erase(rmid);

//	parent->CommitPrintf("%d %d %d %d\n",ms.src,ms.msgId,nicConsts->id-config.nicStart,ovt);
	if(config.use_transceiver) {
                tr = new TransMsg(ms.src,ms.msgId,(nicConsts->id-config.nicStart)); 
		// Be careful. Making assumption that nodeStart == 0
                POSE_invoke(recvMessage(tr),Transceiver,nicConsts->id-config.nicStart,0); 
	} else {	
        	tm = new TaskMsg(ms.src,ms.msgId,ms.index,ms.recvTime,ms.totalLen,
 	       (nicConsts->id-config.nicStart),ms.destNodeCode,ms.destTID);
       		POSE_invoke(recvIncomingMsg(tm), BGnode, config.origNodes + p->hdr.routeInfo.dst, 0);
	}
        }
}

