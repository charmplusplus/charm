#include "BgSim_sim.h"
#include "TCsim_sim.h"

#include<math.h>
/* put elapse in restart and put elapse in tcsim */

extern void getMyTopology(Topology **);

NetInterface::NetInterface(NetInterfaceMsg *niMsg) {
   nicConsts = new NicConsts;
   nicConsts->id = niMsg->id;  nicConsts->numP = niMsg->numP;
   nicConsts->startId = niMsg->startId;

   numRecvd = 0; prevIntervalStart = 0; counter = 0; roundRobin = 0;
   getMyTopology(&topology);
}

Channel::Channel(ChannelMsg *m) {
   id = m->id;  portid = m->portid;  nodeid = m->nodeid; numP = m->numP;
   prevIntervalStart = 0; counter = 0;
}

Switch::Switch(SwitchMsg *s) {
	id = s->id; numP = s->numP;  
	for(int i=0;i<numP*config.switchVc;i++) {
		mapVc[i]=IDLE;
		requested[i] = 0;
		Bufsize[i] = config.switchBufsize;
	}

        getMyTopology(&topology);
        topology->getNeighbours(id-config.switchStart,numP);
}

void NetInterface::recvMsg(NicMsg *nic) {	
	int delay=0,inputPort,msgLenRest = nic->totalLen,packetnum  = 0,curlen,initPort,initVc;
	Packet *p; 
	NicMsg *newNic = new NicMsg; *newNic = *nic;	
	POSE_invoke(storeMsgInAdvance(newNic),NetInterface,nic->routeInfo.dst+config.nicStart,0);
	while(msgLenRest > 0) { 
		p = new Packet; 
		p->hdr = *nic;
		curlen  = minP(config.maxpacksize,msgLenRest);  
		p->hdr.routeInfo.datalen = curlen;
		CkAssert(delay >= 0);  
	
		p->hdr.portId = topology->getStartPort(nicConsts->id-config.nicStart);
		p->hdr.vcid = topology->getStartVc(nicConsts->id-config.nicStart);	
		p->hdr.prevId = nicConsts->id; 
		p->hdr.prev_vcid = -1;
		p->hdr.nextId = topology->getStartSwitch(nicConsts->id-config.nicStart);
		POSE_invoke(recvPacket(p),Switch,nicConsts->startId,delay);

		delay += (curlen/config.switchC_BW);  msgLenRest -= curlen; packetnum ++;
	}
}

void NetInterface::storeMsgInAdvance(NicMsg *m) {
	MsgStore ms; ms = *m; 
	remoteMsgId rmid(m->msgId,m->src);
	storeBuf[rmid] = ms;
}

void NetInterface::recvPacket(Packet *p) {
	int tmp,expected,extra,hops,remlen; TaskMsg *tm; Position src; MsgStore ms;
	remoteMsgId rmid(p->hdr.msgId,p->hdr.src); 
	map<remoteMsgId,int>::iterator it2 = pktMap.find(rmid);

	CkAssert(p->hdr.routeInfo.dst == (nicConsts->id-config.nicStart));
	if(it2 == pktMap.end())  { 
		remlen = p->hdr.totalLen - (p->hdr.routeInfo.datalen); pktMap[rmid] = remlen;
	} else 
		pktMap[rmid] -= p->hdr.routeInfo.datalen;

//	CkPrintf("Time %d Dumping h in dst %d... ",ovt,nicConsts->id-config.nicStart); h->dump();
	if(!pktMap[rmid])  {
	map<remoteMsgId,MsgStore>::iterator it1 = storeBuf.find(rmid);
	if(it1 == storeBuf.end()) {CkPrintf("Something wrong src %d dst %d msgid %d \n",
	p->hdr.src,p->hdr.routeInfo.dst,p->hdr.msgId);parent->CommitError("message was not stored in advance");return;}

	ms = storeBuf[rmid];
	CkPrintf("id %d size %d time %d src %d msgid %d index %d recvTime %d totalLen %d 
	destNodecode %d destTID %d\n", nicConsts->id-config.nicStart,storeBuf.size(),ovt,
	ms.src,ms.msgId,ms.index,ms.recvTime,ms.totalLen,ms.destNodeCode,ms.destTID);	

	if(config.msgstats_on) {
	extra = 
	topology->routingAlgorithm->expectedTime(p->hdr.src,nicConsts->id-config.nicStart,ovt,ms.origovt,p->hdr.totalLen,&hops);
        int curInterval,tmp=prevIntervalStart;
	if(config.collection_interval != 0) {
        curInterval = ovt/config.collection_interval;
        if((curInterval > prevIntervalStart) && (numRecvd)) {
        prevIntervalStart = curInterval;
        parent->CommitPrintf("%d*%d %d %d %.2f\n",nicConsts->id-config.nicStart,numRecvd,
	hops,prevIntervalStart,(float)counter/numRecvd);counter=0; numRecvd = 0;
        }
	}
	counter += (int)(100.0 * extra/(ovt-ms.origovt)); numRecvd++;
	}
	pktMap.erase(rmid);
	storeBuf.erase(rmid);
	

	tm = new TaskMsg(ms.src,ms.msgId,ms.index,ms.recvTime,ms.totalLen,
	(nicConsts->id-config.nicStart),ms.destNodeCode,ms.destTID);
        POSE_invoke(recvIncomingMsg(tm), BGnode, config.numNodes + p->hdr.routeInfo.dst, 0);
	}
}


void Switch::recvPacket(Packet *copyP) {
	Packet *p; p = new Packet; *p = *copyP;
	int outPort,outVc,inPort,inVc,outVcId,nextChannel,inVcId;

	inPort = topology->routingAlgorithm->convertOutputToInputPort(p->hdr.portId); inVc = p->hdr.vcid;
	outPort = topology->routingAlgorithm->selectRoute(id-config.switchStart,p->hdr.routeInfo.dst,numP,topology->next);

	p->hdr.portId = outPort; 
	outVc = topology->outputVcSelect->selectOutputVc(Bufsize,p);	
	outVcId = outPort*config.switchVc+outVc;
	inVcId = inPort*config.switchVc+inVc;

	if((outVc != NO_VC_AVAILABLE) && !requested[inVcId]) { sendPacket(p,outVcId,outPort,inVcId); } 
	else { inBuffer[inVcId].push_back(p->hdr); }
}

void Switch::sendPacket(Packet *p,const int & outVcId,const int & outPort,const int & inVcId) {
	int goingToNic=0,fromNic=0;
	int nextChannel;
	mapVc[outVcId] =  inVcId;
	requested[inVcId] = 1;

	p->hdr.nextId = topology->getNext(outPort,id,numP) ;  // Use this in channel

	if((p->hdr.nextId >= config.nicStart) && (p->hdr.nextId < (config.nicStart+config.numNodes))) goingToNic = 1;
	if((p->hdr.prevId >= config.nicStart) && (p->hdr.prevId < (config.nicStart+config.numNodes))) fromNic = 1;

	if(!goingToNic) Bufsize[outVcId] -= p->hdr.routeInfo.datalen;

	nextChannel = topology->getNextChannel(outPort,id);

	p->hdr.vcid = (outVcId%numP);
	flowStart *f,*f2; f= new flowStart; f->vcid = p->hdr.prev_vcid; f->datalen = p->hdr.routeInfo.datalen; 
	f2 = new flowStart; *f2 = *f; f2->vcid = inVcId;

	POSE_local_invoke(checkNextPacketInVc(f2),f->datalen/config.switchC_BW);
	if(!fromNic) POSE_invoke(updateCredits(f),Switch,p->hdr.prevId,f->datalen/config.switchC_BW); 
	else delete f;

	p->hdr.prev_vcid = outVcId;
	p->hdr.prevId = id;
	p->hdr.dump();
	POSE_invoke(recvPacket(p),Channel,nextChannel,0);
}

void Switch::checkNextPacketInVc(flowStart *f) {
	int outVc; Packet p,*p2; 
	vector<Header>::iterator headOfBuf;
	p.hdr.routeInfo.datalen = f->datalen; p.hdr.portId = f->vcid/numP;

	requested[f->vcid] = 0;

	if(inBuffer[f->vcid].size()) {
	headOfBuf = inBuffer[f->vcid].begin();
		// Be careful so that neccessary data in packet "p" is populated
		outVc = topology->outputVcSelect->selectOutputVc(Bufsize,&p);	
		if((outVc != NO_VC_AVAILABLE) && !requested[outVc+config.switchVc*(headOfBuf->portId)]) {
			p2 = new Packet; p2->hdr = *headOfBuf;
			inBuffer[f->vcid].erase(headOfBuf);
			sendPacket(p2,outVc+p2->hdr.portId*numP,p2->hdr.portId,f->vcid);
		}
	}
}
	
void Switch::updateCredits(flowStart *f) {
	int outPort,outVc,inPort,inVc,nextChannel,vc;
	Packet *p; vector<Header>::iterator it;
	Bufsize[f->vcid] += f->datalen;
	requested[mapVc[f->vcid]] = 0;	

	vc = topology->inputVcSelect->selectInputVc(Bufsize,requested,inBuffer,mapVc[f->vcid]);  // Make sure vc is port*numVc+myvc
	mapVc[f->vcid] = IDLE;	
	if(vc != NO_VC_AVAILABLE) {
	outPort = f->vcid/config.switchVc; outVc = f->vcid % config.switchVc;
	inPort = vc/config.switchVc; inVc = vc%config.switchVc;

	p = new Packet; it = inBuffer[vc].begin(); p->hdr = *it;
	sendPacket(p,f->vcid,outPort,vc);
	}
}

void Channel::recvPacket(Packet *copyP) {
	Packet *p; p = new Packet; *p = *copyP; 

	if(portid == numP) { // For direct networks only
	POSE_invoke(recvPacket(p),NetInterface,config.nicStart+nodeid,p->hdr.routeInfo.datalen/config.switchC_BW);  
	elapse(p->hdr.routeInfo.datalen/config.switchC_BW);
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

	if((p->hdr.nextId > config.nicStart)  && (p->hdr.nextId < (config.nicStart+config.numNodes))) 
		POSE_invoke(recvPacket(p),NetInterface,p->hdr.nextId,p->hdr.routeInfo.datalen/config.switchC_BW);
	else
		POSE_invoke(recvPacket(p),Switch,p->hdr.nextId,config.switchC_Delay);

        counter+= (config.switchC_Delay+copyP->hdr.routeInfo.datalen/config.switchC_BW); 
	elapse(copyP->hdr.routeInfo.datalen/config.switchC_BW);  // Same buffer , we have already elapsed, so no need.
}
