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

InputBuffer::InputBuffer(InputBufferMsg *m){
   ibuf = new InputBufferConsts;
   ibuf->id = m->id; ibuf->numP = m->numP; ibuf->nodeid = m->nodeid;
   ibuf->portid = m->portid;ibuf->startChannelId = m->startChannelId;

   for(int i=0;i<config.switchVc;i++)  { Bufsize[i] = config.switchBufsize;  requested[i] = 0; }
   InputRoundRobin = RequestRoundRobin = AssignVCRoundRobin = 0;

   getMyTopology(&topology);
   topology->getNeighbours(ibuf->nodeid,ibuf->numP);
}

Channel::Channel(ChannelMsg *m) {
   id = m->id;  portid = m->portid;  nodeid = m->nodeid; numP = m->numP;
   prevIntervalStart = 0; counter = 0;
}

Switch::Switch(SwitchMsg *s) {
	id = s->id;
	for(int i=0;i<config.fanout*config.switchVc;i++) {
		mapVc[i]=IDLE;
		requested[i] = 0;
	}
   getMyTopology(&topology);
   topology->getNeighbours(id-config.switchStart,2*config.fanout);
}

void NetInterface::recvMsg(NicMsg *nic) {	
	int delay=0,inputPort,msgLenRest = nic->totalLen,packetnum  = 0,curlen;
	Packet *p; 
	NicMsg *newNic = new NicMsg; *newNic = *nic;	
	POSE_invoke(storeMsgInAdvance(newNic),NetInterface,nic->routeInfo.dst+config.nicStart,0);
	while(msgLenRest > 0) { 
		p = new Packet; 
		p->hdr = *nic;
		curlen  = minP(config.maxpacksize,msgLenRest);  
		p->hdr.routeInfo.datalen = curlen;
		CkAssert(delay >= 0);  
		
		#ifndef INDIRECT_NETWORK
		POSE_invoke(recvPacket(p),InputBuffer,nicConsts->startId+nicConsts->numP,delay);
		#else
		// to keep translator happy
		p->hdr.portId = (nicConsts->id-config.nicStart)%config.fanout;
		p->hdr.prevId = nicConsts->id; p->hdr.vcid = roundRobin; roundRobin = (roundRobin+1)%config.switchVc;
		p->hdr.nextId = config.switchStart + (nicConsts->id-config.nicStart)/config.fanout;
		CkPrintf("NIC: Next ChannelId is %d \n",nicConsts->startId);
		POSE_invoke(recvPacket(p),Channel,nicConsts->startId,delay);
		#endif
		// to keep translator happy
		delay += (curlen/config.switchC_BW);  msgLenRest -= curlen; packetnum ++;
	}
}

void NetInterface::storeMsgInAdvance(NicMsg *m) {
	MsgStore ms;
	ms = *m; 
	remoteMsgId rmid(m->msgId,m->src);
	storeBuf[rmid] = ms;
//	CkPrintf("\nid %d time %d stored src %d msgid %d index %d recvTime %d totalLen %d destNodecode %d destTID %d\n",
//				nicConsts->id-config.nicStart,ovt,ms.src,ms.msgId,ms.index,ms.recvTime,ms.totalLen,ms.destNodeCode,ms.destTID);	
	
}

void NetInterface::recvPacket(Packet *p) {
	int tmp,expected,extra,hops,remlen; TaskMsg *tm; Position src; MsgStore ms;
	remoteMsgId rmid(p->hdr.msgId,p->hdr.src); 
	map<remoteMsgId,int>::iterator it2 = pktMap.find(rmid);

	CkAssert(p->hdr.routeInfo.dst == (nicConsts->id-config.nicStart));
	if(it2 == pktMap.end())  {
		remlen = p->hdr.totalLen - (p->hdr.routeInfo.datalen);
		CkPrintf("Time %d dst %d I am inserting into msgid %d src %d \n",
			ovt,p->hdr.routeInfo.dst,p->hdr.msgId,p->hdr.src);
		pktMap[rmid] = remlen;
	}
	else
		pktMap[rmid] -= p->hdr.routeInfo.datalen;

//	CkPrintf("Time %d Dumping h in dst %d... ",ovt,nicConsts->id-config.nicStart); h->dump();
	if(!pktMap[rmid])  {
	map<remoteMsgId,MsgStore>::iterator it1 = storeBuf.find(rmid);
	if(it1 == storeBuf.end()) {CkPrintf("Something wrong src %d dst %d msgid %d \n",p->hdr.src,p->hdr.routeInfo.dst,p->hdr.msgId);parent->CommitError("message was not stored in advance");return;}

	ms = storeBuf[rmid];
//	CkPrintf("\nid %d size %d time %d src %d msgid %d index %d recvTime %d totalLen %d destNodecode %d destTID %d\n", nicConsts->id-config.nicStart,storeBuf.size(),ovt,
//				ms.src,ms.msgId,ms.index,ms.recvTime,ms.totalLen,ms.destNodeCode,ms.destTID);	
	if(config.msgstats_on) {
	extra = topology->routingAlgorithm->expectedTime(p->hdr.src,nicConsts->id-config.nicStart,ovt,ms.origovt,p->hdr.totalLen,&hops);
       int curInterval,tmp=prevIntervalStart;
	if(config.collection_interval != 0) {
        curInterval = ovt/config.collection_interval;
        if((curInterval > prevIntervalStart) && (numRecvd)) {
        prevIntervalStart = curInterval;
        parent->CommitPrintf("%d*%d %d %d %.2f\n",nicConsts->id-config.nicStart,numRecvd,hops,prevIntervalStart,(float)counter/numRecvd);counter=0; numRecvd = 0;
        }
	}
	counter += (int)(100.0 * extra/(ovt-ms.origovt)); numRecvd++;
	}
	pktMap.erase(it2);
	storeBuf.erase(it1);
	

	tm = new TaskMsg(ms.src,ms.msgId,ms.index,ms.recvTime,ms.totalLen,(nicConsts->id-config.nicStart),ms.destNodeCode,ms.destTID);
        POSE_invoke(recvIncomingMsg(tm), BGnode, config.numNodes + p->hdr.routeInfo.dst, 0);
	}
}


void Switch::recvPacket(Packet *copyP) {
	Packet *p; p = new Packet; *p = *copyP;
	int outPort,outVc,inPort,inVc,outVcId,nextChannel,inVcId;
	inPort = p->hdr.portId; inVc = p->hdr.vcid;
	
	outPort = topology->routingAlgorithm->selectRoute(topology->getStartNode(),topology->getEndNode(),p);
	p->hdr.portId = outPort;
	outVc = topology->outputVcSelect->selectOutputVc(Bufsize,mapVc,p);	
	outVcId = outPort*config.switchVc+outVc;
	inVcId = inPort*config.switchVc+inVc;
	p->hdr.prevId = id;

	if((outVc != NO_VC_AVAILABLE) && !requested[outVcId]) {
		sendPacket(p,outVcId,outPort,inVcId);
	} else {
	p->hdr.portId = (outPort+config.fanout)%config.fanout;
	inBuffer[inPort*config.switchVc+inVc].push_back(p->hdr);
	}
}

// Remove Packet memory if not used above ..

void Switch::sendPacket(Packet *p,const int & outVcId,const int & outPort,const int & inVcId) {
	int numP = 2*config.fanout,goingToNic=0,fromNic=0;
	int nextChannel,fanout=config.fanout;
	mapVc[outVcId] =  inVcId;
	requested[inVcId] = 1;

	// No flow control for the network interface. Go flood them !
	p->hdr.nextId = topology->getNext(outPort,-1,-1) ;  // Use this in channel
	if((p->hdr.nextId > config.nicStart) && (p->hdr.nextId < config.nicStart+config.numNodes)) 
		goingToNic = 1;
	if((p->hdr.prevId > config.nicStart) && (p->hdr.prevId < config.nicStart+config.numNodes)) 
		fromNic = 1;

	if(!goingToNic)
	Bufsize[outVcId] -= p->hdr.routeInfo.datalen;

	nextChannel = config.ChannelStart + (id-config.switchStart)*2*numP + outPort*2;

	p->hdr.portId = (outPort+fanout)%fanout;
	p->hdr.vcid = (outVcId%numP);
	flowStart *f,*f2; f= new flowStart; f->vcid = inVcId; f->datalen = p->hdr.routeInfo.datalen;
	f2 = new flowStart; *f2 = *f; 

	// The next packet has to wait till the current packet is fully out. As we know, that depends on bandwidth
	POSE_local_invoke(checkNextPacketInVc(f2),f->datalen/config.switchC_BW);

	// One of the packets in previous switch, which will occupy the channel occupied earlier by current packet
	// has to wait for a time dependent on bandwidth
	if(!fromNic)
	POSE_invoke(updateCredits(f),Switch,p->hdr.prevId,f->datalen/config.switchC_BW);

	POSE_invoke(recvPacket(p),Channel,nextChannel,0);
}

// Similar to recvPacket, except that routing is not performed. But routing delay is included in the
// latency invoked in the channel.

void Switch::checkNextPacketInVc(flowStart *f) {
	int outVc,numP = 2*config.fanout; Packet p,*p2; 
	vector<Header>::iterator headOfBuf;
	p.hdr.routeInfo.datalen = f->datalen; p.hdr.portId = f->vcid/numP;

	requested[f->vcid] = 0;

	if(!inBuffer[f->vcid].size()) {
	headOfBuf = inBuffer[f->vcid].begin();
		// Be careful so that neccessary data in packet "p" is populated
		outVc = topology->outputVcSelect->selectOutputVc(Bufsize,mapVc,&p);	
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
	mapVc[f->vcid] = IDLE;	
	// Check for the input vc unblocked by the current packet
	// Check for the guys waiting for the current output

	vc = topology->inputVcSelect->selectInputVc(Bufsize,requested,inBuffer,f->vcid);  // Make sure vc is port*numVc+myvc
	if(vc != NO_VC_AVAILABLE) {
	outPort = f->vcid/config.switchVc;
	outVc = f->vcid % config.switchVc;
	inPort = vc/config.switchVc;
	inVc = vc%config.switchVc;

	p = new Packet;
	it = inBuffer[vc].begin();
	p->hdr = *it;
	sendPacket(p,f->vcid,outPort,vc);
	}

	// Search through inBuffer to see if anyone is waiting for this output port ..
}



/*******************************************************************************************
 *  Check if vc available according to given constraints. If available send message to the *
 *  previous node's vc . The message is delayed by a time equal to msglen/bandwidth. This  *
 *  amounts to elapse in that input buffer... Be careful not to elapse this time in channel*
 *  If no eligible vc, then just queue the request. 					   *
 *******************************************************************************************/

// Don't do a silly thing like restarting after len/bw . There should be two messages. 
// One with Zero delay and another with len/bw delay. The first one for the requesting packet
// The second one should be for restarting flow in the same buffer. Another way would be zero delay
// here. Then in restartBufferFlow, checkVC can be called after len/Bw !

/**************************************************************************************************
 * Channel has two meanings ...									  *
 * 1) It is the reception FIFO  								  *
 * 2) It is the channel connection two nodes							  *
 * For case 1) send to the network interface and do some social service afterwards.		  *
 * For case 2) send to the next inputbuffer. Note that 2) lacks len/bw (recvRequest took care)    *
 **************************************************************************************************/

void Channel::recvPacket(Packet *copyP) {
	Packet *p; p = new Packet; *p = *copyP; 

#ifndef INDIRECT_NETWORK

	if(portid == numP) {
//	CkPrintf("Sending to nodeid %d at time %d \n",nodeid,ovt);h->dump();
	POSE_invoke(recvPacket(p),NetInterface,config.nicStart+nodeid,p->hdr.routeInfo.datalen/config.switchC_BW);  
	// POSE_invoke(checkVC(f),InputBuffer,p->hdr.prevId,0);  Isn't this a redundant line ?
	elapse(p->hdr.routeInfo.datalen/config.switchC_BW);
	return;
	}
#endif

// Nodeid to be taken as switchid for indirect network

       int curInterval,tmp=prevIntervalStart,nextId;
	if(config.linkstats_on) {
        curInterval = ovt/config.collection_interval;
        if(curInterval > prevIntervalStart) {
        prevIntervalStart = curInterval;
        parent->CommitPrintf("%d->%d %d %f\n",nodeid,portid,prevIntervalStart,(float)counter/config.collection_interval);
	counter=0;
        }
	}

#ifndef INDIRECT_NETWORK
	POSE_invoke(recvPacket(p),InputBuffer,p->hdr.nextId,config.switchC_Delay); 
#else
		CkPrintf("Channel: Next Switch/NetInterface Id is %d \n",p->hdr.nextId);
	if((p->hdr.nextId > config.nicStart)  && (p->hdr.nextId < (config.nicStart+config.numNodes))) 
		POSE_invoke(recvPacket(p),NetInterface,p->hdr.nextId,p->hdr.routeInfo.datalen/config.switchC_BW);
	else
		POSE_invoke(recvPacket(p),Switch,p->hdr.nextId,config.switchC_Delay);
#endif

	// The below figure should include both the internal and external channel delay . If just external delay 
	// is needed, then config.switchC_Delay should be split up.
        counter+= (config.switchC_Delay+copyP->hdr.routeInfo.datalen/config.switchC_BW); 
	elapse(copyP->hdr.routeInfo.datalen/config.switchC_BW);  // Same buffer , we have already elapsed, so no need.
}


void InputBuffer::recvRequest(Packet *p) {
        Request req(p->hdr.prevId,p->hdr.routeInfo.datalen);flowStart *reqnew; 
	int vc;
       	vc = topology->outputVcSelect->selectOutputVc(Bufsize,p);
 
        if(vc != NO_VC_AVAILABLE) { 
	reqnew = new flowStart;  reqnew->vcid = vc; reqnew->prev_vcid = p->hdr.vcid;
        Bufsize[vc] -= p->hdr.routeInfo.datalen;      
	CkAssert(p->hdr.prevId > 0);
	reqnew->datalen = p->hdr.routeInfo.datalen;
	POSE_invoke(restartBufferFlow(reqnew),InputBuffer,p->hdr.prevId,0);
        } else {
	req.vcid = p->hdr.vcid;
	requestQ[AssignVCRoundRobin].push_back(req); 
	AssignVCRoundRobin = (AssignVCRoundRobin+1)%config.switchVc;
	}
}

void InputBuffer::recvPacket(Packet *copyP) {
	int nextId;
	flowStart *f;
	Packet *p; p = new Packet; *p = *copyP; int len=p->hdr.routeInfo.datalen,nextP,proceed=1,i; 
	Position dst;  dst.init(p->hdr.routeInfo.dst); 

	if((p->hdr.routeInfo.dst == ibuf->nodeid) && !inBuffer[p->hdr.vcid].size()) {
	// This will bypass the input speedup requirement
	// Alternative is to buffer it and let restartBufferFlow take care of later
	requested[p->hdr.vcid] = 1;
	Bufsize[p->hdr.vcid] += p->hdr.routeInfo.datalen;
	CkAssert(p->hdr.src < config.numNodes);
	p->hdr.prevId = ibuf->id;
	POSE_invoke(recvPacket(p),Channel,ibuf->startChannelId+ibuf->numP,0);  
	f = new flowStart; f->prev_vcid = p->hdr.vcid; f->datalen = p->hdr.routeInfo.datalen;
	POSE_local_invoke(checkVC(f),f->datalen/config.switchC_BW);
	return;
	}	

	if(p->hdr.routeInfo.dst != ibuf->nodeid)  {
	nextId = topology->routingAlgorithm->selectRoute(ibuf->nodeid,p->hdr.routeInfo.dst,ibuf->numP,topology->next);
	p->hdr.nextId = topology->getNext(nextId,ibuf->nodeid,ibuf->numP) ;  // make this a data member
	//CkPrintf("Time %d Route selected at %d for src %d dst %d was %d \n",ovt,ibuf->nodeid,h->src,h->routeInfo.dst,topology->next[nextId]);
	}

	p->hdr.prevId = ibuf->id;
	CkAssert(p->hdr.prevId > 0);
	p->hdr.portId = nextId;        

	inBuffer[p->hdr.vcid].push_back(p->hdr);

	for(i=0;i<config.switchVc;i++) {
		if(i == p->hdr.vcid) continue;
		if(inBuffer[i].size()) { proceed = 0; break; }
	}

        if((inBuffer[p->hdr.vcid].size() == 1) && proceed && (!requested[p->hdr.vcid])) { 
	requested[p->hdr.vcid] = 1; 
	POSE_invoke(recvRequest(p),InputBuffer,p->hdr.nextId,0);  
	}
        else { 
	delete p;  
	}
}

/******************************************************************************
 * Restart flow from an stalled vc. After restarting call checkVc to check on *
 * others ( Do some social service :) ) Also be sure to call checkVc as that  *
 * is where requested[vc] is set to 0. Am not doing it here since I don't want*
 * packets that arrive between now and now+len/bandwidth to pass through      *
 ******************************************************************************/

void InputBuffer::restartBufferFlow(flowStart *h) {
        int curlen;
        Packet *newp;Header cp; 
        vector<Header>::iterator headOfBuf;
        Position dst; int vcid=h->prev_vcid,i;
                                                                                                                  
        if(inBuffer[vcid].size() == 0) 
	{ 
       	    parent->CommitError("How ??\n"); 
	    return;
	} 
	else headOfBuf = inBuffer[vcid].begin();

	cp = *headOfBuf;
        newp = new Packet; newp->hdr = cp;  newp->hdr.vcid = h->vcid;   // This last line is crap for dest packets
	newp->hdr.prev_vcid = vcid;
	Bufsize[vcid] += newp->hdr.routeInfo.datalen; 
	inBuffer[vcid].erase(headOfBuf); 
//	parent->CommitPrintf("%d In restartBufferFlow %d  vcid %d erased size now is %d \n",ovt,ibuf->id,vcid,inBuffer[vcid].size());
	curlen = newp->hdr.routeInfo.datalen;

	if(newp->hdr.routeInfo.dst != ibuf->nodeid) 
        POSE_invoke(recvPacket(newp),Channel,ibuf->startChannelId+newp->hdr.portId,0);
	else
        POSE_invoke(recvPacket(newp),Channel,ibuf->startChannelId+ibuf->numP,0);

	flowStart *f; f = new flowStart; *f = *h;
	POSE_local_invoke(checkVC(f),f->datalen/config.switchC_BW);
}

/**********************************************************************
 *  Select the next vc within this port to have the right to request. *
 *  Then check the pending requests from previous nodes ..            *
 **********************************************************************/

void InputBuffer::checkVC(flowStart *h) {
        Packet *newp; Header cp; flowStart *signalTransferStart; 
        vector<Header>::iterator headOfBuf; vector<Request>::iterator req;
        Position dst; int vcid=h->prev_vcid,i=0;
	int vc;
	requested[vcid] = 0;

       	vc = topology->inputVcSelect->selectInputVc(Bufsize,requested,inBuffer,0);
//	parent->CommitPrintf("%d In checkVc %d vcid %d size %d\n",ovt,ibuf->id,vc,inBuffer[vc].size());
 
        if(vc != NO_VC_AVAILABLE ) { 
	headOfBuf = inBuffer[vc].begin(); newp = new Packet; cp = *headOfBuf; newp->hdr = cp; 
	if(newp->hdr.routeInfo.dst != ibuf->nodeid) {
        requested[vc] = 1;
        POSE_invoke(recvRequest(newp),InputBuffer,newp->hdr.nextId,0);
	} else {
	parent->CommitPrintf("This code is not executed ?\n");
        requested[vc] = 1;
	signalTransferStart = new flowStart; 
	signalTransferStart->datalen = newp->hdr.routeInfo.datalen;
	delete newp;
	signalTransferStart->prev_vcid = vc; signalTransferStart->vcid = h->vcid; 
	POSE_local_invoke(restartBufferFlow(signalTransferStart),0);
	}
        } 
           
	int counter = 0;                                                                                                      
        while(counter < (config.switchVc+1)) { 
	RequestRoundRobin =  (RequestRoundRobin+1)%config.switchVc;
        if(requestQ[RequestRoundRobin].size() != 0)  {
                req = requestQ[RequestRoundRobin].begin();
                if(req->datalen <= Bufsize[RequestRoundRobin]) {
                signalTransferStart = new flowStart; signalTransferStart->vcid = RequestRoundRobin;
                signalTransferStart->prev_vcid = req->vcid; Bufsize[RequestRoundRobin] -= req->datalen;
		CkAssert(req->nextId > 0);
		signalTransferStart->datalen = req->datalen;
                POSE_invoke(restartBufferFlow(signalTransferStart),InputBuffer,req->nextId,0);
                requestQ[RequestRoundRobin].erase(req); counter = 0;
                } 
        } 
	counter++;
	}
}
