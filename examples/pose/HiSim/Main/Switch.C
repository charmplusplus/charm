#include "BgSim_sim.h"
#include "TCsim_sim.h"
#include "math.h"

extern void initializeNetwork(Topology **,RoutingAlgorithm **,InputVcSelection **,OutputVcSelection **);
extern void initializeNetwork(Topology **,RoutingAlgorithm **);

/***************************************************************
 * Switch constructor. Initialize data structures and topology *
 ***************************************************************/

Switch::Switch(SwitchMsg *s) {
        id = s->id; numP = s->numP;
	
	if(config.inputBuffering)
        for(int i=0;i<numP*config.switchVc;i++) {
                mapVc[i]=IDLE;
                requested[i] = 0;
                availBufsize[i] = config.switchBufsize;
        }
	else
	for(int i=0;i<=numP;i++) {
		requested[i] = 0;
		availBufsize[i] = config.switchBufsize; 
	}

        initializeNetwork(&topology,&routingAlgorithm,&inputVcSelect,&outputVcSelect);
        topology->getNeighbours(id-config.switchStart,numP);
//      CkPrintf("Node id %d  ",id-config.switchStart);
//      for(int i =0;i<6;i++) CkPrintf(" %d:%d  ",i,topology->next[i]);
}

/******************************************************************
 * Select the output port based on routing algorithm. Then select *
 * an available vc at the port. If no vc available buffer it.     *
 * Otherwise call sendPacket to send the packet			  *
 ******************************************************************/

// Receive packet and route it or buffer depending on current state

void Switch::recvPacket(Packet *copyP) {
        Packet *p; p = new Packet; *p = *copyP;
        int outPort,outVc,inPort,inVc,outVcId,nextChannel,inVcId,bufferid,setRequest;

	if((p->hdr.controlInfo == CONTROL_PACKET) && (p->hdr.routeInfo.dst == (id-config.switchStart))) {
		memcpy(routeTable,p->hdr.nextPort,MAX_ROUTE_HEADER * sizeof(char));
		return;
	}

        inPort = routingAlgorithm->convertOutputToInputPort(id-config.switchStart,p,numP); inVc = p->hdr.vcid;
        p->hdr.portId = inPort;

	if(config.sourceRouting) {
		outPort = p->hdr.nextPort[p->hdr.hop++];
	} else
	if(config.loadRoutingTable) {
		outPort = routeTable[p->hdr.routeInfo.dst];
	} else
        	outPort = routingAlgorithm->selectRoute(id-config.switchStart,p->hdr.routeInfo.dst,numP,topology,p,availBufsize);
			
	
        CkAssert(inPort <= numP);
        p->hdr.portId = outPort;
	if((outPort == numP) && (!config.inputBuffering)) {
		// Direct ticket to node w/o buffering in case of output buffering in direct nets
		sendPacket(p,outPort*config.switchVc,outPort,outPort*config.switchVc);
		return;
	}

	if(config.inputBuffering) 
        outVc = outputVcSelect->selectOutputVc(availBufsize,p,inVc);
	else 
	outVc = outputVcSelect->selectOutputVc(availBufsize,p,inPort); // Never use availBufsize for decision making.

        outVcId = outPort*config.switchVc+outVc;
	if(config.inputBuffering)
	inVcId = inPort*config.switchVc+inVc;
	else {
	inVcId = outVcId;
	// outputBuffering has a special problem in that we already have the output
	// vc and so buffering is to be done only in case not enough downstream tokens
	// if selectOutputVc returned NO_VC_AVAILABLE then outVcId is set to invalid.
	// To prevent that, outVc is valid till outVcId is computed
	if(availBufsize[outPort] < p->hdr.routeInfo.datalen) //uncomment
        	outVc = NO_VC_AVAILABLE; 
	}

	// outputBuffering, don't sent packet if port is sending another packet
	// Basically this is a replication of channel functionality. The only 
	// advantage is we can do QOS in selectInputVc by this. Note that 
	// simulation timing is not affected
	setRequest = (config.inputBuffering)?(requested[inVcId]):requested[outPort];
		
        if((outVc != NO_VC_AVAILABLE) && !setRequest) { 
		sendPacket(p,outVcId,outPort,inVcId); 
	}
        else { 
		Buffer[inVcId].push_back(p->hdr); 
		delete p;
	}
}

/******************************************************************************
 *Send packet to next switch and update credits in previous switch and finally* 
 *invoke a procedure to check next packet to send in the input 		      *
 ******************************************************************************/

void Switch::sendPacket(Packet *p,const int & outVcId,const int & outPort,const int & inVcId) {
        int goingToNic=0,fromNic=0;
        int nextChannel;
        mapVc[outVcId] =  inVcId;
	int setRequest = (config.inputBuffering)?(inVcId):(outPort);


        CkAssert(outPort == p->hdr.portId); 

        p->hdr.nextId = topology->getNext(outPort,id,numP) ;  // Use this in channel

        if((p->hdr.nextId >= config.nicStart) && (p->hdr.nextId < (config.nicStart+config.numNodes))) goingToNic = 1;
        if((p->hdr.prevId >= config.nicStart) && (p->hdr.prevId < (config.nicStart+config.numNodes))) fromNic = 1;

	if(config.inputBuffering || !goingToNic)
        requested[setRequest] = 1; // uncomment

	// availbufsize for output buffering is a very inefficient scheme which 
	// leads to lot of fragmentation. There is a variable for each port indicating
	// the avail buffer space ( actually the worst case avail bufspace)
	// better scheme will take next hop routing into account and have k variables
	// for each port

        if(!goingToNic)  {
		if(config.inputBuffering) 
			availBufsize[outVcId] -= p->hdr.routeInfo.datalen;
		else	  
			availBufsize[outPort] -= p->hdr.routeInfo.datalen;
	}

        nextChannel = topology->getNextChannel(outPort,id,numP);

        p->hdr.vcid = (outVcId%config.switchVc);
        flowStart *f,*f2; f= new flowStart; f->vcid = p->hdr.prev_vcid; f->datalen = p->hdr.routeInfo.datalen;
        f2 = new flowStart; *f2 = *f; f2->vcid = inVcId;

        POSE_local_invoke(checkNextPacketInVc(f2),(POSE_TimeType)(f->datalen/config.switchC_BW));
        if(!fromNic) {
                POSE_invoke(updateCredits(f),Switch,p->hdr.prevId,(POSE_TimeType)(f->datalen/config.switchC_BW));
        }
        else delete f;

        p->hdr.prev_vcid = outVcId;
        p->hdr.prevId = id; 
	p->hdr.prev_src = id-config.switchStart;  // For direct networks

        POSE_invoke(recvPacket(p),Channel,nextChannel,0);
}
// Select a eligible packet at the head of buffer

void Switch::checkNextPacketInVc(flowStart *f) {
        int outVc; Packet p,*p2;
        vector<Header>::iterator headOfBuf;
	if(config.inputBuffering)
        	requested[f->vcid] = 0;
	else
		requested[f->vcid/config.switchVc] = 0;

        if(Buffer[f->vcid].size()) {

        headOfBuf = Buffer[f->vcid].begin(); 
		p.hdr = *headOfBuf;	
                outVc = outputVcSelect->selectOutputVc(availBufsize,&p,f->vcid%config.switchVc);
                if(outVc != NO_VC_AVAILABLE) {
                        p2 = new Packet; p2->hdr = *headOfBuf;
                        Buffer[f->vcid].erase(headOfBuf);
                        sendPacket(p2,outVc+p2->hdr.portId*config.switchVc,p2->hdr.portId,f->vcid);
                }
        }
}

// Update byte credits in the current switch after ack is received 
// Then select a vc which gets to send a packet
void Switch::updateCredits(flowStart *f) {
        int outPort,outVc,inPort,inVc,nextChannel,vc;
        Packet *p; vector<Header>::iterator it;
	if(config.inputBuffering) {
        	requested[mapVc[f->vcid]/config.switchVc] = 0;
        	availBufsize[f->vcid] += f->datalen;
	}
	else {
		requested[f->vcid/config.switchVc] = 0;
        	availBufsize[f->vcid/config.switchVc] += f->datalen;
	}

        vc = inputVcSelect->selectInputVc(availBufsize,requested,Buffer,f->vcid);  // Make sure vc is port*numVc+myvc
        mapVc[f->vcid] = IDLE;
        if(vc != NO_VC_AVAILABLE) {
        outPort = f->vcid/config.switchVc; outVc = f->vcid % config.switchVc;
        inPort = vc/config.switchVc; inVc = vc%config.switchVc;

        p = new Packet; it = Buffer[vc].begin(); p->hdr = *it;
        Buffer[vc].erase(it);
	if(config.inputBuffering)
        sendPacket(p,f->vcid,outPort,vc);
	else
	sendPacket(p,vc,outPort,vc);
        }
}
