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
        for(int i=0;i<numP*config.switchVc;i++) {
                mapVc[i]=IDLE;
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
        int outPort,outVc,inPort,inVc,outVcId,nextChannel,inVcId,bufferid;

//	CkPrintf("At switch %d \n",id-config.switchStart);
        inPort = routingAlgorithm->convertOutputToInputPort(id-config.switchStart,p,numP); inVc = p->hdr.vcid;
        p->hdr.portId = inPort;
        outPort = routingAlgorithm->selectRoute(id-config.switchStart,p->hdr.routeInfo.dst,numP,topology,p,availBufsize);

        CkAssert(inPort <= numP);
        p->hdr.portId = outPort;
        outVc = outputVcSelect->selectOutputVc(availBufsize,p,inVc);
        outVcId = outPort*config.switchVc+outVc;
	if(!config.inputBuffering)  inVcId = outVcId;  else  inVcId = inPort*config.switchVc+inVc;

//      parent->CommitPrintf("recvPacket: ovt %d portid %d supposed portid %d nextid is %d nicEnd is %d src %d dst %d msgid %d\n",
//      ovt,outPort,p->hdr.portId,p->hdr.nextId,config.nicStart+config.numNodes,p->hdr.src,p->hdr.routeInfo.dst,p->hdr.msgId);

	// mapVc and requested do not hold any significance for output buffering. 
        if((outVc != NO_VC_AVAILABLE) && !requested[inVcId]) { sendPacket(p,outVcId,outPort,inVcId); }
        else { Buffer[inVcId].push_back(p->hdr); delete p;}
}

/******************************************************************************
 *Send packet to next switch and update credits in previous switch and finally* 
 *invoke a procedure to check next packet to send in the input 		      *
 ******************************************************************************/

void Switch::sendPacket(Packet *p,const int & outVcId,const int & outPort,const int & inVcId) {
        int goingToNic=0,fromNic=0;
        int nextChannel;
        mapVc[outVcId] =  inVcId;
	if(config.inputBuffering)
        requested[inVcId] = 1;

        CkAssert(outPort == p->hdr.portId);

        p->hdr.nextId = topology->getNext(outPort,id,numP) ;  // Use this in channel

        if((p->hdr.nextId >= config.nicStart) && (p->hdr.nextId < (config.nicStart+config.numNodes))) goingToNic = 1;
        if((p->hdr.prevId >= config.nicStart) && (p->hdr.prevId < (config.nicStart+config.numNodes))) fromNic = 1;

        if(!goingToNic) availBufsize[outVcId] -= p->hdr.routeInfo.datalen;

        nextChannel = topology->getNextChannel(outPort,id,numP);

        p->hdr.vcid = (outVcId%config.switchVc);
        flowStart *f,*f2; f= new flowStart; f->vcid = p->hdr.prev_vcid; f->datalen = p->hdr.routeInfo.datalen;
        f2 = new flowStart; *f2 = *f; f2->vcid = inVcId;

        POSE_local_invoke(checkNextPacketInVc(f2),(int)(f->datalen/config.switchC_BW));
        if(!fromNic) {
                POSE_invoke(updateCredits(f),Switch,p->hdr.prevId,(int)(f->datalen/config.switchC_BW));
        }
        else delete f;

        p->hdr.prev_vcid = outVcId;
        p->hdr.prevId = id; 
	p->hdr.prev_src = id-config.switchStart;  // For direct networks
//      p->hdr.dump();
//      parent->CommitPrintf("sendPacket: ovt %d portid %d supposed portid %d nextid is %d nicEnd is %d src %d dst %d\n",
//                      ovt,outPort,p->hdr.portId,p->hdr.nextId,config.nicStart+config.numNodes,p->hdr.src,p->hdr.routeInfo.dst);

        POSE_invoke(recvPacket(p),Channel,nextChannel,0);
}
// Select a eligible packet at the head of buffer

void Switch::checkNextPacketInVc(flowStart *f) {
        int outVc; Packet p,*p2;
        vector<Header>::iterator headOfBuf;
        requested[f->vcid] = 0;

        if(Buffer[f->vcid].size()) {
        headOfBuf = Buffer[f->vcid].begin();
                // Be careful so that neccessary data in packet "p" is populated
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
void Switch::updateCredits(flowStart *f) {
        int outPort,outVc,inPort,inVc,nextChannel,vc;
        Packet *p; vector<Header>::iterator it;
        availBufsize[f->vcid] += f->datalen;
        requested[mapVc[f->vcid]] = 0;

        // For SLQ, one level is fine, for others two levels of input vc selection should be put in later.
        // First, selection is done on an input port by input port basis for eligible vc. Then on inter-port results are combined
        vc = inputVcSelect->selectInputVc(availBufsize,requested,Buffer,f->vcid);  // Make sure vc is port*numVc+myvc
        mapVc[f->vcid] = IDLE;
        if(vc != NO_VC_AVAILABLE) {
        outPort = f->vcid/config.switchVc; outVc = f->vcid % config.switchVc;
        inPort = vc/config.switchVc; inVc = vc%config.switchVc;

        p = new Packet; it = Buffer[vc].begin(); p->hdr = *it;
        Buffer[vc].erase(it);
        sendPacket(p,f->vcid,outPort,vc);
        }
}
