using namespace std;
#include <stdlib.h>
#include <math.h>
#include <map>
#include <vector>
#include <string>
#include <deque>
#include <algorithm>
#include <functional>
#include <utility>
#include <limits.h>
#include "util.h"
#include "InitNetwork.h"

//#define DEBUG

#ifdef DEBUG
#define print_praveen CkPrintf
#else
#define print_praveen //
#endif

#define CPU_OVERHEAD 400
#define START_LATENCY 200
#define IDLE -1

enum {
	X_POS = 0,
	Y_POS,
	Z_POS,
	X_NEG,
	Y_NEG,
	Z_NEG,
	INJECTION_PORT
};

extern unsigned int netLength;
extern unsigned int netHeight;
extern unsigned int netWidth;
extern Config config;

#define maxP(a,b) ((a)>(b)?(a):(b))
#define minP(a,b) ((a)<(b)?(a):(b))

/*******************************************************
 * Used for signifiying position of a node in 3D grid  *
 * Contains functions to get neighbours of a node and  *
 * also conversion from id to (x,y,z) co-ordinates and *
 * vice versa 					       *
 *******************************************************/

class Position {
	public :
	int x,y,z;

	Position() {}
	Position(int a,int b,int c): x(a),y(b),z(c){};

	void init(int id) {
		z = id / (netLength * netHeight);
		id = id - z*(netLength * netHeight);
		y = id / (netLength);
		x = id - y*netLength;
	}

	void initnew(int a,int b,int c) { x = a; y = b; z = c; }

	Position operator +(int axis) {
		Position ret = *this;
		switch(axis) {  // Use -1 to signify an invalid.node
			case X_POS: if(netLength != 1) ret.x = (x+1)%netLength; break;
			case Y_POS: if(netHeight != 1) ret.y = (y+1)%netHeight; break;
			case Z_POS: if(netWidth != 1) ret.z = (z+1)%netWidth; break;
			case X_NEG: if(netLength != 1) ret.x = (x+netLength-1)%netLength; break;
			case Y_NEG: if(netHeight != 1) ret.y = (y+netHeight-1)%netHeight; break;
			case Z_NEG: if(netWidth != 1) ret.z = (z+netWidth-1)%netWidth; break;
		}
		return ret;
	}

	int getId() { return(x+y*netLength+z*netLength*netHeight); }

	void getNeighbours(int *next) {
		Position tmp = *this,tmp2;
		for(int i=0;i<6;i++) {
			tmp2 = (tmp+i);
			next[i]  = tmp2.getId();
		}
	}

	Position & operator=(const Position& obj) {
		x = obj.x; y = obj.y; z = obj.z;
		return *this;
	}
	int operator != (const Position& obj) {
		if((x != obj.x) || (y != obj.y) || (z != obj.z)) return 1;
		return 0;
	}
};

// Part of packet header
class RoutingInformation {
	public:
	int dst;
	int datalen; // Current packet length
};

// message passed from Node to Nic. For explanation of fields look at class Packet
class NicMsg {
	public:
	RoutingInformation routeInfo;
	int src;
	int pktId;
	int msgId;
	int totalLen;   // Total length of message. is constant across packets of a message
	int destNodeCode;
	int index;
	int destTID;
	POSE_TimeType recvTime;
	POSE_TimeType origovt;

	NicMsg(){}
	NicMsg & operator=(const NicMsg &obj)
	{
		eventMsg::operator=(obj);
		src = obj.src;  routeInfo = obj.routeInfo;
		pktId = obj.pktId; msgId = obj.msgId;  
		totalLen = obj.totalLen;  origovt = obj.origovt;
		destNodeCode = obj.destNodeCode;
		index = obj.index;
		destTID = obj.destTID; 
		recvTime = obj.recvTime; 
		return *this;
	}
};

#define MAX_ROUTE_HEADER 128  // This variable sucks
#define CONTROL_PACKET 0x01
#define DATA_PACKET 0x02
#define SUBNET_MANAGER 0

// Part of the packet. Contains all neccessary details required for protocol and routing
class Header {
        public:
	RoutingInformation routeInfo;  // routing informating
	unsigned char nextPort[MAX_ROUTE_HEADER]; // next port to take 
	unsigned char hop;
        int src,prev_src;              // original source and previous node id
        int pktId;		       // packet id , not used in inorder arrival
        int msgId;		       // message id, set by node
        int totalLen,nextId,prevId;    // total length of message
        int portId; 		       // output port id
        int vcid,prev_vcid;	       // vcid = vcid at input , prev_vcid = prev output global vcid
	int controlInfo;
        Header(){} // most common case
        Header & operator=(const Header &obj)
        {
                src = obj.src; routeInfo = obj.routeInfo;
                pktId = obj.pktId; msgId = obj.msgId; 
                totalLen = obj.totalLen; prevId = obj.prevId;
                portId = obj.portId; nextId = obj.nextId;
                vcid = obj.vcid; prev_vcid = obj.prev_vcid;
		prev_src = obj.prev_src; hop = obj.hop; controlInfo = obj.controlInfo;
		memcpy(nextPort, obj.nextPort, MAX_ROUTE_HEADER * sizeof(char));
                return *this;
        }

        Header & operator=(const NicMsg &obj)
        {
		src = obj.src;  routeInfo = obj.routeInfo;
		pktId = obj.pktId; msgId = obj.msgId;  
		totalLen = obj.totalLen;  
                return *this;
        }

        void dump()
        {
                CkPrintf("HEADER src %d dst %d portId %d msgId %d \n",
                                src,routeInfo.dst,portId,msgId);
        }

        bool operator==(const Header &obj) const {
                if(src==obj.src && routeInfo.dst==obj.routeInfo.dst && pktId==obj.pktId && msgId==obj.msgId)  return true;  else  return false;
        }

        bool operator < (const Header &obj) const {
                if(msgId < obj.msgId) return true; else
                if((msgId == obj.msgId) && (pktId < obj.pktId)) return true; else
		if((msgId == obj.msgId) && (pktId == obj.pktId) && (src < obj.src)) return true;
                else return false;
        }
};

// Packet message
class Packet {
	public:
	Header hdr;
	Packet(){}
	Packet & operator=(const Packet &obj)
	{
		eventMsg::operator=(obj);
		hdr = obj.hdr;
		return *this;
	}

        void dump() { hdr.dump(); }

};


// Used to store message details directly at the destination, bypassing intermediate nodes
// For details of fields, look at class Header or TCsim.C
class MsgStore {
        public:
        int src;
        int pktId;
        int msgId;
        int totalLen;
        POSE_TimeType recvTime;
        int destNodeCode;
        int index;
        int destTID;
	POSE_TimeType origovt;

        MsgStore(){}

        MsgStore & operator=(const MsgStore &obj)
        {
                src = obj.src; pktId = obj.pktId; msgId = obj.msgId;  origovt = obj.origovt;
                totalLen = obj.totalLen; destNodeCode = obj.destNodeCode;
                index = obj.index; destTID = obj.destTID; recvTime = obj.recvTime; 
                return *this;
        }

        MsgStore & operator=(const NicMsg &obj)
        {
                src = obj.src; pktId = obj.pktId; msgId = obj.msgId;  origovt = obj.origovt;
                totalLen = obj.totalLen; 
                destNodeCode = obj.destNodeCode; index = obj.index;
                destTID = obj.destTID; recvTime = obj.recvTime; 
	}

        bool operator==(const MsgStore &obj) const {
                if(src==obj.src && pktId==obj.pktId && msgId==obj.msgId)  return true;  else  return false;
        }
};

// Message to restart stopped flow at buffers
class flowStart {
        public:
	int nextId; 
        int vcid;
	int prev_vcid;
	int datalen;
        flowStart(){}
        flowStart & operator=(const flowStart &obj)
        {
        eventMsg::operator=(obj);
        vcid = obj.vcid;
	prev_vcid = obj.prev_vcid;
	nextId = obj.nextId;
	datalen = obj.datalen;
        return *this;
        }
	void dump() {
	CkPrintf("nextId = %d vcid = %d prev_vcid = %d datalen = %d \n",
		nextId,vcid,prev_vcid,datalen);
	}
};

// Unique id to recognize a message, comprising source and message id
class remoteMsgId
{
	public:
	int msgId;
	int nodeId;

	remoteMsgId(){}
	remoteMsgId(int m,int s):msgId(m),nodeId(s){}
	remoteMsgId & operator=(const remoteMsgId &obj)
	{	msgId = obj.msgId;	nodeId = obj.nodeId;  return *this;  }

	bool operator<(const remoteMsgId &obj) const {
		if(nodeId < obj.nodeId) return true; else 
		if((nodeId == obj.nodeId) && (msgId < obj.msgId)) return true;
		else return false;
	}
	bool operator==(const remoteMsgId &obj) const {
		if(msgId==obj.msgId  && nodeId==obj.nodeId) return true;  else  return false;
	}
};

// Message while initializing a network
class NetInterfaceMsg {
	public:
	int id;
	int startId; // startId of next switch
	int numP;
	
	NetInterfaceMsg(){}
	NetInterfaceMsg(int i,int start,int np):
		id(i),startId(start),numP(np){}

	~NetInterfaceMsg(){}	

	NetInterfaceMsg& operator = (const NetInterfaceMsg& obj) {
		eventMsg::operator=(obj);
		id = obj.id;
		startId = obj.startId;
		numP = obj.numP;
		return *this;
	}
		
};

// Message while initializing a channel
class ChannelMsg {
	public:
	int id;
	int portid;
	int nodeid;
	int numP;
	
	ChannelMsg(){}
	ChannelMsg(int i,int pid, int nid,int nump):
		id(i),portid(pid),nodeid(nid),numP(nump){}

	~ChannelMsg(){}	

	ChannelMsg& operator = (const ChannelMsg& obj) {
		eventMsg::operator=(obj);
		id = obj.id;
		portid = obj.portid;
		nodeid = obj.nodeid;
		numP = obj.numP;
		return *this;
	}
};		

// Message while initializing a Switch
class SwitchMsg {
	public:
	int id;
	int numP;
	SwitchMsg(){}
	SwitchMsg(int i,int p):id(i),numP(p){}
	~SwitchMsg(){}
	SwitchMsg &operator = (const SwitchMsg &obj) {
		eventMsg::operator=(obj);
		id = obj.id;
		numP = obj.numP;
		return *this;
	}
};

#define NO_VC_AVAILABLE -1

// Constants in a NIC poser, which are not stored during checkpointing
class NicConsts {
	public:
	int id;
	Position pos;
	int startId,numP;
};

#ifndef compile
#include "../Topology/MainTopology.h"
#include "../Routing/MainRouting.h"
#include "../OutputVcSelection/MainOutputVcSelection.h"
#include "../InputVcSelection/MainInputVcSelection.h"

class Switch {
	public:
	map <int,int> availBufsize;	 // Downstream buffer sizes .. Note this is not current switch buffer sizes.
	map <int,int> mapVc;     // map input to output
	map <int,vector <Header> > Buffer;  // Save the packets when path is stalled
	map <int,int> requested;  // Used to do head of line blocking
	unsigned char routeTable[MAX_ROUTE_HEADER]; // next port to take 
	
	int id,numP;
        unsigned  char InputRoundRobin,RequestRoundRobin,AssignVCRoundRobin;
	// Be careful not to put variable data in any of these. Rollback will kill the simulation
        Topology *topology;  // Topology of machine
        RoutingAlgorithm *routingAlgorithm; // Routing strategy
        OutputVcSelection *outputVcSelect; // Output VC selection strategy
        InputVcSelection *inputVcSelect;  // Input VC selection strategy

	Switch(){}
	Switch(SwitchMsg *m);

	void recvPacket(Packet *);
	void recvPacket_anti(Packet *){restore(this);}
	void recvPacket_commit(Packet *){}
	void updateCredits(flowStart *);
	void updateCredits_anti(flowStart *){restore(this);}
	void updateCredits_commit(flowStart *){}
	void checkNextPacketInVc(flowStart *);
	void checkNextPacketInVc_anti(flowStart *){restore(this);}
	void checkNextPacketInVc_commit(flowStart *){}
	void sendPacket(Packet *,const int &,const int &,const int &);
	
	~Switch(){}

	Switch& operator=(const Switch& obj) {
	rep::operator=(obj);
	availBufsize = obj.availBufsize; InputRoundRobin = obj.InputRoundRobin; RequestRoundRobin = obj.RequestRoundRobin;  
	Buffer = obj.Buffer; AssignVCRoundRobin = obj.AssignVCRoundRobin; topology = obj.topology; 
	requested = obj.requested; mapVc = obj.mapVc;
	memcpy(routeTable, obj.routeTable, MAX_ROUTE_HEADER * sizeof(char));
	return *this;
	}
	bool operator==(const Switch& obj) {
	return(id==obj.id);
	}
};

// Should take care of contention when multiple ports are sending data to nic ...
class NetInterface {
	public:
	map <remoteMsgId,int > pktMap; // Used to keep track of packets received from various sources
	map <remoteMsgId,MsgStore> storeBuf; // Used to store part of incoming messages directly. See class MsgStore for details
	int numRecvd,roundRobin; // roundRobin is used for load balancing packets.
	POSE_TimeType prevIntervalStart,counter; // These are used for statistics printing
	Topology *topology; 
	RoutingAlgorithm *routingAlgorithm;
	
	NicConsts *nicConsts;

	NetInterface(){}
	NetInterface(NetInterfaceMsg *niMsg);

	void recvMsg(NicMsg *);
	void recvMsg_anti(NicMsg *){restore(this);}
	void recvMsg_commit(NicMsg *){}
	void recvPacket(Packet *);
	void recvPacket_anti(Packet *){restore(this);}
	void recvPacket_commit(Packet *){}
	void storeMsgInAdvance(NicMsg *);
	void storeMsgInAdvance_anti(NicMsg *){restore(this);}
	void storeMsgInAdvance_commit(NicMsg *){}
	~NetInterface(){} 

	NetInterface& operator=(const NetInterface& obj) {
	rep::operator=(obj);
	pktMap = obj.pktMap; numRecvd = obj.numRecvd;  nicConsts = obj.nicConsts; storeBuf = obj.storeBuf;
	prevIntervalStart = obj.prevIntervalStart; counter = obj.counter; topology = obj.topology;roundRobin = obj.roundRobin;
	return *this;
	}

	bool operator==(const NetInterface& obj) {
	return(nicConsts->id==((obj.nicConsts)->id));
	}
};

#endif
// Channel poser
class Channel {
	public:
	int id;
	POSE_TimeType prevIntervalStart,counter;
	int portid,nodeid,numP; // To do statistics collection for link 

	Channel() {}
	~Channel(){}
	Channel(ChannelMsg *m);
	void recvPacket(Packet *);
	void recvPacket_anti(Packet *){restore(this);}
	void recvPacket_commit(Packet *){}

	Channel& operator=(const Channel& obj) {
	rep::operator=(obj);
	id=obj.id;  portid = obj.portid; 
	prevIntervalStart = obj.prevIntervalStart; counter = obj.counter;nodeid =obj.nodeid;numP = obj.numP;
	return *this;
	}

	bool operator==(const Channel& obj) {
	return(id==obj.id);
	}
};
