#ifndef __UTIL_H
#define  __UTIL_H

#include <stdlib.h>
#include <math.h>
#include "charm++.h"

class Config {
public:
   int maxpacksize;
   int switchVc;
   int switchBufsize;
   float switchC_BW;
   int switchC_Delay;
   int collection_interval;
   int linkstats_on;
   int msgstats_on;
   int netsim_on;
   int skip_on;
   int check_on;
   int InputBufferStart;
   int nicStart;
   int ChannelStart;
   int numNodes;
   int origNodes;
   int switchStart;
   int numP;
   int receptionSerial;
   int inputSpeedup;
   int use_transceiver;
   int inputBuffering;

  int NodeStart;
  int HCAStart;
  int DMAchannelStart;
  int SwitchStart;
  int VCStart;
  int CMgrStart;       //The collective manager

  int numVC;           // Number of Virtual Channels
  int maxPacketSize;   // Maximum Packet Size (bytes)
  int creditSize;      // Number of bytes per credit
  int headerSize;      // Size of header in each packet
  int numSwitches;     // Computed from number of nodes
  int VCBufSize;       // Size of Buffer for each VC in 'Credits'
  int HCABufSize;       // Size of Buffer for each NCA in 'Credits'
  int channelPropDelay;   //Channel delay in ns
  int channelBandwidth;   //Channel bandwidth in GB/s
  int switchDelay;        //Switch delay us
  int DMADelaySmall;      //Scheduling delay for the DMA for short messages
  int DMADelayLarge;      //Scheduling delay for the DMA for large messages
  int numPorts;           // Number of ports in a switch
  int numRails;           // Number of fattrees connected to a node

  int HCASendDelay;     //Computation time in HCA
  int HCARecvDelay;     //HCA Receive overhead

  int procSendOverhead;  //Processor overhead of sending messages
  int procRecvOverhead;  //Processor overhead of receiving messages

  int HCAPktSendDelay;     //Computation time in HCA for each packet
  int HCAPktRecvDelay;     //HCA Receive overhead for each packet

  int cacheInjectionThreshold;  //Size of L2 cache. Cache injection
                                //will make message passing much
                                //faster. So small messages can be
                                //sent this way.

  int adaptiveRouting;      //Adaptive routing or static routing
 
 public:

  Config (): check_on(0) {}
  
  void readConfig(CkArgMsg *m);
  int getNumVC           () {return numVC;           }
  int getMaxPacketSize   () {return maxPacketSize;   }
  int getCreditSize      () {return creditSize;      }
  int getHeaderSize      () {return headerSize;      }
  int getNumNodes        () {return numNodes;        }
  int getNumSwitches     () {return numSwitches;     }
  int getVCBufSize       () {return VCBufSize;       }
  int getHCABufSize      () {return HCABufSize;      }
  int getChannelPropDelay() {return channelPropDelay;}
  int getChannelBandwidth() {return channelBandwidth;}
  int getSwitchDelay     () {return switchDelay;     }

  int getDMADelaySmall   () {return DMADelaySmall;   }
  int getDMADelayLarge   () {return DMADelayLarge;   }

  int getNumPorts        () {return numPorts;        }
  int getFanout          () {return numPorts/2;      }
  int getNumRails        () {return numRails;        }

  int getHCASendDelay    () {return HCASendDelay;    }
  int getHCARecvDelay    () {return HCARecvDelay;    }
  int getProcSendOverhead() {return procSendOverhead;}
  int getProcRecvOverhead() {return procRecvOverhead;}

  int getHCAPktSendDelay () {return HCAPktSendDelay; }
  int getHCAPktRecvDelay () {return HCAPktRecvDelay; }

  int getInjectionThreshold(){return cacheInjectionThreshold;}

  int isAdaptiveRouting  () {return adaptiveRouting;}

  int getNodeStart       () {return NodeStart;      }
  int getHCAStart        () {return HCAStart;       }
  int getDMAchannelStart () {return DMAchannelStart;}
  int getSwitchStart     () {return SwitchStart;    }
  int getVCStart         () {return VCStart;        }
  int getChannelStart    () {return ChannelStart;   }
  int getCollectiveManagerStart() {return CMgrStart;}
  void setNodeStart      (int n) {NodeStart       = n;}
  void setHCAStart       (int n) {HCAStart        = n;}
  void setDMAchannelStart(int n) {DMAchannelStart = n;}
  void setSwitchStart    (int n) {SwitchStart     = n;}
  void setVCStart        (int n) {VCStart         = n;}
  void setChannelStart   (int n) {ChannelStart    = n;}
  void setCollectiveManagerStart(int n) {CMgrStart= n;}
  
};

class MachineParams {
public:
	Config *config;
	int procs;
	int BGnodes;
	
	MachineParams() {}
};


PUPbytes(Config);

#endif
