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
   int switchC_BW;
   int switchC_Delay;
   int collection_interval;
   int linkstats_on;
   int msgstats_on;
   int netsim_on;
   int skip_on;
   int InputBufferStart;
   int nicStart;
   int ChannelStart;
   int numNodes;
   int switchStart;
   int fanout;
 
 public:

  Config () {}
  
  void readConfig(CkArgMsg *m);
  
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
