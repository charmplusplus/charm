#include <stdio.h>
#include "pose.h"
#include "pgm.h"
#include "TCsim_sim.h"
#include "BgSim_sim.h"
#include "InitNetwork.h"
#include "util.h"
#include <unistd.h>

Config config;
extern void InitNetwork(MachineParams *);

main::main(CkArgMsg *m)
{ 
  int totalBGProcs, numX, numY, numZ, numCth, numPes;
  int i,myid,procs,mapPE,nodeid,portid,tmp,extra,nnodes,counter2;
  CkGetChareID(&mainhandle);
  CProxy_main M(mainhandle);

  config.readConfig(m);

  BgLoadTraceSummary("bgTrace", totalBGProcs, numX, numY, numZ, numCth, numWth, numPes);
  CkPrintf("bgtrace: totalBGProcs=%d X=%d Y=%d Z=%d #Cth=%d #Wth=%d #Pes=%d\n",
	   totalBGProcs, numX, numY, numZ, numCth, numWth, numPes);
  netLength = numX;
  netHeight = numY;
  netWidth = numZ;

  CkPrintf("Opts: netsim on: %d\n", config.netsim_on);

  POSE_init();

  int BGnodes = netLength*netWidth*netHeight, n=0,l,h,w,switchP;
  config.numNodes = BGnodes;
  config.origNodes = BGnodes;
  Position pos,p;

  procs = CkNumPes();
  config.nicStart = BGnodes*2 ;

  MachineParams *mp; mp = new MachineParams;
  mp->config = &config;
  mp->procs = procs;
  mp->BGnodes = BGnodes;
  InitNetwork(mp);
	
  CkPrintf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",config.maxpacksize,config.switchVc,config.switchBufsize,config.switchC_BW,config.switchC_Delay,config.collection_interval,config.linkstats_on,config.msgstats_on,config.netsim_on,config.skip_on,config.InputBufferStart,config.nicStart,config.ChannelStart,config.numNodes);

	  BGprocMsg *bgm;
	  for (w=0; w<netWidth; w++)
  	     for (h=0; h<netHeight; h++)
    	        for (l=0; l<netLength; l++) {
			p.initnew(l,h,w);
			myid = p.getId();
			myid = procs*myid/BGnodes;	
       	        int switchIdx = (n/numWth)+2*BGnodes;
                BGnodeMsg *nodem = new BGnodeMsg(n, numWth, switchIdx);  // What do I have to put instead of n & switchIdx
		nodem->Timestamp(0);
		
//		CkPrintf("myid is %d procs is %d p.getId() is %d BGnodes is %d \n",myid,procs,p.getId(),BGnodes);	
		(*(CProxy_BGnode *) &POSE_Objects)[(n/numWth)+BGnodes*numWth].insert(nodem,myid);
	
	         for (int i=0; i<numWth; i++) {
		  bgm = new BGprocMsg(n+BGnodes);
		  bgm->Timestamp(0);
	  
		  (*(CProxy_BGproc *) &POSE_Objects)[n].insert(bgm,myid);
		  n++;
		}
      	       }
}
#include "Pgm.def.h"
