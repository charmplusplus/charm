/*
Load-balancing test program:
  Orion Sky Lawlor, 10/19/1999

  Added more complex comm patterns
  Robert Brunner, 11/3/1999

  updated by
  Gengbin Zheng
*/

#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "LBDatabase.h"
#include "Topo.h"
#include "CentralLB.h"
#include "DummyLB.h"
#include "RandCentLB.h"
//#include "RecBisectBfLB.h"
#include "RefineLB.h"
#include "RefineCommLB.h"
#include "GreedyCommLB.h"
#include "MetisLB.h"
#include "GreedyLB.h"
#include "NeighborLB.h"
#include "WSLB.h"
#include "OrbLB.h"
#include "HybridLB.h"
#include "HbmLB.h"
#include "GraphPartLB.h"
#include "GraphBFTLB.h"

#include "lb_test.decl.h"

#if defined(_WIN32)
#define strcasecmp stricmp
#endif

CkChareID mid;//Main ID
CkGroupID topoid;
CProxy_Lb_array hproxy;//Array ID
int n_loadbalance;

#define N_LOADBALANCE 500 /*Times around ring until we load balance*/

#define DEBUGF(x)       // CmiPrintf x
int cycle_count,element_count,step_count,print_count;
int min_us,max_us;

int specialTracing = 0;

void initialize()
{
  if (traceIsOn() == 0) {
    if (CkMyPe() == 0)
    CkPrintf("traceprojections was off at initial time.\n");
    specialTracing = 1;
  }
}

class HiMsg : public CMessage_HiMsg {
public:
  int length;
  int chksum;
  int refnum;
  char* data;
};

class main : public CBase_main {
public:
  int nDone;

  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m);

  void maindone(void) {
#if 0
    nDone++;
    if (nDone==element_count) {
      CkPrintf("All done\n");
      CkExit();
    }
#endif
    CkPrintf("All done\n");
    CkExit();
  };

private:
  void arg_error(char* argv0);
};

static const struct {
  const char *name;//Name of strategy (on command line)
  const char *description;//Text description of strategy
  LBCreateFn create;//Strategy routine
} StratTable[]={
  {"none",
   "none - The null load balancer, collect data, print data",
   CreateCentralLB},
  {"dummy",
   "dummy - The null load balancer, collect data, do nothing",
   CreateDummyLB},
  {"neighbor",
   "neighbor - The neighborhood load balancer",
   CreateNeighborLB},
  {"workstation",
   "workstation - Like neighbor, but for workstation performance",
   CreateWSLB},
  {"random",
   "random - Assign objects to processors randomly",
   CreateRandCentLB},
  {"greedy",
   "greedy - (Greedy) Use the greedy algorithm to place heaviest object on the "
   "least-loaded processor until done",
   CreateGreedyLB},
  {"metis",
   "metis - Use Metis(tm) to partition object graph",
   CreateMetisLB},
  {"refine",
   "refine - Move a very few objects away from most heavily-loaded processor",
   CreateRefineLB},
  {"refinecomm",
   "refinecomm - Move a very few objects away from most heavily-loaded processor, taking communicaiton into account",
   CreateRefineCommLB},
  {"orb",
   "orb - Orthogonal Recursive Bisection to partition according to coordinates",
   CreateOrbLB},
  {"comm",
   "comm - Greedy with communication",
   CreateGreedyCommLB},
  /*{"recbf",
   "recbf - Recursive partitioning with Breadth first enumeration, with 2 nuclei",
   CreateRecBisectBfLB},*/
  {"hybrid",
   "hybrid - Hybrid strategy",
   CreateHybridLB},
  {"hbm",
   "HbmLB - HBM strategy",
   CreateHbmLB},
  {"graphpart",
   "GraphPartLB - graph partitioning",
   CreateGraphPartLB},
  {"graphbft",
   "GraphBFTLB - breadth first traversal",
   CreateGraphBFTLB},

  {NULL,NULL,NULL}
};

static void programBegin(void *dummy,int size,void *data)
{
  //Start everybody computing
/*
  for (int i=0;i<element_count;i++)
    hproxy[i].ForwardMessages();
*/
  hproxy.ForwardMessages();
}

main::main(CkArgMsg *m) 
{
  char *strategy;//String name for strategy routine
  char *topology;//String name for communication topology
  int stratNo;
  nDone=0;

  int cur_arg = 1;

  if (m->argc > cur_arg)
    element_count=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    step_count=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);
  
  if (m->argc > cur_arg)
    print_count=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);
  
  if (m->argc > cur_arg)
    n_loadbalance=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    min_us=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    max_us=atoi(m->argv[cur_arg++]);
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    strategy=m->argv[cur_arg++];
  else arg_error(m->argv[0]);

  if (m->argc > cur_arg)
    topology=m->argv[cur_arg++];
  else arg_error(m->argv[0]);

  //Look up the user's strategy in table
  stratNo=0;
  while (StratTable[stratNo].name!=NULL) {
    if (0==strcasecmp(strategy,StratTable[stratNo].name)) {
      //We found the user's chosen strategy!
      StratTable[stratNo].create();//Create this strategy
      break;
    }
    stratNo++;
  }

  CkPrintf("%d processors\n",CkNumPes());
  CkPrintf("%d elements\n",element_count);
  CkPrintf("Print every %d steps\n",print_count);
  CkPrintf("Sync every %d steps\n",n_loadbalance);
  CkPrintf("First node busywaits %d usec; last node busywaits %d usec\n",
	   min_us,max_us);

  mid = thishandle;


  if (StratTable[stratNo].name==NULL)
    //The user's strategy name wasn't in the table-- bad!
    CkAbort("ERROR! Strategy not found!  \n");
	
  topoid = Topo::Create(element_count,topology,min_us,max_us);
  if (topoid.isZero())
    CkAbort("ERROR! Topology not found!  \n");

  hproxy = CProxy_Lb_array::ckNew(element_count);
  hproxy.setReductionClient(programBegin, NULL);

/*
  //Start everybody computing
  for (int i=0;i<element_count;i++)
    hproxy[i].ForwardMessages();
*/
}

void main::arg_error(char* argv0)
{
  CkPrintf("Usage: %s \n"
    "<elements> <steps> <print-freq> <lb-freq> <min-dur us> <max-dur us>\n"
    "<strategy> <topology>\n"
    "<strategy> is the load-balancing strategy:\n",argv0);
  int stratNo=0;
  while (StratTable[stratNo].name!=NULL) {
    CkPrintf("  %s\n",StratTable[stratNo].description);
    stratNo++;
  }

  int topoNo = 0;
  CkPrintf("<topology> is the object connection topology:\n");
  while (TopoTable[topoNo].name) {
    CkPrintf("  %s\n",TopoTable[topoNo].desc);
    topoNo++;
  }

  CmiPrintf("\n"
	   " The program creates a ring of element_count array elements,\n"
	   "which all compute and send to their neighbor cycle_count.\n"
	   "Computation proceeds across the entire ring simultaniously.\n"
	   "Orion Sky Lawlor, olawlor@acm.org, PPL, 10/14/1999\n");
  CmiAbort("Abort!");
}

class Lb_array : public CBase_Lb_array {
public:
  Lb_array(void) {
    //    CkPrintf("Element %d created\n",thisIndex);

    //Find out who to send to, and how many to receive
    TopoMap = CProxy_Topo::ckLocalBranch(topoid);
    send_count = TopoMap->SendCount(thisIndex);
    send_to = new Topo::MsgInfo[send_count];
    TopoMap->SendTo(thisIndex,send_to);
    recv_count = TopoMap->RecvCount(thisIndex)+1;
    
    // Benchmark the work function
    work_per_sec = CalibrateWork();

    //Create massive load imbalance by making load
    // linear in processor number.
    usec = (int)TopoMap->Work(thisIndex);
    DEBUGF(("Element %d working for %d ms\n",thisIndex,usec));

    //msec=meanms+(devms-meanms)*thisIndex/(element_count-1);

    // Initialize some more variables
    nTimes=0;
    sendTime=0;
//    lastTime=CmiWallTimer();
    n_received = 0;
    resumed = 1;
    busywork = (int)(usec*1e-6*work_per_sec);
    
    int i;
    for(i=0; i < future_bufsz; i++)
      future_receives[i]=0;
	
    usesAtSync=CmiTrue;

    contribute(sizeof(i), &i, CkReduction::sum_int);
  }

  //Packing/migration utilities
  Lb_array(CkMigrateMessage *m) {
    DEBUGF(("Migrated element %d to processor %d\n",thisIndex,CkMyPe()));
    TopoMap = CProxy_Topo::ckLocalBranch(topoid);
    //Find out who to send to, and how many to receive
    send_count = TopoMap->SendCount(thisIndex);
    send_to = new Topo::MsgInfo[send_count];
    TopoMap->SendTo(thisIndex,send_to);
    recv_count = TopoMap->RecvCount(thisIndex)+1;
    resumed = 0;
    lastTime = CmiWallTimer();
  }

  virtual void pup(PUP::er &p)
  {
	ArrayElement1D::pup(p);//<- pack our superclass
	p(nTimes);p(sendTime);
	p(usec);//p(lastTime);
	p(work_per_sec);
	p(busywork);
	p(n_received);
	p(future_receives,future_bufsz);
  }

  void Compute(HiMsg *m) { 
    //Perform computation
    if (m->refnum > nTimes) {
      // CkPrintf("[%d] Future message received %d %d\n", thisIndex,nTimes,m->refnum);
      int future_indx = m->refnum - nTimes - 1;
      if (future_indx >= future_bufsz) {
	CkPrintf("[%d] future_indx is too far in the future %d, expecting %d, got %d\n",
		 thisIndex,future_indx,nTimes,m->refnum);
	thisProxy[thisIndex].Compute(m);
      } else {
	future_receives[future_indx]++;
	delete m;
      }
    } else if (m->refnum < nTimes) {
      CkPrintf("[%d] Stale message received %d %d\n",
	       thisIndex,nTimes,m->refnum);
      delete m;
    } else {
      n_received++;

      //      CkPrintf("[%d] %d n_received=%d of %d\n",
      //      	       CkMyPe(),thisIndex,n_received,recv_count);
      if (n_received == recv_count) {
	//	CkPrintf("[%d] %d computing %d\n",CkMyPe(),thisIndex,nTimes);

	if (nTimes && nTimes % print_count == 0) {
	  //Write out the current time
	  if (thisIndex==1) {
	    double now = CmiWallTimer();
	    CkPrintf("GREP00\t%d\t%lf\t%lf\n",
		     nTimes,now,now-lastTime);
	    lastTime=now;
	  }
	}

	n_received = future_receives[0];

	// Move all the future_receives down one slot
	int i;
	for(i=1;i<future_bufsz;i++)
	  future_receives[i-1] = future_receives[i];
	future_receives[future_bufsz-1] = 0;

	nTimes++;//Increment our "times around"	

	double startTime=CmiWallTimer();
	// First check contents of message
	//     int chksum = 0;
	//     for(int i=0; i < m->length; i++)
	//       chksum += m->data[i];
      
	//     if (chksum != m->chksum)
	//       CkPrintf("Checksum mismatch! %d %d\n",chksum,m->chksum);

	//Do Computation:
	work(busywork,&result);
	
	int loadbalancing = 0;
	if (nTimes == step_count) {
	  //We're done-- send a message to main telling it to die
          CkCallback cb(CkIndex_main::maindone(), mid);
          contribute(0, NULL, CkReduction::sum_int, cb);
	} else if (nTimes % n_loadbalance == 0) {
          if (specialTracing) {
            if (nTimes/n_loadbalance == 1) traceBegin();
            if (nTimes/n_loadbalance == 3) traceEnd();
          }
	  //We're not done yet...
	  //Either load balance, or send a message to the next guy
	  DEBUGF(("Element %d AtSync on PE %d\n",thisIndex,CkMyPe()));
#if 0
	  AtSync();
#else
          CkCallback cb(CkIndex_Lb_array::pause(), thisProxy);
          contribute(0, NULL, CkReduction::sum_int, cb);
#endif
	  loadbalancing = 1;
	} else ForwardMessages();
      }
      delete m;
    }
  }

  void ResumeFromSync(void) { //Called by Load-balancing framework
#if 0
    resumed = 1;
    DEBUGF(("Element %d resumeFromSync on PE %d\n",thisIndex,CkMyPe()));
    thisProxy[thisIndex].ForwardMessages();
#else
    CkCallback cb(CkIndex_Lb_array::restart(), thisProxy);
    contribute(0, NULL, CkReduction::sum_int, cb);
#endif
  }

  void pause() {
    AtSync();
  }

  void restart() {
    resumed = 1;
    lastTime = CmiWallTimer();
    DEBUGF(("Element %d resumeFromSync on PE %d\n",thisIndex,CkMyPe()));
    thisProxy[thisIndex].ForwardMessages();
  }

  void ForwardMessages(void) { //Pass it on
    if (sendTime == 0) lastTime = CmiWallTimer();

    if (resumed != 1)
      CkPrintf("[%d] %d forwarding %d %d %d\n",CkMyPe(),thisIndex,
	       sendTime,nTimes,resumed);
    for(int s=0; s < send_count; s++) {
      int msgbytes = send_to[s].bytes;
      if (msgbytes != 1000)
	CkPrintf("[%d] %d forwarding %d bytes (%d,%d,%p) obj %p to %d\n",
		 CkMyPe(),thisIndex,msgbytes,s,send_count,send_to,
		 this,send_to[s].obj);
      HiMsg* msg = new(msgbytes,0) HiMsg;
      msg->length = msgbytes;
      //      msg->chksum = 0;
      //      for(int i=0; i < msgbytes; i++) {
      //	msg->data[i] = i;
      //	msg->chksum += msg->data[i];
      //      }
      msg->refnum = sendTime;

      //      CkPrintf("[%d] %d sending to %d at %d:%d\n",
      //	       CkMyPe(),thisIndex,send_to[s].obj,nTimes,nCycles);
      thisProxy[send_to[s].obj].Compute(msg);
    }
    int mybytes=1;
    HiMsg* msg = new(mybytes,0) HiMsg;
    msg->length = mybytes;
    msg->refnum = sendTime;
    thisProxy[thisIndex].Compute(msg);

    sendTime++;
  }

private:
  int CalibrateWork() {
    static int calibrated=-1;

    if (calibrated != -1) return calibrated;
    const double calTime=0.05; //Time to spend in calibration
    double wps = 0;
    // First, count how many iterations for 1 second.
    // Since we are doing lots of function calls, this will be rough
    const double end_time = CmiWallTimer()+calTime;
    wps = 0;
    while(CmiWallTimer() < end_time) {
      work(100,&result);
      wps+=100;
    }

    // Now we have a rough idea of how many iterations there are per
    // second, so just perform a few cycles of correction by
    // running for what we think is 1 second.  Then correct
    // the number of iterations per second to make it closer
    // to the correct value

#if 0
    CkPrintf("[%d] Iter  %.0f per second\n",CmiMyPe(), wps);
#endif
    for(int i=0; i < 2; i++) {
      const double start_time = CmiWallTimer();
      work((int)wps,&result);
      const double end_time = CmiWallTimer();
      const double correction = calTime / (end_time-start_time);
      wps *= correction;
#if 0
      CkPrintf("Iter %d -> %.0f per second\n",i,wps);
#endif
    }
    calibrated = (int)(wps/calTime);
    if (CkMyPe() == 0) CkPrintf("calibrated iterations %d\n",calibrated);
    return calibrated;
  };

  void work(int iter_block,int* _result) {
    *_result=0;
    for(int i=0; i < iter_block; i++) {
      *_result=(int)(sqrt(1+cos(*_result*1.57)));
    }
  };

public:
  enum { future_bufsz = 50 };

private:
  int nTimes;//Number of times I've been called
  int sendTime;//Step number for sending (in case I finish receiving
               //before sending
  int usec;//Milliseconds to "compute"
  double lastTime;//Last time recorded
  int work_per_sec;
  int busywork;
  int result;

  Topo* TopoMap;
  int send_count;
  int recv_count;
  Topo::MsgInfo* send_to;
  int n_received;
  int future_receives[future_bufsz];
  int resumed;
};

#include "lb_test.def.h"

