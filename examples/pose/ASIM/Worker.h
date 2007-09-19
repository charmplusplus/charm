// Message size types
#define MIX_MS 0
#define SMALL 1
#define MEDIUM 2
#define LARGE 3

// Message sizes
#define SM_MSG_SZ 10
#define MD_MSG_SZ 100
#define LG_MSG_SZ 1000

// Grain sizes
#define MIX_GS 0
#define FINE 1
#define MEDIUM_GS 2
#define COARSE 3

// Granularity
#define FINE_GRAIN   0.000010
#define MEDIUM_GRAIN 0.001000
#define COARSE_GRAIN 0.010000

// Distribution types
#define RANDOM 0
#define IMBALANCED 1
#define UNIFORM 2

// Connectivity types
#define SPARSE 0
#define HEAVY 1
#define FULL 2

class WorkerData {
 public:
  int numObjs, numMsgs, msgSize, distribution, connectivity, locality;
  int grainSize, elapsePattern, offsetPattern, sendPattern;
  double granularity;
  POSE_TimeType elapseTimes[5];
  int numSends[5], offsets[5], neighbors[100];
  int numNbrs;
  WorkerData& operator=(const WorkerData& obj) {
    int i;
    eventMsg::operator=(obj);
    numObjs = obj.numObjs;
    numMsgs = obj.numMsgs;
    msgSize = obj.msgSize;
    distribution = obj.distribution;
    connectivity = obj.connectivity;
    locality = obj.locality;
    grainSize = obj.grainSize;
    elapsePattern = obj.elapsePattern;
    offsetPattern = obj.offsetPattern;
    sendPattern = obj.sendPattern;
    granularity = obj.granularity;
    for (i=0; i<5; i++) {
      elapseTimes[i] = obj.elapseTimes[i];
      numSends[i] = obj.numSends[i];
      offsets[i] = obj.offsets[i];
    }
    for (i=0; i<100; i++)  neighbors[i] = obj.neighbors[i];
    numNbrs = obj.numNbrs;
    return *this;
  }
  void dump() {
    int i;
    CkPrintf("#Nbrs=%d [", numNbrs);
    for (i=0; i<numNbrs; i++) {
      CkPrintf("%d", neighbors[i]);
      if (i != numNbrs-1) CkPrintf(", ");
    }
    CkPrintf("]\n");
    CkPrintf("Elapse Offset #Sends\n");
    for (i=0; i<5; i++)
      CkPrintf("%6d %6d %6d\n", elapseTimes[i], offsets[i], numSends[i]);
  }
};

class SmallWorkMsg {
 public:
  int data[SM_MSG_SZ];
  SmallWorkMsg& operator=(const SmallWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<SM_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class MediumWorkMsg {
 public:
  int data[MD_MSG_SZ];
  MediumWorkMsg& operator=(const MediumWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<MD_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class LargeWorkMsg {
 public:
  int data[LG_MSG_SZ];
  LargeWorkMsg& operator=(const LargeWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<LG_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};


class worker {
  int numObjs, numMsgs, msgSize, distribution, connectivity, locality;
  int grainSize, elapsePattern, offsetPattern, sendPattern;
  double granularity;
  int elapseTimes[5], numSends[5], offsets[5], neighbors[100];
  int numNbrs, elapseIdx, sendIdx, nbrIdx, offsetIdx, msgIdx, gsIdx;
  int data[100];
 public:
  worker();
  worker(WorkerData *m); 
  ~worker() { }
  worker& operator=(const worker& obj);
  void pup(PUP::er &p) { 
    chpt<state_worker>::pup(p); 
    p(numObjs); p(numMsgs); p(msgSize); p(distribution); p(connectivity);
    p(locality); p(grainSize); p(elapsePattern); p(offsetPattern); 
    p(sendPattern); p(granularity);
    p(elapseTimes, 5); p(numSends, 5); p(offsets, 5); p(neighbors, 100);
    p(numNbrs); p(elapseIdx); p(sendIdx); p(nbrIdx); p(offsetIdx); 
    p(msgIdx); p(gsIdx);
    p(data, 100);
  }
  void dump() {
    chpt<state_worker>::dump();
    CkPrintf("[worker: ");
  }
  void doWork();

  // Event methods
  void workSmall(SmallWorkMsg *m);
  void workSmall_anti(SmallWorkMsg *m);
  void workSmall_commit(SmallWorkMsg *m);
  void workMedium(MediumWorkMsg *m);
  void workMedium_anti(MediumWorkMsg *m);
  void workMedium_commit(MediumWorkMsg *m);
  void workLarge(LargeWorkMsg *m);
  void workLarge_anti(LargeWorkMsg *m);
  void workLarge_commit(LargeWorkMsg *m);
  void terminus(){}
};

