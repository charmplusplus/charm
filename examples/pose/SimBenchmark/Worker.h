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
  int numObjs, numMsgs, msgSize, locality, density, grainSize;
  double granularity;
  WorkerData& operator=(const WorkerData& obj) {
    int i;
    eventMsg::operator=(obj);
    numObjs = obj.numObjs;
    numMsgs = obj.numMsgs;
    msgSize = obj.msgSize;
    locality = obj.locality;
    grainSize = obj.grainSize;
    granularity = obj.granularity;
    density = obj.density;
    return *this;
  }
};

class SmallWorkMsg {
 public:
  int data[SM_MSG_SZ];
  int fromPE;
  SmallWorkMsg& operator=(const SmallWorkMsg& obj) {
    eventMsg::operator=(obj);
    for (int i=0; i<SM_MSG_SZ; i++) data[i] = obj.data[i];
    fromPE = obj.fromPE;
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
  int numObjs, numMsgs, msgSize, locality, grainSize, density, sent, totalObjs,
    localMsgs, remoteMsgs, localNbr, remoteNbr, localCount, remoteCount,
    fromLocal, fromRemote, received;
  double granularity, localDensity;
  int data[100];
 public:
  worker();
  worker(WorkerData *m); 
  ~worker() { }
  worker& operator=(const worker& obj);
  void pup(PUP::er &p) { 
    chpt<state_worker>::pup(p); 
    p(numObjs); p(numMsgs); p(msgSize); p(density); p(localDensity);
    p(locality); p(grainSize); p(granularity); p(sent); p(totalObjs); 
    p(localMsgs); p(remoteMsgs); p(localNbr); p(remoteNbr); p(received);
    p(localCount); p(remoteCount); p(fromLocal); p(fromRemote);
    p(data, 100);
  }
  void dump() {
    chpt<state_worker>::dump();
    CkPrintf("[worker: ");
  }
  void doWork();
  void terminus();

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
};

