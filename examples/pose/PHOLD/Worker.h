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

class WorkerData {
 public:
  int numObjs, numMsgs, tscale, locality, density, grainSize;
  double granularity;
  WorkerData& operator=(const WorkerData& obj) {
    int i;
    eventMsg::operator=(obj);
    numObjs = obj.numObjs;
    numMsgs = obj.numMsgs;
    tscale = obj.tscale;
    locality = obj.locality;
    grainSize = obj.grainSize;
    granularity = obj.granularity;
    return *this;
  }
};

class WorkMsg {
 public:
  int fromPE;
  WorkMsg& operator=(const WorkMsg& obj) {
    eventMsg::operator=(obj);
    fromPE = obj.fromPE;
    return *this;
  }
};

class worker {
  int numObjs, numMsgs, tscale, locality, grainSize;
  double granularity;
 public:
  worker();
  worker(WorkerData *m); 
  ~worker() { }
  worker& operator=(const worker& obj);
  void pup(PUP::er &p) { 
    chpt<state_worker>::pup(p); 
    p(numObjs); p(numMsgs); p(tscale); p(locality); p(grainSize); 
    p(granularity);
  }
  void cpPup(PUP::er &p) { 
    p(numObjs); p(numMsgs); p(tscale); p(locality); p(grainSize); 
    p(granularity);
  }

  // Event methods
  void terminus(){ 
    //CkPrintf("called terminus\n");
  }
  void work(WorkMsg *m);
  void work_anti(WorkMsg *m);
  void work_commit(WorkMsg *m);
};

