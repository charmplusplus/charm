// Message size types
#define MIX_MS 0
#define SMALL 1
#define MEDIUM 2
#define LARGE 3

// Message sizes
#define SM_MSG_SZ 10
#define MD_MSG_SZ 100
#define LG_MSG_SZ 1000

// Distribution types
#define RANDOM 0
#define IMBALANCED 1
#define UNIFORM 2

// Connectivity types
#define SPARSE 0
#define HEAVY 1
#define FULL 2

class TeamData {
 public:
  int numTeams, numWorkers, numObjs;
  TeamData& operator=(const TeamData& obj) {
    eventMsg::operator=(obj);
    numTeams = obj.numTeams;
    numWorkers = obj.numWorkers;
    numObjs = obj.numObjs;
    return *this;
  }
};

class WorkerData {
 public:
  int workerID, numWorkers;
  int numObjs, numMsgs, msgSize, distribution, connectivity, locality;
  int offsetPattern, sendPattern;
  int numSends[5], offsets[5], neighbors[100];
  int numNbrs;
  WorkerData& operator=(const WorkerData& obj) {
    int i;
    eventMsg::operator=(obj);
    workerID = obj.workerID;
    numWorkers = obj.numWorkers;
    numObjs = obj.numObjs;
    numMsgs = obj.numMsgs;
    msgSize = obj.msgSize;
    distribution = obj.distribution;
    connectivity = obj.connectivity;
    locality = obj.locality;
    offsetPattern = obj.offsetPattern;
    sendPattern = obj.sendPattern;
    for (i=0; i<5; i++) {
      numSends[i] = obj.numSends[i];
      offsets[i] = obj.offsets[i];
    }
    for (i=0; i<100; i++)  neighbors[i] = obj.neighbors[i];
    numNbrs = obj.numNbrs;
    return *this;
  }
};

class SmallWorkMsg {
 public:
  int workerID;
  int data[SM_MSG_SZ];
  SmallWorkMsg& operator=(const SmallWorkMsg& obj) {
    eventMsg::operator=(obj);
    workerID = obj.workerID;
    for (int i=0; i<SM_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class MediumWorkMsg {
 public:
  int workerID;
  int data[MD_MSG_SZ];
  MediumWorkMsg& operator=(const MediumWorkMsg& obj) {
    eventMsg::operator=(obj);
    workerID = obj.workerID;
    for (int i=0; i<MD_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};

class LargeWorkMsg {
 public:
  int workerID;
  int data[LG_MSG_SZ];
  LargeWorkMsg& operator=(const LargeWorkMsg& obj) {
    eventMsg::operator=(obj);
    workerID = obj.workerID;
    for (int i=0; i<LG_MSG_SZ; i++) data[i] = obj.data[i];
    return *this;
  }
};


class worker {
 public:
  int workerID, numWorkers;
  int numObjs, numMsgs, msgSize, distribution, connectivity, locality;
  int offsetPattern, sendPattern;
  int numSends[5], offsets[5], neighbors[100];
  int numNbrs, sendIdx, nbrIdx, offsetIdx, msgIdx;
  int data[100];
  worker();
  void set(WorkerData *m); 
  worker& operator=(const worker& obj);
  void pup(PUP::er &p) { 
    p(workerID); p(numWorkers);
    p(numObjs); p(numMsgs); p(msgSize); p(distribution); p(connectivity);
    p(locality); p(offsetPattern); p(sendPattern);
    p(numSends, 5); p(offsets, 5); p(neighbors, 100); p(data, 100);
    p(numNbrs); p(sendIdx); p(nbrIdx); p(offsetIdx); p(msgIdx);
  }
};

class team {
 public:
  int numTeams, numWorkers, numObjs, workersRecvd;
  worker *myWorkers;
  team() { }
  team(TeamData *m);
  ~team() { delete[] myWorkers; }
  team& operator=(const team& obj) {
    rep::operator=(obj);
    numTeams = obj.numTeams;
    numWorkers = obj.numWorkers;
    numObjs = obj.numObjs;
    workersRecvd = obj.workersRecvd;
    myWorkers = new worker[numWorkers];
    for (int i=0; i<numWorkers; i++) myWorkers[i] = obj.myWorkers[i];
    return *this;
  }
  void pup(PUP::er &p) { 
    chpt<state_team>::pup(p); 
    p(numTeams);  p(numWorkers);  p(numObjs);  p(workersRecvd);
    if (p.isUnpacking()) myWorkers = new worker[numWorkers];
    p(myWorkers, numWorkers);
  }
  void addWorker(WorkerData *wd);
  void start(eventMsg *em);
  void start_anti(eventMsg *em);
  void start_commit(eventMsg *em);
  void workSmall(SmallWorkMsg *sm);
  void workSmall_anti(SmallWorkMsg *sm);
  void workSmall_commit(SmallWorkMsg *sm);
  void workMedium(MediumWorkMsg *mm);
  void workMedium_anti(MediumWorkMsg *mm);
  void workMedium_commit(MediumWorkMsg *mm);
  void workLarge(LargeWorkMsg *lm);
  void workLarge_anti(LargeWorkMsg *lm);
  void workLarge_commit(LargeWorkMsg *lm);
  void doWork(int k);
};
