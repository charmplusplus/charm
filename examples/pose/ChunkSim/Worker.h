// Distribution types
#define RANDOM 0
#define IMBALANCED 1
#define UNIFORM 2

// Connectivity types
#define SPARSE 0
#define HEAVY 1
#define FULL 2

#define WORKER_SZ 1000

class TeamData {
 public:
  int teamID, numTeams, numWorkers;
  TeamData& operator=(const TeamData& obj) {
    eventMsg::operator=(obj);
    teamID = obj.teamID;
    numTeams = obj.numTeams;
    numWorkers = obj.numWorkers;
    return *this;
  }
};

class WorkMsg {
 public:
  int workerID;
  int data[10];
  WorkMsg& operator=(const WorkMsg& obj) {
    eventMsg::operator=(obj);
    workerID = obj.workerID;
    for (int i=0; i<10; i++) data[i] = obj.data[i];
    return *this;
  }
};


class worker {
 public:
  int workerID;
  int data[WORKER_SZ];
  worker();
  void set(int wid); 
  worker& operator=(const worker& obj);
  void pup(PUP::er &p) { p(workerID); p(data, WORKER_SZ); }
};

class team {
 public:
  int teamID, numTeams, numWorkers;
  worker *myWorkers;
  team() { }
  team(TeamData *m);
  ~team() { delete[] myWorkers; }
  team& operator=(const team& obj) {
    rep::operator=(obj);
    teamID = obj.teamID;
    numTeams = obj.numTeams;
    numWorkers = obj.numWorkers;
    myWorkers = new worker[numWorkers/numTeams];
    for (int i=0; i<numWorkers/numTeams; i++) myWorkers[i] = obj.myWorkers[i];
    return *this;
  }
  void pup(PUP::er &p) { 
    chpt<state_team>::pup(p); 
    p(teamID); p(numTeams); p(numWorkers);
    if (p.isUnpacking()) myWorkers = new worker[numWorkers/numTeams];
    p(myWorkers, numWorkers/numTeams);
  }
  void work(WorkMsg *sm);
  void work_anti(WorkMsg *sm);
  void work_commit(WorkMsg *sm);
  void doWork(int k);
};
