#include "charm++.h"
#include "Worker.decl.h"

extern CProxy_worker wArray; 

class WorkerData : public CMessage_WorkerData {
 public:
  int numMsgs, msgSize;
};

class WorkMsg : public CMessage_WorkMsg {
 public:
  int *data;
};

class worker : public ArrayElement1D {
  int numMsgs, msgSize, sent;
  double lsum, lmax, lmin, rsum, rmax, rmin;
 public:
  worker(WorkerData *m);
  worker(CkMigrateMessage *m) { }  
  // Event methods
  void doStuff();
  void work(WorkMsg *m);
};

