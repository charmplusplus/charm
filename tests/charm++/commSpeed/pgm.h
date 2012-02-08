#include "Pgm.decl.h"

extern CProxy_main mp; 

class main : public CBase_main {
  int numMsgs, msgSize, numIter;
  double localAvg, localMax, localMin, remoteAvg, remoteMax, remoteMin, startTime, initTime;
public:
  main(CkArgMsg *m);
  main(CkMigrateMessage *) {};
  void finish(double avgLocal, double avgRemote);
};
