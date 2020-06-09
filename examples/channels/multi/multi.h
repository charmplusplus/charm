#include "charm++.h"
#include "channel.h"
#include "multi.decl.h"

CProxy_Main mainProxy;
int numWorkers;

class JobConsumer : public CBase_JobConsumer {
  CProxy_CkMultiChannel<2, int> channel;
public:
  JobConsumer(CkMigrateMessage *m) {}
  JobConsumer(CProxy_CkMultiChannel<2, int> channel_) : channel(channel_) {}
  void run();
};

class JobProducer : public CBase_JobProducer {
  CProxy_CkMultiChannel<2, int> channel;
public:
  JobProducer(CkMigrateMessage *m) {}
  JobProducer(CProxy_CkMultiChannel<2, int> channel_) : channel(channel_) {}
  void run();
};

class Main : public CBase_Main {
  CProxy_CkMultiChannel<2, int> channel;
public:
  Main(CkArgMsg *);
  void stageI();
  void stageII(std::pair<int, int> response);
};
