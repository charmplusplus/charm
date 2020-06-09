#define NUM_ITERS 16
#include "charm++.h"
#include "channel.h"
#include "stress.decl.h"

const int numArrays = 16;
const int elementsPerArray = 16;
const int itersPerHeartbeat = 4;
CProxy_Main mainProxy;

class JobConsumer : public CBase_JobConsumer {
  CProxy_CkMultiChannel<NUM_ITERS, int> channel;
public:
  JobConsumer(CkMigrateMessage *m) {}
  JobConsumer(CProxy_CkMultiChannel<NUM_ITERS, int> channel_) : channel(channel_) {}
  void run();
};

class JobProducer : public CBase_JobProducer {
  CProxy_CkMultiChannel<NUM_ITERS, int> channel;
public:
  JobProducer(CkMigrateMessage *m) {}
  JobProducer(CProxy_CkMultiChannel<NUM_ITERS, int> channel_) : channel(channel_) {}
  void run();
};

class Main : public CBase_Main {
  CProxy_CkMultiChannel<NUM_ITERS, int> channel;
public:
  Main(CkArgMsg *);
  void stageI();
  void stageII(std::pair<int, int> response);
};
