#include "charm++.h"
#include "channel.h"
#include "single.decl.h"

class JobConsumer : public CBase_JobConsumer {
  CProxy_CkChannel<int> channel;
public:
  JobConsumer(CkMigrateMessage *m) {}
  JobConsumer(CProxy_CkChannel<int> channel_) : channel(channel_) {}
  void run();
};

class JobProducer : public CBase_JobProducer {
  CProxy_CkChannel<int> channel;
public:
  JobProducer(CkMigrateMessage *m) {}
  JobProducer(CProxy_CkChannel<int> channel_) : channel(channel_) {}
  void run();
};

class Main : public CBase_Main {
public:
  Main(CkArgMsg *);
};
