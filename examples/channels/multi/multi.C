#include "multi.h"
#include <cstdlib>
#include <ctime>
#include <numeric>

Main::Main(CkArgMsg *m) {
  mainProxy = thisProxy;
  numWorkers = CkNumPes();
  CkPrintf("[MAIN] Creating channel, producer, and consumer.\n");
  channel = CProxy_CkMultiChannel<2, int>::ckNew();
  CProxy_JobConsumer consumer = CProxy_JobConsumer::ckNew(channel);
  CProxy_JobProducer producer = CProxy_JobProducer::ckNew(channel);
  // Run the consumers and producer
  consumer.run();
  producer.run();
}

void Main::stageI() {
  CkPrintf("[MAIN] Setting a callback for data from any channel.\n");
  CkCallback cb(CkIndex_Main::stageII(std::make_pair(0, 0)), mainProxy);
  int channels[2];
  std::iota(channels, channels + 2, 0);
  channel.receiveFromAny(channels, 2, cb);
  CkPrintf("[MAIN] Sending data to a random channel...\n");
  std::srand(std::time(nullptr));
  int ch = std::rand() % 2;
  channel.send(ch, ch);
}

void Main::stageII(std::pair<int, int> response) {
  CkPrintf("[MAIN] Received data from channel %d...\n", response.first);
  CkExit();
}

void JobConsumer::run() {
  CkPrintf("[CONSUMER%d] Received job %d from channel 0.\n", thisIndex, channel.receive(0));
  channel.send(1, thisIndex);
}

void JobProducer::run() {
  CkPrintf("[PRODUCER] Putting %d jobs into channel 0.\n", numWorkers);
  for (int i = 0; i < numWorkers; i++) {
    channel.send(0, 2 * (i + 1));
  }
  CkPrintf("[PRODUCER] Awaiting completion notices from consumers (on channel 1).\n");
  channel.waitN(1, numWorkers);
  mainProxy.stageI();
}

#include "multi.def.h"

