#include "stress.h"
#include <cstdlib>
#include <ctime>
#include <numeric>

Main::Main(CkArgMsg *m) {
  mainProxy = thisProxy;
  channel = CProxy_CkMultiChannel<NUM_ITERS, int>::ckNew();
  // Create all of the consumer arrays
  std::array<CProxy_JobConsumer, numArrays> consumers;
  for (int i = 0; i < numArrays; i++) {
    consumers[i] = CProxy_JobConsumer::ckNew(channel, elementsPerArray);
  }
  // Create the producer
  CProxy_JobProducer producer = CProxy_JobProducer::ckNew(channel);
  // Run the consumers and producer
  for (auto consumer : consumers) {
    consumer.run();
  }
  producer.run();
  // Exit on quiescence
  CkStartQD(CkCallback(CkCallback::ckExit));
}

void JobConsumer::run() {
  int channels[NUM_ITERS];
  std::iota(channels, channels + NUM_ITERS, 0);
  std::pair<int, int> result = channel.receiveAny(channels, NUM_ITERS);
  int iter = result.first + 1;
  if (iter % itersPerHeartbeat == 0) {
    channel.send(result.first, result.second);
  }
  if (iter < NUM_ITERS) {
    thisProxy[thisIndex].run();
  }
}

void JobProducer::run() {
  const int numElements = numArrays * elementsPerArray;
  for (int i = 0; i < NUM_ITERS; i++) {
    CkPrintf("[%d/%d] Putting %d values into the channel.\n", i + 1, NUM_ITERS, numElements);
    for (int j = 0; j < numElements; j++) {
      channel.send(i, j);
    }
    if ((i + 1) % itersPerHeartbeat == 0) {
      CkPrintf("[%d/%d] Waiting for %d values from the channel.\n", i + 1, NUM_ITERS, numElements);
      channel.waitN(i, numElements);
    }
  }
}

#include "stress.def.h"

