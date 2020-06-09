#include "single.h"

Main::Main(CkArgMsg *m) {
  CkPrintf("[MAIN] Creating channel, producer, and consumer.\n");
  CProxy_CkChannel<int> channel = CProxy_CkChannel<int>::ckNew();
  CProxy_JobConsumer consumer = CProxy_JobConsumer::ckNew(channel);
  CProxy_JobProducer producer = CProxy_JobProducer::ckNew(channel);
  // Run the consumers and producer
  consumer.run();
  producer.run();
  // Exit on quiescence
  CkStartQD(CkCallback(CkCallback::ckExit));
}

void JobConsumer::run() {
  CkPrintf("[CONSUMER%d] Received job %d from the channel.\n", thisIndex, channel.receive());
}

void JobProducer::run() {
  int numJobs = CkNumPes();
  CkPrintf("[PRODUCER] Putting %d jobs into the channel.\n", numJobs);
  for (int i = 0; i < numJobs; i++) {
    channel.send(2 * (i + 1));
  }
}

#include "single.def.h"

