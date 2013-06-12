#include "charm++.h"

extern "C" void createCustomPartitions(int numparts, int *partitionSize, int *nodeMap) {
  CkAbort("Dummy partitioner invoked when custom partitioner should be invoked. Aborting \n");
}

