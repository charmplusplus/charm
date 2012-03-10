#include <stdio.h>
#include "converse.h"

#define HIGH_PRIO 0
#define LOW_PRIO 1
#define NUM_YIELD 10

int endCounter = 0;

//determine completion based on threads calling it
void threadDone() {
  endCounter++;
  if (endCounter == 2) CsdExitScheduler();
}

//worker function for worker1, yields with a low priority
void worker1Work(void* msg) {
  printf("start worker1\n");
  CthYield();
  printf("worker1 resumed first time\n");
  unsigned int prio  = LOW_PRIO;
  for(int i = 0; i < NUM_YIELD; i++) {
    CthYieldPrio(CQS_QUEUEING_IFIFO,0,&prio);
    printf("worker1 resumed %dth time\n",i);
  }
  threadDone();
}

//worker function for worker2, yields with a high priority
void worker2Work(void* msg) {
  printf("start worker2\n");
  CthYield();
  printf("worker2 resumed first time\n");
  unsigned int prio  = HIGH_PRIO;
  for(int i = 0; i < NUM_YIELD; i++) {
    CthYieldPrio(CQS_QUEUEING_IFIFO,0,&prio);
    printf("worker2 resumed %dth time\n",i);
  }
  threadDone();
}

//create two worker threads and push them on scheduler Q
void initThreads(int argc, char* argv[]) {
  printf("called initThreads\n");
  CthThread worker1 = CthCreateMigratable((CthVoidFn)worker1Work, 0, 160000);
  CthThread worker2 = CthCreateMigratable((CthVoidFn)worker2Work, 0, 160000);
  CthAwaken(worker1); CthAwaken(worker2);
}

int main(int argc, char* argv[]) {
  ConverseInit(argc, argv, initThreads, 0, 0);
}
