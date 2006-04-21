// X10_lib.h

#include <pup.h>
#include <converse.h>
#include <charm++.h>



extern void mainThread(void);


typedef void* FinishHandle;

void *beginFinish();
void endFinish(void *FinishFutureList);
void asyncCall(void *FinishFutureList, int place, int whichFunction, void *packedParams);
