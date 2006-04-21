// X10_lib.h

#include <pup.h>
#include <converse.h>
#include <charm++.h>



extern void mainThread(void);
extern void asnycHandler(int whichStatement);
extern void futureHandler(int whichStatement);

typedef void* FinishHandle;
typedef CkFutureID* FutureHandle;

FinishHandle beginFinish();
void endFinish(void *FinishFutureList);
void asyncCall(void *FinishFutureList, int place, int whichFunction, void *packedParams);

FutureHandle futureCall(int place, int whichFunction, void *packedParams);
void *futureForce(FutureHandle);
