/***************************************************************************
 *
 *  Benchmark to measure performnce of CmiAlloc/CmiFree and traversals
 *  of associated memory
 *
 *  Two types of benchmarking has been done-
 *
 *  1. A flurry of operations of same type and on same size
 *
 *  2. A random but commuatatively organized  mix of operations on a range
 *     of data size.
 *
 *  Author- Nikhil Jain
 *
 ***************************************************************************/

#include <converse.h>
#include "commbench.h"

#define CLUBMALLOC 1000
#define RANDMALLOC 1000
#define RANDOPS 1000
#define POWER 15
#define MAXSIZE 16384
typedef double myType;
myType value = 10;
int dist[] = {1,5,9,10};

CpvStaticDeclare(int, memoryIdx);

double memoryTest(){
  double starttime, endtime;  
  double extraOverhead;
  myType **ptrs = NULL;
  int i,j,k,size;
  myType sums[200];
  double times[POWER][4];
  int sizes[RANDMALLOC];
  int ops[RANDOPS][3];

  ptrs = (myType **)malloc(CLUBMALLOC*sizeof(myType *));
  for(i=0; i<CLUBMALLOC; i++) ptrs[i] = 0;
  starttime = CmiWallTimer();
  for(i=0; i<CLUBMALLOC; i++) ptrs[i] = (myType *)0xabcd;
  endtime = CmiWallTimer();
  extraOverhead = endtime - starttime;
  
  size = 1;
  for(i = 0; i < POWER; i++,size*=2) {

    for(j=0; j<CLUBMALLOC; j++) ptrs[j] = 0;
    starttime = CmiWallTimer();
    for(j=0; j<CLUBMALLOC; j++) {
      ptrs[j] = (myType *)CmiAlloc(size*sizeof(myType));
    }
    endtime = CmiWallTimer();
    times[i][0] = endtime - starttime - extraOverhead;
    
    for(j=0; j<CLUBMALLOC; j++) {
      for(k = 0; k < size; k++) {
        ptrs[j][k] = value;
      }
    }

    starttime = CmiWallTimer();
    for(j=0; j<CLUBMALLOC; j++) {
      for(k = 0; k < size; k++) {
        ptrs[j][k] = value;
      }
    }
    endtime = CmiWallTimer();
    times[i][1] = endtime - starttime - extraOverhead;

    for(j = 0; j < 200; j++) {
      sums[j] = 0;
    }

    starttime = CmiWallTimer();
    for(j=0; j<CLUBMALLOC; j++) {
      for(k = 0; k < size; k++) {
        sums[k%200] += ptrs[j][k];
      }
    }
    endtime = CmiWallTimer();
    times[i][2] = endtime - starttime - extraOverhead;

    starttime = CmiWallTimer();
    for(j=0; j<CLUBMALLOC; j++) CmiFree(ptrs[j]);
    endtime = CmiWallTimer();
    times[i][3] = endtime - starttime - extraOverhead;
  }

  if(CmiMyPe()==0){
    CmiPrintf("Performance number of clubbed malloc-traversal-free\n");
    CmiPrintf("Size\tIterations\tMalloc\tWrite\tRead\tFree\n");
    size = 1;
    for(i = 0; i < POWER; i++,size*=2) {
      CmiPrintf("%d\t%d\t%E\t%E\t%E\t%E\n",size,CLUBMALLOC,times[i][0],
               times[i][1],times[i][2],times[i][3]);
    }
  }

  free(ptrs);
  ptrs = (myType **)malloc(RANDMALLOC*sizeof(myType *));

  srand(7187);
  for(i=0; i<RANDMALLOC; i++) {
    sizes[i] = rand() % MAXSIZE;
  }

  for(i=0; i<RANDMALLOC; i++) {
      ptrs[i] = (myType *)CmiAlloc(sizes[i]*sizeof(myType));
  }

  for(i=0; i<RANDOPS; i++) {
    ops[i][0] = rand()%RANDMALLOC;
    ops[i][1] = rand()%dist[3];
    if(ops[i][1] < dist[0]) {
      ops[i][1] = 0;
      ops[i][2] =rand()%MAXSIZE;
    } else if(ops[i][1] < dist[1]) {
      ops[i][1] = 1;
      ops[i][2] = sizes[ops[i][0]];
    } if(ops[i][1] < dist[2]) {
      ops[i][1] = 2;
      ops[i][2] = sizes[ops[i][0]];
    } else {
      ops[i][1] = 3;
    }
  }

  starttime = CmiWallTimer();
  for(i=0; i<RANDOPS; i++) {
    switch(ops[i][1]) {
      case 0:
        if(ptrs[ops[i][0]] != NULL)
          CmiFree(ptrs[ops[i][0]]);
        ptrs[ops[i][0]] = (myType *)CmiAlloc(ops[i][2]*sizeof(myType));
        break;
      case 1:
        if(ptrs[ops[i][0]] == NULL)
          ptrs[ops[i][0]] = (myType *)CmiAlloc(ops[i][2]*sizeof(myType));
        for(j = 0; j < ops[i][2]; j++) {
          ptrs[ops[i][0]][j] = value;
        }
        break;
     case 2:
        if(ptrs[ops[i][0]] == NULL)
          ptrs[ops[i][0]] = (myType *)CmiAlloc(ops[i][2]*sizeof(myType));
        for(j = 0; j < ops[i][2]; j++) {
          sums[k%200] += ptrs[ops[i][0]][j];
        }
        break;
    case 3:
      if(ptrs[ops[i][0]] != NULL){
        CmiFree(ptrs[ops[i][0]]);
        ptrs[ops[i][0]] = NULL;
      }
    }
  }
  endtime = CmiWallTimer();

  if(CmiMyPe()==0){
    CmiPrintf("Time taken by random malloc-traversal-free benchmark with following commutative distribution: malloc %d, write %d, read %d, free %d, max malloc size %d, length of table %d, number of ops %d is %E.\n",dist[0],dist[1],dist[2],dist[3],MAXSIZE,RANDMALLOC,RANDOPS, (endtime-starttime));
  }

  for(i=0; i<RANDMALLOC; i++) {
    if(ptrs[i] != NULL)
      CmiFree(ptrs[i]);
  }
  free(ptrs);
}

static void memoryHandler(EmptyMsg *msg){
	/* Make sure the memory contention on a node happens roughly at the same time */
	CmiNodeBarrier();
	memoryTest();	
	CmiNodeBarrier();
	
	if(CmiMyPe()==0){
    CmiSetHandler(msg, CpvAccess(ack_handler));
    CmiSyncSend(0, sizeof(EmptyMsg), msg);
	}
	else {
	  CmiFree(msg);
	}
}

void memoryAccess_init(void)
{
  EmptyMsg msg;

  CmiPrintf("Single core malloc/free/traversal performance numbers\n");
  memoryTest();
             
  CmiPrintf("Multi core malloc/free/traversal performance numbers\n");
  CmiSetHandler(&msg, CpvAccess(memoryIdx));
  CmiSyncBroadcastAll(sizeof(EmptyMsg), &msg);
}

void memoryAccess_moduleinit(void)
{
  CpvInitialize(int, memoryIdx);
  CpvAccess(memoryIdx) = CmiRegisterHandler((CmiHandler)memoryHandler);
}

