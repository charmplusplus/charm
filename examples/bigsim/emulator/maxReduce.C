#include <stdlib.h>
#include "blue.h"

#define A_SIZE 4

#define X_DIM 3
#define Y_DIM 3
#define Z_DIM 3

int REDUCE_HANDLER_ID;
int COMPUTATION_ID;

extern "C" void reduceHandler(char *);
extern "C" void computeMax(char *);

class ReductionMsg {
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  int max;
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class ComputeMsg {
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  int dummy;
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

void BgEmulatorInit(int argc, char **argv)
{
  BgSetSize(X_DIM, Y_DIM, Z_DIM);
}

void BgNodeStart(int argc, char **argv) {
  int x, y, z;
  BgGetMyXYZ(&x, &y, &z);

  REDUCE_HANDLER_ID = BgRegisterHandler(reduceHandler);
  COMPUTATION_ID = BgRegisterHandler(computeMax);

  CmiPrintf("Initializing node %d, %d, %d\n", x,y,z);

  ComputeMsg *msg = new ComputeMsg;
  BgSendLocalPacket(ANYTHREAD, COMPUTATION_ID, LARGE_WORK, sizeof(ComputeMsg), (char *)msg);
}

void reduceHandler(char *info) {
  // assumption: THey are initialized to zero?
  static int max[X_DIM][Y_DIM][Z_DIM];
  static int num_msg[X_DIM][Y_DIM][Z_DIM];

  int i,j,k;
  int external_max;

  BgGetMyXYZ(&i,&j,&k);
  external_max = ((ReductionMsg *)info)->max;
  delete (ReductionMsg *)info;
  num_msg[i][j][k]++;

  if ((i == 0) && (j == 0) && (k == 0)) {
    // master node expects 4 messages:
    // 1 from itself;
    // 1 from the i dimension;
    // 1 from the j dimension; and
    // 1 from the k dimension
    if (num_msg[i][j][k] < 4) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Can report max data after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      CmiPrintf("The maximal value is %d \n", max[i][j][k]);
      BgShutdown();
      return;
    }
  } else if ((i == 0) && (j == 0) && (k != Z_DIM - 1)) {
    // nodes along the k-axis other than the last one expects 4 messages:
    // 1 from itself;
    // 1 from the i dimension;
    // 1 from the j dimension; and
    // 1 from the k dimension
    if (num_msg[i][j][k] < 4) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Forwards max data to node i,j,k-1 after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      ReductionMsg *msg = new ReductionMsg;
      msg->max = max[i][j][k];
      BgSendPacket(i,j,k-1,ANYTHREAD,REDUCE_HANDLER_ID,LARGE_WORK, sizeof(ReductionMsg), (char *)msg);
    }
  } else if ((i == 0) && (j == 0) && (k == Z_DIM - 1)) {
    // the last node along the k-axis expects 3 messages:
    // 1 from itself;
    // 1 from the i dimension; and
    // 1 from the j dimension
    if (num_msg[i][j][k] < 3) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Forwards max data to node i,j,k-1 after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      ReductionMsg *msg = new ReductionMsg;
      msg->max = max[i][j][k];
      BgSendPacket(i,j,k-1,ANYTHREAD,REDUCE_HANDLER_ID,LARGE_WORK, sizeof(ReductionMsg), (char *)msg);
    }
  } else if ((i == 0) && (j != Y_DIM - 1)) {
    // for nodes along the j-k plane except for the last and first row of j,
    // we expect 3 messages:
    // 1 from itself;
    // 1 from the i dimension; and
    // 1 from the j dimension
    if (num_msg[i][j][k] < 3) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Forwards max data to node i,j-1,k after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      ReductionMsg *msg = new ReductionMsg;
      msg->max = max[i][j][k];
      BgSendPacket(i,j-1,k,ANYTHREAD,REDUCE_HANDLER_ID,LARGE_WORK, sizeof(ReductionMsg), (char *)msg);
    }
  } else if ((i == 0) && (j == Y_DIM - 1)) {
    // for nodes along the last row of j on the j-k plane,
    // we expect 2 messages:
    // 1 from itself;
    // 1 from the i dimension;
    if (num_msg[i][j][k] < 2) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Forwards max data to node i,j-1,k after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      ReductionMsg *msg = new ReductionMsg;
      msg->max = max[i][j][k];
      BgSendPacket(i,j-1,k,ANYTHREAD,REDUCE_HANDLER_ID,LARGE_WORK, sizeof(ReductionMsg), (char *)msg);
    }
  } else if (i != X_DIM - 1) {
    // for nodes anywhere the last row of i,
    // we expect 2 messages:
    // 1 from itself;
    // 1 from the i dimension;
    if (num_msg[i][j][k] < 2) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Forwards max data to node i-1,j,k after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      ReductionMsg *msg = new ReductionMsg;
      msg->max = max[i][j][k];
      BgSendPacket(i-1,j,k,ANYTHREAD,REDUCE_HANDLER_ID,LARGE_WORK, sizeof(ReductionMsg), (char *)msg);
    }
  } else if (i == X_DIM - 1) {
    // last row of i, we expect 1 message:
    // 1 from itself;
    if (num_msg[i][j][k] < 1) {
      // not ready yet, so just find the max
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
    } else {
      // done. Forwards max data to node i-1,j,k after making last comparison
      if (max[i][j][k] < external_max) {
	max[i][j][k] = external_max;
      }
      ReductionMsg *msg = new ReductionMsg;
      msg->max = max[i][j][k];
      BgSendPacket(i-1,j,k,-1,REDUCE_HANDLER_ID,LARGE_WORK, sizeof(ReductionMsg), (char *)msg);
    }
  }
}

void computeMax(char *info) {
  int A[A_SIZE][A_SIZE];
  int i, j;
  int max = 0;

  int x,y,z; // test variables
  BgGetMyXYZ(&x,&y,&z);

  // Initialize
  for (i=0;i<A_SIZE;i++) {
    for (j=0;j<A_SIZE;j++) {
      A[i][j] = i*j;
    }
  }

  CmiPrintf("Finished Initializing %d %d %d!\n",  x , y , z);

  // Find Max
  for (i=0;i<A_SIZE;i++) {
    for (j=0;j<A_SIZE;j++) {
      if (max < A[i][j]) {
	max = A[i][j];
      }
    }
  }

  CmiPrintf ("Finished Finding Max\n");

  // prepare to reduce
  ReductionMsg *msg = new ReductionMsg;
  msg->max = max;
  BgSendLocalPacket(ANYTHREAD, REDUCE_HANDLER_ID, LARGE_WORK, sizeof(ReductionMsg), (char *)msg);

  CmiPrintf("Sent reduce message to myself with max value %d\n", max);
}


