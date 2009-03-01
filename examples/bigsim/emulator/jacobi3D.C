//
// 3D Jacobi code for the Blue Gene emulator
// - Version 0.01 (27 Dec 2000) Chee Wai Lee

// partly rewritten by Gengbin Zheng on 2/20/2001 porting to my Converse based
// BlueGene emulator

#include <math.h>
#include <stdlib.h>
#include "blue.h"

#define MAX_ARRAY_SIZE 36 // the suns don't like a larger size
#define A_SIZE_DEFAULT 16
#define NUM_DBLMSG_COUNT 15

int reduceID;
int computeID;
int ghostID;
int exchangeID;
int outputID;

// source ghost region tags
#define LEFT            1
#define RIGHT           2
#define BELOW           3
#define ABOVE           4
#define FRONT           5
#define BACK            6

extern "C" void reduce(char *);
extern "C" void compute(char *);
extern "C" void ghostrecv(char *);
extern "C" void ghostexchange(char *);
extern "C" void outputData(char *);

void initArray(double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
	       int, int, int);
void copyArray(double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
	       double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
	       int, int, int,
	       int, int, int);
void copyXArray(double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		double [1][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		int, int, int,
		int, int, int);
void copyYArray(double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		double [MAX_ARRAY_SIZE][1][MAX_ARRAY_SIZE],
		int, int, int,
		int, int, int);
void copyZArray(double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][1],
		int, int, int,
		int, int, int);
void printArray(double [MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		int, int, int);

class reduceMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  bool done;
  // IMPORTANT! kernel messages are allocated using CmiAlloc
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class computeMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class ghostMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  // respect the 128-byte limit of a bluegene message
  double data[NUM_DBLMSG_COUNT]; 
  int datacount;
  int source;
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class exchangeMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

class outputMsg
{
public:
  char core[CmiBlueGeneMsgHeaderSizeBytes];
  void *operator new(size_t s) { return CmiAlloc(s); }
  void operator delete(void* ptr) { CmiFree(ptr); }
};

typedef struct {
  double gdata[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
  double maindata[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
  double tempdata[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
  double ghost_x1[1][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
  double ghost_x2[1][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
  double ghost_y1[MAX_ARRAY_SIZE][1][MAX_ARRAY_SIZE];
  double ghost_y2[MAX_ARRAY_SIZE][1][MAX_ARRAY_SIZE];
  double ghost_z1[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][1];
  double ghost_z2[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][1];
  int my_x_size;
  int my_y_size;
  int my_z_size;
  int reduction_count;   // maintained only by PE[0][0][0] 
  int ghost_x1_elements_total;
  int ghost_x2_elements_total;
  int ghost_y1_elements_total;
  int ghost_y2_elements_total;
  int ghost_z1_elements_total;
  int ghost_z2_elements_total;
  int ghost_x1_elements_current;
  int ghost_x2_elements_current;
  int ghost_y1_elements_current;
  int ghost_y2_elements_current;
  int ghost_z1_elements_current;
  int ghost_z2_elements_current;
  bool done;             // maintained only by PE[0][0][0]
  int iteration_count;
} nodeData;

int g_x_blocksize;
int g_y_blocksize;
int g_z_blocksize;
double g_epsilon;
int g_iteration_count = 0;
int g_dataXsize = A_SIZE_DEFAULT;
int g_dataYsize = A_SIZE_DEFAULT;
int g_dataZsize = A_SIZE_DEFAULT;

void BgEmulatorInit(int argc, char **argv) 
{
  if (argc < 7) {
    CmiPrintf("Usage: jacobi3D <BG_x> <BG_y> <BG_z> <numCommTh> <numWorkTh>");
    CmiAbort("<epsilon> [<x>] [<y>] [<z>]\n");
  } 

  BgSetSize(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  BgSetNumCommThread(atoi(argv[4]));
  BgSetNumWorkThread(atoi(argv[5]));

  if (CmiMyRank() == 0)
  g_epsilon = atof(argv[6]);

    switch (argc) {
      case 8: {
	g_dataXsize = atoi(argv[7]);
	break;
      }
      case 9: {
	g_dataXsize = atoi(argv[7]);
	g_dataYsize = atoi(argv[8]);
	break;
      }
      case 10: {
	g_dataXsize = atoi(argv[7]);
	g_dataYsize = atoi(argv[8]);
	g_dataZsize = atoi(argv[9]);
	break;
      }
    }

    int numBgX, numBgY, numBgZ;
    BgGetSize(&numBgX, &numBgY, &numBgZ);
    g_x_blocksize = g_dataXsize/numBgX;
    g_y_blocksize = g_dataYsize/numBgY;
    g_z_blocksize = g_dataZsize/numBgZ;

    if (CmiMyPe() == 0) {
    CmiPrintf("Bluegene size: %d %d %d\n",
	     numBgX, numBgY, numBgZ);
    CmiPrintf("Parameters: %d %d %d, Epsilon: %e\n",
	     g_dataXsize, g_dataYsize, g_dataZsize, g_epsilon);
    }
}

void BgNodeStart(int argc, char **argv) 
{
  reduceID = BgRegisterHandler(reduce);
  computeID = BgRegisterHandler(compute);
  ghostID = BgRegisterHandler(ghostrecv);
  exchangeID = BgRegisterHandler(ghostexchange);
  outputID = BgRegisterHandler(outputData);

  /*
  CmiPrintf("Starting BgInit at Node %d %d %d\n", 
	   bgNode->thisIndex.x,
	   bgNode->thisIndex.y,
	   bgNode->thisIndex.z);
  */
  int x_start, y_start, z_start;
  int x_end, y_end, z_end;

  int x,y,z;
  BgGetMyXYZ(&x, &y, &z);
  // determine partition indices given BG node address by simple uniform
  // partitioning.
  x_start = x * g_x_blocksize;
  y_start = y * g_y_blocksize;
  z_start = z * g_z_blocksize;

  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);
  if (x == numBgX - 1) {
    x_end = g_dataXsize - 1;
  } else {
    x_end = x_start + g_x_blocksize - 1;
  }
  if (y == numBgY - 1) {
    y_end = g_dataYsize - 1;
  } else {
    y_end = y_start + g_y_blocksize - 1;
  }
  if (z == numBgZ - 1) {
    z_end = g_dataZsize - 1;
  } else {
    z_end = z_start + g_z_blocksize - 1;
  }

  //create node private data
  nodeData *nd = new nodeData;

  // populate global data with default values
  initArray(nd->gdata, g_dataXsize, g_dataYsize, g_dataZsize);

  nd->my_x_size = x_end - x_start + 1;
  nd->my_y_size = y_end - y_start + 1;
  nd->my_z_size = z_end - z_start + 1;
  nd->reduction_count = numBgX * numBgY * numBgZ;
  if (x != 0) {
    nd->ghost_x1_elements_total = (nd->my_y_size)*(nd->my_z_size);
  } else {
    nd->ghost_x1_elements_total = 0;
  }
  if (x != numBgX - 1) {
    nd->ghost_x2_elements_total = (nd->my_y_size)*(nd->my_z_size);
  } else {
    nd->ghost_x2_elements_total = 0;
  }
  if (y != 0) {
    nd->ghost_y1_elements_total = (nd->my_x_size)*(nd->my_z_size);
  } else {
    nd->ghost_y1_elements_total = 0;
  }
  if (y != numBgY - 1) {
    nd->ghost_y2_elements_total = (nd->my_x_size)*(nd->my_z_size);
  } else {
    nd->ghost_y2_elements_total = 0;
  }
  if (z != 0) {
    nd->ghost_z1_elements_total = (nd->my_x_size)*(nd->my_y_size);
  } else {
    nd->ghost_z1_elements_total = 0;
  }
  if (z != numBgZ - 1) {
    nd->ghost_z2_elements_total = (nd->my_x_size)*(nd->my_y_size);
  } else {
    nd->ghost_z2_elements_total = 0;
  }
  nd->ghost_x1_elements_current = 0; 
  nd->ghost_x2_elements_current = 0; 
  nd->ghost_y1_elements_current = 0; 
  nd->ghost_y2_elements_current = 0; 
  nd->ghost_z1_elements_current = 0; 
  nd->ghost_z2_elements_current = 0; 
  nd->done = true; // assume computation complete
  nd->iteration_count = 0;
  /*
  CmiPrintf("Node (%d,%d,%d) has block size %d %d %d\n", 
	   bgNode->thisIndex.x, bgNode->thisIndex.y, bgNode->thisIndex.z,
	   nd->my_x_size, nd->my_y_size, nd->my_z_size);
  */
  // create and populate main data from global data structure
  copyArray(nd->gdata, nd->maindata, 
	    nd->my_x_size, nd->my_y_size, nd->my_z_size,
	    x_start, y_start, z_start);

  //  printArray(nd->maindata, nd->my_x_size, nd->my_y_size, nd->my_z_size);

  // create and populate the size ghost regions
  // boundary conditions have to be checked
  // "x-left" plane
  if  (x != 0) {
    copyXArray(nd->gdata, nd->ghost_x1,
	       1, nd->my_y_size, nd->my_z_size,
	       x_start - 1, y_start, z_start);
  }
  // "x-right" plane
  if (x != numBgX) {
    copyXArray(nd->gdata, nd->ghost_x2, 
	       1, nd->my_y_size, nd->my_z_size,
	       x_end + 1, y_start, z_start);
  }
  // "y-bottom" plane
  if  (y != 0) {
    copyYArray(nd->gdata, nd->ghost_y1, 
	       nd->my_x_size, 1, nd->my_z_size,
	       x_start, y_start - 1, z_start);
  }
  // "y-top" plane
  if  (y != numBgY) {
    copyYArray(nd->gdata, nd->ghost_y2, 
	       nd->my_x_size, 1, nd->my_z_size,
	       x_start, y_end + 1, z_start);
  }
  // "z-front" plane
  if  (z != 0) {
    copyZArray(nd->gdata, nd->ghost_z1, 
	       nd->my_x_size, nd->my_y_size, 1,
	       x_start, y_start, z_start - 1);
  }
  // "z-back" plane
  if  (z != numBgZ) {
    copyZArray(nd->gdata, nd->ghost_z2, 
	       nd->my_x_size, nd->my_y_size, 1,
	       x_start, y_start, z_end + 1);
  }

  computeMsg *msg = new computeMsg;
  BgSendLocalPacket(ANYTHREAD,computeID, LARGE_WORK, sizeof(computeMsg), (char *)msg);

  BgSetNodeData((char *)nd);
}


void compute(char *info) 
{
  delete (computeMsg *)info;
  bool done = true; // assume done before computation begins
  nodeData *localdata = (nodeData *)BgGetNodeData();

  int x,y,z;
  int x_size, y_size, z_size;
  BgGetMyXYZ(&x,&y,&z);
  x_size = localdata->my_x_size;
  y_size = localdata->my_y_size;
  z_size = localdata->my_z_size;

  // CmiPrintf("Node %d %d %d computing values\n",x,y,z);

  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);

  // one iteration of jacobi computation
  for (int i=0; i < x_size; i++) {
    for (int j=0; j < y_size; j++) {
      for (int k=0; k < z_size; k++) {
	int count = 0;
	double sum = 0.0;

	// decide if node is on x1 boundary of bluegene configuration
	if (x != 0) {
	  // decide if element requires ghost region data
	  if (i == 0) {
	    sum += (localdata->ghost_x1)[0][j][k];
	  } else {
	    sum += (localdata->maindata)[i-1][j][k];
	  }
	  count++;
	} else {
	  // no ghost regions to work on. just ignore.
	  if (i != 0) {
	    sum += (localdata->maindata)[i-1][j][k];
	    count++;
	  }
	}
	if (x != numBgX - 1) {
	  if (i == x_size - 1) {
	    sum += (localdata->ghost_x2)[0][j][k];
	  } else {
	    sum += (localdata->maindata)[i+1][j][k];
	  }
	  count++;
	} else {
	  // no ghost regions to work on. just ignore.
	  if (i != x_size - 1) {
	    sum += (localdata->maindata)[i+1][j][k];
	    count++;
	  }
	}
	if (y != 0) {
	  if (j == 0) {
	    sum += (localdata->ghost_y1)[i][0][k];
	  } else {
	    sum += (localdata->maindata)[i][j-1][k];
	  }
	  count++;
	} else {
	  // no ghost regions to work on. just ignore.
	  if (j != 0) {
	    sum += (localdata->maindata)[i][j-1][k];
	    count++;
	  }
	}
	if (y != numBgY - 1) {
	  if (j == y_size - 1) {
	    sum += (localdata->ghost_y2)[i][0][k];
	  } else {
	    sum += (localdata->maindata)[i][j+1][k];
	  }
	  count++;
	} else {
	  // no ghost regions to work on. just ignore.
	  if (j != y_size - 1) {
	    sum += (localdata->maindata)[i][j+1][k];
	    count++;
	  }
	}
	if (z != 0) {
	  if (k == 0) {
	    sum += (localdata->ghost_z1)[i][j][0];
	  } else {
	    sum += (localdata->maindata)[i][j][k-1];
	  }
	  count++;
	} else {
	  // no ghost regions to work on. just ignore.
	  if (k != 0) {
	    sum += (localdata->maindata)[i][j][k-1];
	    count++;
	  }
	}
	if (z != numBgZ - 1) {
	  if (k == z_size - 1) {
	    sum += (localdata->ghost_z2)[i][j][0];
	  } else {
	    sum += (localdata->maindata)[i][j][k+1];
	  }
	  count++;
	} else {
	  // no ghost regions to work on. just ignore.
	  if (k != z_size - 1) {
	    sum += (localdata->maindata)[i][j][k+1];
	    count++;
	  }
	}

	(localdata->tempdata)[i][j][k] = sum / count;
	/*
	CmiPrintf("New: %f, Old: %f, Diff: %f, Epsilon: %f\n",
		 (localdata->tempdata)[i][j][k],
		 (localdata->maindata)[i][j][k],
		 fabs((localdata->tempdata)[i][j][k] - 
		      (localdata->maindata)[i][j][k]),
		 g_epsilon);
	*/
	if (fabs((localdata->maindata)[i][j][k] - 
		 (localdata->tempdata)[i][j][k])
	    > g_epsilon) {
	  done = false;  // we're not finished yet!
	}
      }
    }
  }  // end of for loop in jacobi iteration
  
  copyArray(localdata->tempdata, localdata->maindata,
	    x_size, y_size, z_size,
	    0, 0, 0);
  localdata->iteration_count++;
  /*
  CmiPrintf("Array computation of Node (%d,%d,%d) at iteration %d\n",
	   x, y, z,
	   localdata->iteration_count);
  printArray(localdata->maindata,x_size,y_size,z_size);
  */
  // perform reduction to confirm if all regions are done. In this 
  // simplified version, everyone sends to processor 0,0,0
  reduceMsg *msg = new reduceMsg();
  msg->done = done;

  if ((x == 0) && (y == 0) && (z == 0)) {
    BgSendLocalPacket(ANYTHREAD,reduceID, SMALL_WORK, sizeof(reduceMsg), (char *)msg);
  } else {
    BgSendPacket(0,0,0,ANYTHREAD,reduceID, SMALL_WORK, sizeof(reduceMsg), (char *)msg);
  }
  
}

void ghostexchange(char *info) {
  delete (exchangeMsg *)info;
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);

  /*
  CmiPrintf("exchange procedure being activated on Node (%d,%d,%d)\n",
	   x, y, z);
  */
  nodeData *localdata = (nodeData *)BgGetNodeData();

  double tempframe[NUM_DBLMSG_COUNT];

  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);
  // No exchange to be done! Immediately go to compute!
  if ((numBgX == 1) &&
      (numBgY == 1) &&
      (numBgZ == 1)) {
    computeMsg *msg = new computeMsg;
    BgSendLocalPacket(ANYTHREAD,computeID, LARGE_WORK, sizeof(computeMsg), (char *)msg);
    return;
  }

  // exchange computed ghost regions
  // initialize message data for x1-planar ghost region
  if (x != 0) {
    int localcount = 0;
    int x_size = 1;
    int y_size = localdata->my_y_size;
    int z_size = localdata->my_z_size;
    for (int i=0; i < x_size; i++) {
      for (int j=0; j < y_size; j++) {
	for (int k=0; k < z_size; k++) {
	  tempframe[localcount] = (localdata->maindata)[i][j][k];
	  localcount++;
	  if ((localcount == NUM_DBLMSG_COUNT) ||
	      ((i == x_size-1) && (j == y_size-1) && (k == z_size-1))) {
	    ghostMsg *x1_msg = new ghostMsg;
	    for (int count=0; count < localcount; count++) {
	      (x1_msg->data)[count] = tempframe[count];
	    }
	    x1_msg->source = RIGHT; // sending message to left
	    x1_msg->datacount = localcount;
	    BgSendPacket(x-1,y,z,ANYTHREAD,ghostID,SMALL_WORK, sizeof(ghostMsg), (char *)x1_msg);
	    localcount = 0;
	  }
	}
      }
    }
  }
  
  // initialize message data for x2-planar ghost region
  if (x != numBgX - 1) {
    int localcount = 0;
    int x_size = 1;
    int y_size = localdata->my_y_size;
    int z_size = localdata->my_z_size;
    for (int i=0; i < x_size; i++) {
      for (int j=0; j < y_size; j++) {
	for (int k=0; k < z_size; k++) {
	  tempframe[localcount] = 
	    (localdata->maindata)[localdata->my_x_size-1][j][k];
	  localcount++;
	  if ((localcount == NUM_DBLMSG_COUNT) ||
	      ((i == x_size-1) && (j == y_size-1) && (k == z_size-1))) {
	    ghostMsg *x2_msg = new ghostMsg;
	    for (int count=0;count < localcount; count++) {
	      (x2_msg->data)[count] = tempframe[count];
	    }
	    x2_msg->source = LEFT; // sending message to right
	    x2_msg->datacount = localcount;
	    BgSendPacket(x+1,y,z,ANYTHREAD,ghostID,SMALL_WORK, sizeof(ghostMsg), (char *)x2_msg);
	    localcount = 0;
	  }
	}
      }
    }
  }

  // initialize message data for y1-planar ghost region
  if (y != 0) {
    int localcount = 0;
    int x_size = localdata->my_x_size;
    int y_size = 1;
    int z_size = localdata->my_z_size;
    for (int i=0; i < x_size; i++) {
      for (int j=0; j < y_size; j++) {
	for (int k=0; k < z_size; k++) {
	  tempframe[localcount] = (localdata->maindata)[i][0][k];
	  localcount++;
	  if ((localcount == NUM_DBLMSG_COUNT) ||
	      ((i == x_size-1) && (j == y_size-1) && (k == z_size-1))) {
	    ghostMsg *y1_msg = new ghostMsg;
	    for (int count=0;count < localcount; count++) {
	      (y1_msg->data)[count] = tempframe[count];
	    }
	    y1_msg->source = ABOVE; // sending message below
	    y1_msg->datacount = localcount;
	    BgSendPacket(x,y-1,z,ANYTHREAD,ghostID,SMALL_WORK, sizeof(ghostMsg), (char *)y1_msg);
	    localcount = 0;
	  }
	}
      }
    }
  }

  // initialize message data for x2-planar ghost region
  if (y != numBgY - 1) {
    int localcount = 0;
    int x_size = localdata->my_x_size;
    int y_size = 1;
    int z_size = localdata->my_z_size;
    for (int i=0; i < x_size; i++) {
      for (int j=0; j < y_size; j++) {
	for (int k=0; k < z_size; k++) {
	  tempframe[localcount] = 
	    (localdata->maindata)[i][localdata->my_y_size-1][k];
	  localcount++;
	  if ((localcount == NUM_DBLMSG_COUNT) ||
	      ((i == x_size-1) && (j == y_size-1) && (k == z_size-1))) {
	    ghostMsg *y2_msg = new ghostMsg;
	    for (int count=0;count < localcount; count++) {
	      (y2_msg->data)[count] = tempframe[count];
	    }
	    y2_msg->source = BELOW; // sending message up
	    y2_msg->datacount = localcount;
	    BgSendPacket(x,y+1,z,ANYTHREAD,ghostID,SMALL_WORK, sizeof(ghostMsg), (char *)y2_msg);
	    localcount = 0;
	  }
	}
      }
    }
  }
  
  // initialize message data for z1-planar ghost region
  if (z != 0) {
    int localcount = 0;
    int x_size = localdata->my_x_size;
    int y_size = localdata->my_y_size;
    int z_size = 1;
    for (int i=0; i < x_size; i++) {
      for (int j=0; j < y_size; j++) {
	for (int k=0; k < z_size; k++) {
	  tempframe[localcount] = (localdata->maindata)[i][j][0];
	  localcount++;
	  if ((localcount == NUM_DBLMSG_COUNT) ||
	      ((i == x_size-1) && (j == y_size-1) && (k == z_size-1))) {
	    ghostMsg *z1_msg = new ghostMsg;
	    for (int count=0; count < localcount; count++) {
	      (z1_msg->data)[count] = tempframe[count];
	    }
	    z1_msg->source = BACK; // sending message to the front
	    z1_msg->datacount = localcount;
	    BgSendPacket(x,y,z-1,ANYTHREAD,ghostID,SMALL_WORK, sizeof(ghostMsg), (char *)z1_msg);
	    localcount = 0;
	  }
	}
      }
    }
  }

  // initialize message data for z2-planar ghost region
  if (z != numBgZ - 1) {
    int localcount = 0;
    int x_size = localdata->my_x_size;
    int y_size = localdata->my_y_size;
    int z_size = 1;
    for (int i=0; i < x_size; i++) {
      for (int j=0; j < y_size; j++) {
	for (int k=0; k < z_size; k++) {
	  tempframe[localcount] = 
	    (localdata->maindata)[i][j][localdata->my_z_size-1];
	  localcount++;
	  if ((localcount == NUM_DBLMSG_COUNT) ||
	      ((i == x_size-1) && (j == y_size-1) && (k == z_size-1))) {
	    ghostMsg *z2_msg = new ghostMsg;
	    for (int count=0; count < localcount; count++) {
	      (z2_msg->data)[count] = tempframe[count];
	    }
	    z2_msg->source = FRONT; // sending message to the back
	    z2_msg->datacount = localcount;
	    BgSendPacket(x,y,z+1,ANYTHREAD,ghostID,SMALL_WORK, sizeof(ghostMsg), (char *)z2_msg);
	    localcount = 0;
	  }
	}
      }
    }
  }
}

void reduce(char *info) 
{
  // only received by processor 0,0,0
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);

  /*
  CmiPrintf("reduce procedure being activated on Node (%d,%d,%d)\n",
	   x, y, z);
  */
  nodeData *localdata = (nodeData *)BgGetNodeData();

  localdata->done &= ((reduceMsg *)info)->done;
  delete (reduceMsg *)info;
  localdata->reduction_count--;

  int numBgX, numBgY, numBgZ;
  BgGetSize(&numBgX, &numBgY, &numBgZ);

  // if that was the last message
  if (localdata->reduction_count == 0) {
    if (!localdata->done) {
      // more work to be done, so ask every node to exchange ghost regions
      for (int i=0; i<numBgX;i++) {
	for (int j=0; j<numBgY;j++) {
	  for (int k=0; k<numBgZ;k++) {
	    exchangeMsg *msg = new exchangeMsg;
	    if ((i == 0) && (j == 0) && (k == 0)) {
	      BgSendLocalPacket(ANYTHREAD,exchangeID, LARGE_WORK, sizeof(exchangeMsg),(char *)msg);
	    } else {
	      BgSendPacket(i,j,k,ANYTHREAD,exchangeID,SMALL_WORK, sizeof(exchangeMsg),(char *)msg);
	    }
	  }
	}
      }
      // resetting the value of the reduction count
      localdata->reduction_count = numBgX * numBgY * numBgZ;
      // resetting the completion status of the computation to assume true
      localdata->done = true; 
    } else {
      // instruct all computing nodes to output their individual results
      for (int i=0; i<numBgX;i++) {
	for (int j=0; j<numBgY;j++) {
	  for (int k=0; k<numBgZ;k++) {
	    outputMsg *msg = new outputMsg;
	    if ((i == 0) && (j == 0) && (k == 0)) {
	      BgSendLocalPacket(ANYTHREAD,outputID, LARGE_WORK, sizeof(outputMsg), (char *)msg);
	    } else {
	      BgSendPacket(i,j,k,ANYTHREAD,outputID,SMALL_WORK, sizeof(outputMsg), (char *)msg);
	    }
	  }
	}
      }
    }
  } 
}

void ghostrecv(char *info)
{
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);

  /*
  CmiPrintf("Node (%d,%d,%d) receiving exchange message\n",
	   x, y, z);
  */
  nodeData *localdata = (nodeData *)BgGetNodeData();
  
  int x_size = localdata->my_x_size;
  int y_size = localdata->my_y_size;
  int z_size = localdata->my_z_size;

  int x1_total = localdata->ghost_x1_elements_total;
  int x2_total = localdata->ghost_x2_elements_total;
  int y1_total = localdata->ghost_y1_elements_total;
  int y2_total = localdata->ghost_y2_elements_total;
  int z1_total = localdata->ghost_z1_elements_total;
  int z2_total = localdata->ghost_z2_elements_total;

  // determine the source of the ghost region and copy the data to the
  // appropriate local ghost region.
  if (((ghostMsg *)info)->source == LEFT) {
    for (int count=0;count<(((ghostMsg*)info)->datacount);count++) {
      int x_offset = 
	((localdata->ghost_x1_elements_current)/(y_size*z_size)) % 1;
      int y_offset =
	((localdata->ghost_x1_elements_current)/z_size) % y_size;
      int z_offset =
	(localdata->ghost_x1_elements_current) % z_size;
      (localdata->ghost_x1)[x_offset][y_offset][z_offset] =
	((ghostMsg*)info)->data[count];
      localdata->ghost_x1_elements_current++;
    }
  } else if (((ghostMsg *)info)->source == RIGHT) {
    for (int count=0;count<(((ghostMsg*)info)->datacount);count++) {
      int x_offset = 
	((localdata->ghost_x2_elements_current)/(y_size*z_size)) % 1;
      int y_offset =
	((localdata->ghost_x2_elements_current)/z_size) % y_size;
      int z_offset =
	(localdata->ghost_x2_elements_current) % z_size;
      (localdata->ghost_x2)[x_offset][y_offset][z_offset] =
	((ghostMsg*)info)->data[count];
      localdata->ghost_x2_elements_current++;
    }
  } else if (((ghostMsg *)info)->source == BELOW) {
    for (int count=0;count<(((ghostMsg*)info)->datacount);count++) {
      int x_offset = 
	((localdata->ghost_y1_elements_current)/(1*z_size)) % x_size;
      int y_offset =
	((localdata->ghost_y1_elements_current)/z_size) % 1;
      int z_offset =
	(localdata->ghost_y1_elements_current) % z_size;
      (localdata->ghost_y1)[x_offset][y_offset][z_offset] =
	((ghostMsg*)info)->data[count];
      localdata->ghost_y1_elements_current++;
    }
  } else if (((ghostMsg *)info)->source == ABOVE) {
    for (int count=0;count<(((ghostMsg*)info)->datacount);count++) {
      int x_offset = 
	((localdata->ghost_y2_elements_current)/(1*z_size)) % x_size;
      int y_offset =
	((localdata->ghost_y2_elements_current)/z_size) % 1;
      int z_offset =
	(localdata->ghost_y2_elements_current) % z_size;
      (localdata->ghost_y2)[x_offset][y_offset][z_offset] =
	((ghostMsg*)info)->data[count];
      localdata->ghost_y2_elements_current++;
    }
  } else if (((ghostMsg *)info)->source == FRONT) {
    for (int count=0;count<(((ghostMsg*)info)->datacount);count++) {
      int x_offset = 
	((localdata->ghost_z1_elements_current)/(y_size*1)) % x_size;
      int y_offset =
	((localdata->ghost_z1_elements_current)/1) % y_size;
      int z_offset =
	(localdata->ghost_z1_elements_current) % 1;
      (localdata->ghost_z1)[x_offset][y_offset][z_offset] =
	((ghostMsg*)info)->data[count];
      localdata->ghost_z1_elements_current++;
    }
  } else if (((ghostMsg *)info)->source == BACK) {
    for (int count=0;count<(((ghostMsg*)info)->datacount);count++) {
      int x_offset = 
	((localdata->ghost_z2_elements_current)/(y_size*1)) % x_size;
      int y_offset =
	((localdata->ghost_z2_elements_current)/1) % y_size;
      int z_offset =
	(localdata->ghost_z2_elements_current) % 1;
      (localdata->ghost_z2)[x_offset][y_offset][z_offset] =
	((ghostMsg*)info)->data[count];
      localdata->ghost_z2_elements_current++;
    }
  }

  // if that was the last message
  if ((x1_total == localdata->ghost_x1_elements_current) &&
      (x2_total == localdata->ghost_x2_elements_current) &&
      (y1_total == localdata->ghost_y1_elements_current) &&
      (y2_total == localdata->ghost_y2_elements_current) &&
      (z1_total == localdata->ghost_z1_elements_current) &&
      (z2_total == localdata->ghost_z2_elements_current)) {
    // reset exchange counts
    localdata->ghost_x1_elements_current = 0;
    localdata->ghost_x2_elements_current = 0;
    localdata->ghost_y1_elements_current = 0;
    localdata->ghost_y2_elements_current = 0;
    localdata->ghost_z1_elements_current = 0;
    localdata->ghost_z2_elements_current = 0;
    
    computeMsg *msg = new computeMsg;
    BgSendLocalPacket(ANYTHREAD,computeID, LARGE_WORK, sizeof(computeMsg), (char *)msg); // get to work!
  }
  delete (ghostMsg*)info;
}

void outputData(char *info) {
  delete (outputMsg *)info;
  int x,y,z;
  BgGetMyXYZ(&x,&y,&z);

  /*
  CmiPrintf("Node (%d,%d,%d) printing data.\n",
	   x, y, z);
  */
  nodeData *localdata = (nodeData *)BgGetNodeData();
  
  int x_size = localdata->my_x_size;
  int y_size = localdata->my_y_size;
  int z_size = localdata->my_z_size;

  CmiPrintf("Final output at Node (%d,%d,%d) with iteration count = %d:\n",
	   x, y, z,
	   localdata->iteration_count);
//  printArray(localdata->maindata,x_size,y_size,z_size);

  if ((x == 0) && (y == 0) && (z == 0)) {
    BgShutdown();
  }
}

void initArray(double target[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
	       int x, int y, int z) {
  for (int i=0; i < x; i++) {
    for (int j=0; j < y; j++) {
      for (int k=0; k < z; k++) {
	target[i][j][k] = (i+j+k)*1.0;
      }
    }
  }
}

void copyArray(double source[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
	       double dest[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
	       int x, int y, int z,
	       int offset_x, int offset_y, int offset_z) {
  for (int i=0; i < x; i++) {
    for (int j=0; j < y; j++) {
      for (int k=0; k < z; k++) {
	dest[i][j][k] = source[i+offset_x][j+offset_y][k+offset_z];
      }
    }
  }
}

void copyXArray(double source[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		double dest[1][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		int x, int y, int z,
		int offset_x, int offset_y, int offset_z) {
  for (int i=0; i < x; i++) {
    for (int j=0; j < y; j++) {
      for (int k=0; k < z; k++) {
	dest[i][j][k] = source[i+offset_x][j+offset_y][k+offset_z];
      }
    }
  }
}

void copyYArray(double source[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		double dest[MAX_ARRAY_SIZE][1][MAX_ARRAY_SIZE],
		int x, int y, int z,
		int offset_x, int offset_y, int offset_z) {
  for (int i=0; i < x; i++) {
    for (int j=0; j < y; j++) {
      for (int k=0; k < z; k++) {
	dest[i][j][k] = source[i+offset_x][j+offset_y][k+offset_z];
      }
    }
  }
}

void copyZArray(double source[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		double dest[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][1],
		int x, int y, int z,
		int offset_x, int offset_y, int offset_z) {
  for (int i=0; i < x; i++) {
    for (int j=0; j < y; j++) {
      for (int k=0; k < z; k++) {
	dest[i][j][k] = source[i+offset_x][j+offset_y][k+offset_z];
      }
    }
  }
}

void printArray(double target[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE][MAX_ARRAY_SIZE],
		int x, int y, int z) {
  for (int i=0; i < x; i++) {
    for (int j=0; j < y; j++) {
      for (int k=0; k < z; k++) {
	CmiPrintf("%f ",target[i][j][k]);
      }
      CmiPrintf("\n");
    }
  }
}

