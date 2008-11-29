// Read-only global variables

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Compute compute;

/*readonly*/ int N;
/*readonly*/ int K;
/*readonly*/ int num_chares_per_dim;
/*readonly*/ int T;

static unsigned long next = 1;

int myrand(int numpes) {
  next = next * 1103515245 + 12345;
  return((unsigned)(next/65536) % numpes);
}

#define USE_TOPOMAP	0
#define USE_RRMAP	0
#define USE_RNDMAP	0


#define UNROLL_DEPTH 3
#define MAGIC_A 1.0
#define MAGIC_B 4.0

class Main : public CBase_Main {
  
  double startTime;
  double endTime;
  
  public:    
  Main(CkArgMsg* m);
  void done();
};

class Compute: public CBase_Compute {
  int step;
  float *A[2], *B[2], *C;

  int row, col;
  int remaining;
  int whichLocal;
  int iteration;
  //int comps;


  public:
  Compute();
  Compute(CkMigrateMessage* m);
  ~Compute();

  void start();
  void compute();
  void recvBlockA(float *block, int size, int whichBuf);
  void recvBlockB(float *block, int size, int whichBuf);
  void resumeFromBarrier();

};

class ComputeMap : public CBase_ComputeMap {
  
  int arrayDimX, arrayDimY;

  int *map;

  public:
  ComputeMap(int, int);
  ~ComputeMap(){delete []map;}
  int procNum(int, const CkArrayIndex &idx);

};
