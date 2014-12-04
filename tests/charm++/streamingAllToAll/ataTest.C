#include "NDMeshStreamer.h"
#include "ataTest.decl.h"
#include "TopoManager.h"
#include "ataDatatype.h"
#include "limits.h"
#include "envelope.h"


CProxy_Main mainProxy;
CProxy_GroupMeshStreamer<DataItem, Participant, SimpleMeshRouter> aggregator;
CProxy_Participant allToAllGroup;

#define TRAM_BUFFER_SIZE (16 * 1024 / DATA_ITEM_SIZE)

enum allToAllTestType{usingTram, directSends, finishedTests};

class Main : public CBase_Main {
private:
  double startTime;
  int dataSizeMin;
  int dataSizeMax;
  int iters;
  int bufferSize;
  int testType;
public:
  Main(CkArgMsg *args) {

    if (args->argc >= 3) {
      dataSizeMin = atoi(args->argv[1]);
      dataSizeMax = atoi(args->argv[2]);
    }
    else {
      dataSizeMin = 32;
      dataSizeMax = 16384;
    }
    bufferSize =
      args->argc == 4 ? atoi(args->argv[3]) : TRAM_BUFFER_SIZE;
    CkPrintf("size of envelope: %d\n\n", sizeof(envelope));
    delete args;

    iters = dataSizeMin / DATA_ITEM_SIZE;
    allToAllGroup = CProxy_Participant::ckNew();

#if !CMK_BLUEGENEQ
    int nDims = 2;
    int dims[2] = {CkNumNodes(), CkNumPes() / CkNumNodes()};
    CkPrintf("TEST 1: Using %dD TRAM Topology: %d %d\n",
             nDims, dims[0], dims[1]);

    // Alternative 3D topology
    // int nDims = 3;
    // int dim1 = CkNumNodes();
    // int dim2 = 1;
    // if (dim1 != 1) {
    //   while (dim2 < dim1) {
    //     dim2 *= 2;
    //     dim1 /= 2;
    //   }
    // }
    // int dims[3] = {dim1, dim2, CkNumPes() / CkNumNodes()};
    // CkPrintf("Topology: %d %d %d\n", dims[0], dims[1], dims[2]);
#else
    TopoManager tmgr;

    int nDims = 3;
    int dims[3] = {tmgr.getDimNA() * tmgr.getDimNB(),
                   tmgr.getDimNC() * tmgr.getDimND() * tmgr.getDimNE(),
                   tmgr.getDimNT()};
    CkPrintf("TEST 1: Using %dD TRAM Topology: %d %d %d\n",
             nDims, dims[0], dims[1], dims[2]);

    // Alternative TRAM topologies for Blue Gene/Q using Topology Manager
    // int nDims = 4;
    // int dims[4] = {tmgr.getDimNA() * tmgr.getDimNB(), tmgr.getDimNC(),
    //                tmgr.getDimND() * tmgr.getDimNE(), tmgr.getDimNT()};

    // int nDims = 6;
    // int dims[6] = {tmgr.getDimNA(), tmgr.getDimNB(), tmgr.getDimNC(),
    //                tmgr.getDimND() * tmgr.getDimNE(),
    //                tmgr.getDimNT() / 8, 8};
#endif

    mainProxy = thisProxy;


    aggregator = CProxy_GroupMeshStreamer<DataItem, Participant,
                                          SimpleMeshRouter>::
    ckNew(nDims, dims, allToAllGroup, bufferSize, 1, 0.1);
    testType = usingTram;
  }

  void prepare() {
    if (testType == usingTram) {
      CkCallback startCb(CkIndex_Main::start(), thisProxy);
      CkCallback endCb(CkIndex_Main::allDone(), thisProxy);
      aggregator.init(1, startCb, endCb, INT_MIN, false);
    }
    else {
      start();
    }
  }

  void start() {
    startTime = CkWallTimer();
    allToAllGroup.communicate(iters, testType == usingTram);
  }

  void allDone() {
    double elapsedTime = CkWallTimer() - startTime;
    CkPrintf("Elapsed time for all-to-all of %8d bytes sent in %6d %10s"
             " of %2d bytes each (%3s using TRAM): %.6f seconds\n",
             iters * DATA_ITEM_SIZE, iters,
             iters == 1 ? "iteration" : "iterations", DATA_ITEM_SIZE,
             testType == directSends ? "not" : "", elapsedTime);
    if (iters == dataSizeMax / DATA_ITEM_SIZE) {
      ++testType;
      if (testType == finishedTests) {
        CkExit();
      }
      else {
        CkPrintf("\nTEST 2: Using point to point sends\n");
        iters = dataSizeMin / DATA_ITEM_SIZE;
        prepare();
      }
    }
    else {
      iters *= 2;
      prepare();
    }
  }

};


class Participant : public CBase_Participant{
private:
  int *neighbors;
  DataItem myItem;
public:
  Participant() {

    int numPes = CkNumPes();
    neighbors = new int[numPes];
    for (int i = 0; i < numPes; i++) {
      neighbors[i] = i;
    }

    // shuffle to prevent bottlenecks
    for (int i = numPes-1; i >= 0; i--) {
      int shuffleIndex = rand() % (i+1);
      int temp = neighbors[i];
      neighbors[i] = neighbors[shuffleIndex];
      neighbors[shuffleIndex] = temp;
    }

    contribute(CkCallback(CkReductionTarget(Main, prepare), mainProxy));
  }

  void communicate(int iters, bool useTram) {
    GroupMeshStreamer<DataItem, Participant, SimpleMeshRouter> *localStreamer;
    if (useTram) {
      localStreamer = aggregator.ckLocalBranch();
    }

    int ctr = 0;
    for (int i = 0; i < iters; i++) {
      for (int j=0; j<CkNumPes(); j++) {
        if (useTram) {
          localStreamer->insertData(myItem, neighbors[j]);
        }
        else {
          allToAllGroup[neighbors[j]].receive(myItem);
          ctr++;
        }
      }
      if (!useTram) {
        if (ctr == 1024) {
          ctr = 0;
          CthYield();
        }
      }
    }
    if (useTram) {
      localStreamer->done();
    }
    else {
      contribute(CkCallback(CkReductionTarget(Main, allDone), mainProxy));
    }
  }

  void process(const DataItem &item) {
    // nothing here - we only care about communication
  }

  void receive(DataItem item) {
    // nothing here - we only care about communication
  }

};

#include "ataTest.def.h"
