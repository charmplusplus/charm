#include "sparse.h"

/*readonly*/bool verbose;
/*readonly*/CProxy_Main main_proxy;

Main::Main(CkArgMsg* args) {
  verbose = false;
  main_proxy = thisProxy;
  thisProxy.run_tests();
}

void Main::checkTest(int actual) {
  CkPrintf("Expected: %d, Actual: %d\n", expected, actual);
  if (expected != actual) {
    CkAbort("Test failed!\n");
  }
}

void Main::setExpected(CkArrayIndex end) {
  if (end.dimension <= 3) {
    setExpected(CkArrayIndex3D(0,0,0), end, CkArrayIndex3D(1,1,1));
  } else {
    setExpected(CkArrayIndex6D(0,0,0,0,0,0), end, CkArrayIndex6D(1,1,1,1,1,1));
  }
}

void Main::setExpected(CkArrayIndex start, CkArrayIndex end, CkArrayIndex step) {
  int dimension = end.dimension;
  int starts[6], ends[6], steps[6];
  for (int i = 0; i < 6; i++) {
    if (i < dimension && dimension <= 3) {
      starts[i] = start.data()[i];
      ends[i] = end.data()[i];
      steps[i] = step.data()[i];
    } else if (i < dimension && dimension > 3) {
      starts[i] = ((short*)start.data())[i];
      ends[i] = ((short*)end.data())[i];
      steps[i] = ((short*)step.data())[i];
    } else {
      starts[i] = 0;
      ends[i] = 1;
      steps[i] = 1;
    }
  }

  expected = 0;
  for (int i0 = starts[0]; i0 < ends[0]; i0 += steps[0]) {
    for (int i1 = starts[1]; i1 < ends[1]; i1 += steps[1]) {
      for (int i2 = starts[2]; i2 < ends[2]; i2 += steps[2]) {
        for (int i3 = starts[3]; i3 < ends[3]; i3 += steps[3]) {
          for (int i4 = starts[4]; i4 < ends[4]; i4 += steps[4]) {
            for (int i5 = starts[5]; i5 < ends[5]; i5 += steps[5]) {
              expected += i0 + i1 + i2 + i3 + i4 + i5;
            }
          }
        }
      }
    }
  }
}
              

void Main::test1D(CkArrayOptions options) {
  CkPrintf("1D Test - start: %d, end: %d, step: %d\n",
      options.getStart().data()[0],
      options.getEnd().data()[0],
      options.getStep().data()[0]);
  CProxy_Array1D proxy = CProxy_Array1D::ckNew(options);
  proxy.ping();
}

void Main::test2D(CkArrayOptions options) {
  CkPrintf("2D Test - start: (%d,%d), end: (%d,%d), step: (%d,%d)\n",
      options.getStart().data()[0], options.getStart().data()[1],
      options.getEnd().data()[0], options.getEnd().data()[1],
      options.getStep().data()[0], options.getStep().data()[1]);
  CProxy_Array2D proxy = CProxy_Array2D::ckNew(options);
  proxy.ping();
}

void Main::test3D(CkArrayOptions options) {
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();
  CkPrintf("3D Test - start: (%d,%d,%d), end: (%d,%d,%d), step: (%d,%d,%d)\n",
      start.data()[0], start.data()[1], start.data()[2],
      end.data()[0], end.data()[1], end.data()[2],
      step.data()[0], step.data()[1], step.data()[2]);
  CProxy_Array3D proxy = CProxy_Array3D::ckNew(options);
  proxy.ping();
}

void Main::test4D(CkArrayOptions options) {
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();
  short int* start_data = (short int*)start.data();
  short int* end_data = (short int*)end.data();
  short int* step_data = (short int*)step.data();
  CkPrintf("4D Test - start: (%d,%d,%d,%d), "
           "end: (%d,%d,%d,%d), "
           "step: (%d,%d,%d,%d)\n",
      start_data[0], start_data[1], start_data[2], start_data[3],
      end_data[0], end_data[1], end_data[2], end_data[3],
      step_data[0], step_data[1], step_data[2], step_data[3]);
  CProxy_Array4D proxy = CProxy_Array4D::ckNew(options);
  proxy.ping();
}

void Main::test5D(CkArrayOptions options) {
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();
  short int* start_data = (short int*)start.data();
  short int* end_data = (short int*)end.data();
  short int* step_data = (short int*)step.data();
  CkPrintf("5D Test - start: (%d,%d,%d,%d,%d), "
           "end: (%d,%d,%d,%d,%d), "
           "step: (%d,%d,%d,%d,%d)\n",
      start_data[0], start_data[1], start_data[2], start_data[3], start_data[4],
      end_data[0], end_data[1], end_data[2], end_data[3], end_data[4],
      step_data[0], step_data[1], step_data[2], step_data[3], step_data[4]);
  CProxy_Array5D proxy = CProxy_Array5D::ckNew(options);
  proxy.ping();
}

void Main::test6D(CkArrayOptions options) {
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();
  short int* start_data = (short int*)start.data();
  short int* end_data = (short int*)end.data();
  short int* step_data = (short int*)step.data();
  CkPrintf("6D Test - start: (%hd,%hd,%hd,%hd,%hd,%hd), "
           "end: (%hd,%hd,%hd,%hd,%hd,%hd), "
           "step: (%hd,%hd,%hd,%hd,%hd,%hd)\n",
      start_data[0], start_data[1], start_data[2], start_data[3], start_data[4], start_data[5],
      end_data[0], end_data[1], end_data[2], end_data[3], end_data[4], end_data[5],
      step_data[0], step_data[1], step_data[2], step_data[3], step_data[4], step_data[5]);
  CProxy_Array6D proxy = CProxy_Array6D::ckNew(options);
  proxy.ping();
}

Array1D::Array1D() {
  if (verbose) {
    CkPrintf("%i: [%i]\n", CkMyPe(), thisIndex);
  }
}

void Array1D::ping() {
  int contribution = thisIndex;
  CkCallback cb(CkReductionTarget(Main, testDone), main_proxy);
  contribute(sizeof(contribution), &contribution, CkReduction::sum_int, cb);
}

Array2D::Array2D() {
  if (verbose) {
    CkPrintf("%i: [%i, %i]\n", CkMyPe(), thisIndex.x, thisIndex.y);
  }
}

void Array2D::ping() {
  int contribution = thisIndex.x + thisIndex.y;
  CkCallback cb(CkReductionTarget(Main, testDone), main_proxy);
  contribute(sizeof(contribution), &contribution, CkReduction::sum_int, cb);
}

Array3D::Array3D() {
  if (verbose) {
    CkPrintf("%i: [%i, %i, %i]\n", CkMyPe(), thisIndex.x, thisIndex.y, thisIndex.z);
  }
}

void Array3D::ping() {
  int contribution = thisIndex.x + thisIndex.y + thisIndex.z;
  CkCallback cb(CkReductionTarget(Main, testDone), main_proxy);
  contribute(sizeof(contribution), &contribution, CkReduction::sum_int, cb);
}

Array4D::Array4D() {
  if (verbose) {
    CkPrintf("%i: [%i, %i, %i, %i]\n", CkMyPe(), thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
  }
}

void Array4D::ping() {
  int contribution = thisIndex.w + thisIndex.x + thisIndex.y + thisIndex.z;
  CkCallback cb(CkReductionTarget(Main, testDone), main_proxy);
  contribute(sizeof(contribution), &contribution, CkReduction::sum_int, cb);
}

Array5D::Array5D() {
  if (verbose) {
    CkPrintf("%i: [%i, %i, %i, %i, %i]\n", CkMyPe(), thisIndex.v, thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
  }
}

void Array5D::ping() {
  int contribution = thisIndex.v + thisIndex.w + thisIndex.x + thisIndex.y + thisIndex.z;
  CkCallback cb(CkReductionTarget(Main, testDone), main_proxy);
  contribute(sizeof(contribution), &contribution, CkReduction::sum_int, cb);
}

Array6D::Array6D() {
  if (verbose) {
    CkPrintf("%i: [%i, %i, %i, %i, %i, %i]\n", CkMyPe(), thisIndex.x1, thisIndex.y1, thisIndex.z1, thisIndex.x2, thisIndex.y2, thisIndex.z2);
  }
}

void Array6D::ping() {
  int contribution = thisIndex.x1 + thisIndex.y1 + thisIndex.z1 + thisIndex.x2 + thisIndex.y2 + thisIndex.z2;
  CkCallback cb(CkReductionTarget(Main, testDone), main_proxy);
  contribute(sizeof(contribution), &contribution, CkReduction::sum_int, cb);
}

#include "sparse.def.h"
