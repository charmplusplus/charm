#include "array_map.decl.h"

#include <numeric>

// Custom map which sums up the integers in the index and maps to procs based
// on their least significant bits. Declared as a group in the .ci file.
class BitMap : public CkArrayMap {
private:
  int maxBitMap;
public:
  BitMap() {
    if ((CkNumPes() & (CkNumPes() - 1)) != 0) {
      CkAbort("Bit map requires a power of two number of PEs\n");
    }
    maxBitMap = CkNumPes() - 1;
  }

  int gcd(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;
    if (a == b) return a;
    if (a > b) return gcd(a-b, b);
    return gcd(a, b-a);
  }

  int registerArray(CkArrayIndex& numElements, CkArrayID aid) { return 0; }

  // Call that returns the homePE of element with index idx
  int procNum(int arrayHdl, const CkArrayIndex& idx) {
    int i = 0;
    // Sum up the number of ints in the index (not the same as dimension)
    for (int x = 0; x < idx.nInts; x++) {
      i += idx.data()[x];
    }
    // Mask the sum to a PE number
    return i & maxBitMap;
  }

  // Call that is run when an array is bulk constructed using ckNew to populate
  // the elements on each PE. Called once per PE. In most cases this function
  // can be left out in favor of the default implementation.
  void populateInitial(
      int arrayHdl, CkArrayOptions& options, void* ctorMsg, CkArray* mgr) {
    // Start, end, and step determine the initial elements to be created.
    CkArrayIndex start = options.getStart();
    CkArrayIndex end = options.getEnd();
    CkArrayIndex step = options.getStep();

    if (end.dimension == 1) {
      // For the simple case where we have a 1D chare array, we can directly
      // compute which elements belong on our PE and create them.
      int mod = (start.data()[0] + (CkMyPe() * step.data()[0])) % CkNumPes();
      int first = start.data()[0] + mod * step.data()[0];
      int lcm = (CkNumPes() * step.data()[0]) / gcd(CkNumPes(), step.data()[0]);
      for (int i = first; i < end.data()[0]; i += lcm) {
        mgr->insertInitial(CkArrayIndex1D(i), CkCopyMsg(&ctorMsg));
      }

      // Inform the mgr that we are done creating elements, and free ctorMsg.
      mgr->doneInserting();
      CkFreeMsg(ctorMsg);
    } else {
      // For higher dimension indices we fall back to the default behavior,
      // which loops through the entire requested index space and checks if
      // procNum(...) returns our PE.
      CkArrayMap::populateInitial(arrayHdl, options, ctorMsg, mgr);
    }
  }
};

class Main : public CBase_Main {
public:
  Main(CkArgMsg* msg) {
    delete msg;

    // Sums for testing mapping
    int sum1, sum2;
    // Create a new map object
    CProxy_BitMap myMap = CProxy_BitMap::ckNew();


    // Create a simple array with elements [0,10) (indices sum to 45)
    sum1 = 10 * (0 + 9) / 2;
    CkArrayOptions opts1(10);
    opts1.setMap(myMap);
    CProxy_Array1 a1 = CProxy_Array1::ckNew(sum1, opts1);

    // Create a more complex 1D array with elements [2,32) with step size of 3
    // (indices sum to 155)
    sum1 = 10 * (2 + 29) / 2;
    CkArrayOptions opts2;
    opts2.setStart(CkArrayIndex1D(2));
    opts2.setEnd(CkArrayIndex1D(32));
    opts2.setStep(CkArrayIndex1D(3));
    opts2.setMap(myMap);
    CProxy_Array1 a2 = CProxy_Array1::ckNew(sum1, opts2);

    // Create a 2D array whose first dimension has elements [0,10), and whose
    // second dimension has elements [0,2). (sums of 90 and 10).
    sum1 = (10 * (0 + 9) / 2) * 2;
    sum2 = (2 * (0 + 1) / 2) * 10;
    CkArrayOptions opts3(10,2);
    opts3.setMap(myMap);
    CProxy_Array2 a3 = CProxy_Array2::ckNew(sum1, sum2, opts3);

    // After all arrays are created and sums are checked we can exit.
    CkStartQD(CkCallback(CkCallback::ckExit));
  }
};

class Array1 : public CBase_Array1 {
private:
  int sum;

public:
  Array1(int s) : sum(s) {
    // Asser that the map put this element in the correct spot.
    CkAssert(thisIndex % CkNumPes() == CkMyPe());

    CkPrintf("Array1: created element %d on %d\n", thisIndex, CkMyPe());

    // Check the sum of all elements indices to check that populateInitial
    // created all the elements it should have.
    CkCallback cb = CkCallback(CkReductionTarget(Array1, checkSum), thisProxy);
    contribute(sizeof(int), &thisIndex, CkReduction::sum_int, cb);
  }

  void checkSum(int s) {
    CkAssert(sum == s);
  }
};

class Array2 : public CBase_Array2 {
private:
  int sum1, sum2;
public:
  Array2(int s1, int s2) : sum1(s1), sum2(s2) {
    // Asser that the map put this element in the correct spot.
    CkAssert((thisIndex.x + thisIndex.y) % CkNumPes() == CkMyPe());

    CkPrintf("Array2: created element (%d,%d) on %d\n",
        thisIndex.x, thisIndex.y, CkMyPe());

    // Check the sum of all elements indices to check that populateInitial
    // created all the elements it should have.
    int indices[] = { thisIndex.x, thisIndex.y };
    CkCallback cb = CkCallback(CkReductionTarget(Array2, checkSum), thisProxy);
    contribute(2 * sizeof(int), indices, CkReduction::sum_int, cb);
  }

  void checkSum(int s1, int s2) {
    CkAssert(sum1 == s1 && sum2 == s2);
  }
};

#include "array_map.def.h"
