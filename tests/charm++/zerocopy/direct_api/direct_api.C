#include "direct_api.decl.h"

#define CONSTANT 557

int numElements;
CProxy_Main mProxy;

void assignValuesToIndex(int *arr, int size);
void assignValuesToConstant(int *arr, int size, int constantVal);
void verifyValuesWithConstant(int *arr, int size, int constantVal);
void verifyValuesWithIndex(int *arr, int size, int startIndex);

class Main : public CBase_Main {
  public:
    Main(CkArgMsg *m) {
      if(m->argc !=2 ) {
        CkAbort("Usage: ./zerocopy_with_qd <array size>, where <array size> is even\n");
      }
      numElements = atoi(m->argv[1]);
      if(numElements % 2 != 0) {
        CkAbort("<array size> argument is not even\n");
      }
      delete m;

      mProxy = thisProxy;

      CProxy_testArr arr1 = CProxy_testArr::ckNew(numElements);
      arr1.test1();
    };

    void test1Done() {
      CkPrintf("[%d][%d][%d] Test 1 (ZC Direct API with srcSize < destSize (both GET and PUT) completed\n", CkMyPe(), CkMyNode(), CkMyRank());
      CkExit();
    }
};

class testArr : public CBase_testArr {

  int destIndex;
  int *buff1, *buff2;

  static int largeSize;
  static int smallSize;
  CkCallback test1DoneCb;
  CkCallback destCompletionCb;

  public:

    testArr() {
      destIndex = numElements - 1 - thisIndex;

      if(thisIndex < numElements/2) {
        buff1 = new int[smallSize]; // Source 1
        buff2 = new int[largeSize]; // Destination 2
      } else {
        buff1 = new int[largeSize]; // Destination 1
        buff2 = new int[smallSize]; // Source 2
      }

      test1DoneCb = CkCallback(CkIndex_Main::test1Done(), mProxy);
      destCompletionCb = CkCallback(CkIndex_testArr::validateReceivedData(),thisProxy[thisIndex]);
    }

    void test1() {
      if(thisIndex < numElements/2) {
        // Assign the source buffer (buff1) to CONSTANT
        assignValuesToConstant(buff1, smallSize, CONSTANT);

        // Create a CkNcpyBuffer (for get source) to send to destIndex
        CkNcpyBuffer src1(buff1, sizeof(int) * smallSize);

        // Assign the destination buffer (buff2) to the array index
        assignValuesToIndex(buff2, largeSize);

        // Create a CkNcpyBuffer (for put destination) to send to destIndex
        CkNcpyBuffer dest2(buff2, sizeof(int) * largeSize, destCompletionCb);

        thisProxy[destIndex].recvBufferInfo(src1, dest2);
      }
    }

    void validateReceivedData() {
      int *buff;
      if(thisIndex < numElements/2) {
        buff = buff2;
      } else {
        buff = buff1;
      }

      // Verify that source buffer values were received in dest buffer (i.e. the first 'smallSize' elements are all = CONSTANT
      verifyValuesWithConstant(buff, smallSize, CONSTANT);

      // Verify that the remaining values are equal to Index
      verifyValuesWithIndex(buff, largeSize, smallSize);

      contribute(test1DoneCb);
    }

    // executed on half of the array elements
    void recvBufferInfo(CkNcpyBuffer src1, CkNcpyBuffer dest2) {
      // Assign the destination buffer (buff 1) to the array index
      assignValuesToIndex(buff1, largeSize);

      // Create a CkNcpyBuffer (for get destination)
      CkNcpyBuffer dest1(buff1, sizeof(int) * largeSize, destCompletionCb);
      dest1.get(src1);

      // Assign the source buffer (buff2) to CONSTANT
      assignValuesToConstant(buff2, smallSize, CONSTANT);

      // Create a CkNcpyBuffer (for put source)
      CkNcpyBuffer src2(buff2, sizeof(int) * smallSize);
      src2.put(dest2);
    }
};

int testArr::largeSize = 2000;
int testArr::smallSize = 200;

// Util methods
void assignValuesToIndex(int *arr, int size){
  for(int i=0; i<size; i++)
     arr[i] = i;
}

void assignValuesToConstant(int *arr, int size, int constantVal){
  for(int i=0; i<size; i++)
     arr[i] = constantVal;
}

void verifyValuesWithConstant(int *arr, int size, int constantVal){
  for(int i=0; i<size; i++) {
     CkAssert(arr[i] == constantVal);
  }
}

void verifyValuesWithIndex(int *arr, int size, int startIndex){
  for(int i=startIndex; i<size; i++)
     CkAssert(arr[i] == i);
}

#include "direct_api.def.h"
