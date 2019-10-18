#include "simpleZCVec.decl.h"
#include <numeric>
#include <vector>

int numElements;

//Main chare
class Main : public CBase_Main{
  int counter;

  public:
    Main(CkArgMsg *m){
      if(m->argc!=2){
        ckout<<"Usage: zerocopy <numelements>"<<endl;
        CkExit(1);
      }
      numElements = atoi(m->argv[1]);
      delete m;
      if(numElements%2 != 0){
        ckout<<"Argument <numelements> should be even"<<endl;
        CkExit(1);
      }

      counter = 0;
      CProxy_zerocopyObject zerocopyObj = CProxy_zerocopyObject::ckNew(thisProxy, numElements);
    }

    void done(){
      CkPrintf("[%d][%d][%d] All sending and receiving of vectors is complete\n", CmiMyPe(), CmiMyNode(), CmiMyRank());
      CkExit();
    }
};

//zerocopy object chare
class zerocopyObject : public CBase_zerocopyObject{
  int destIndex, iSize;
  CkCallback cb, doneCb;
  CProxy_Main mainProxy;
  std::vector<int> vec2; // vec2, member variable
  int cbCounter;


  public:
    zerocopyObject(CProxy_Main mProxy){

      mainProxy = mProxy;

      destIndex = numElements - 1 - thisIndex;
      iSize = 200;
      cbCounter = 0;

      doneCb = CkCallback(CkReductionTarget(Main, done), mainProxy);

      if(thisIndex < numElements/2) {

        // vec1, dynamically allocated vector pointer
        std::vector<int> *vec1 = new std::vector<int>(iSize);
        std::iota(vec1->begin(), vec1->end(), 0);

        vec2.resize(iSize);
        std::iota(vec2.begin(), vec2.end(), 0);

        int idx_zerocopySenderCompleteCB = CkIndex_zerocopyObject::zerocopySenderCompleteCB(NULL);
        cb = CkCallback(idx_zerocopySenderCompleteCB, thisProxy[thisIndex]);

        thisProxy[destIndex].zerocopySend(CkSendBuffer(vec1->data(), cb), vec1->size(), CkSendBuffer(vec2.data(), cb), vec2.size());
      }
    }

    void zerocopySenderCompleteCB(CkDataMsg *m){ // Get access to the array information sent via zerocopy
      CkNcpyBuffer *src = (CkNcpyBuffer *)(m->data);

      // Delete the dynamically allocated vector
      delete (int *)(src->ptr);

      delete m;

      // wait for both the vectors to be sent
      if(++cbCounter == 2)
        contribute(doneCb); // Signal to the main chare on completion
    }

    void zerocopySend(int *ptr1, int size1, int *ptr2, int size2){
      for(int i=0; i<size1; i++) CkAssert(ptr1[i] == i);
      for(int i=0; i<size2; i++) CkAssert(ptr2[i] == i);

      // Signal to the main chare on completion
      contribute(doneCb);
    }
};

#include "simpleZCVec.def.h"
