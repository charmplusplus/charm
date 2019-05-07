#include "zc_post_modify_size.decl.h"
#define SIZE 2000
#define NUM_ELEMENTS_PER_PE 10
#define CONSTANT 188

CProxy_arr1 arrProxy;
CProxy_grp1 grpProxy;
CProxy_nodegrp1 ngProxy;
CProxy_tester chareProxy;

void assignValuesToIndex(int *arr, int size){
  for(int i=0; i<size; i++)
     arr[i] = i;
}

void assignValuesToConstant(int *arr, int size, int constantVal){
  for(int i=0; i<size; i++)
     arr[i] = constantVal;
}

void verifyValuesWithConstant(int *arr, int size, int constantVal){
  for(int i=0; i<size; i++)
     CkAssert(arr[i] == constantVal);
}

void verifyValuesWithIndex(int *arr, int size, int startIndex){
  for(int i=startIndex; i<size; i++)
     CkAssert(arr[i] == i);
}

class main : public CBase_main {
  public:
    void main(CkArgMsg *m) {

      // Create a chare array
      arrProxy = CProxy_arr1::ckNew(CkNumPes() * NUM_ELEMENTS_PER_PE);

      // Create a group
      grpProxy = CProxy_grp1::ckNew();

      // Create a nodegroup
      ngProxy = CProxy_nodegrp1::ckNew();

    }
};

class tester1 : public CBase_tester1 {
  int *srcBuffer;
  public:
    void tester1() {
      srcBuffer = new int[SIZE];
      assignValuesToConstant(srcBuffer, size, CONSTANT);
      arrProxy[9].recv_zerocopy(srcBuffer, SIZE);
    }
};

class arr1 : public CBase_arr1 {
  int *destBuffer;
  public:
    void arr1() {
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE);
    }
}

class grp1 : public CBase_grp1 {
  int *destBuffer;
  public:
    void grp1() {
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE);
    }
}

class nodegrp1 : public CBase_nodegrp1 {
  int *destBuffer;
  public:
    void nodegrp1() {
      destBuffer = new int[SIZE];
      assignValuesToIndex(destBuffer, SIZE);
    }
}

#include "zc_post_modify_size.def.h"
