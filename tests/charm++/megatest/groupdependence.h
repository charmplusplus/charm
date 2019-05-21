#include "groupdependence.decl.h"
#include "megatest.h"

readonly<CProxy_tester> testObj;

void groupdependence_moduleinit(void) {
  // Create the chare to drive the testing for group dependence
  testObj = CProxy_tester::ckNew(0);
}

void groupdependence_init(void) {
  ((CProxy_tester)testObj).next_test();
}

// Group on which other classes have a dependence on
class groupA : public CBase_groupA {
  public:
  bool called;
  groupA() {
    called = true;
  }
};

// Test Group with dependence on GroupA
class groupB : public CBase_groupB {
  public:
  bool called;
  groupB() {
    called = true;
  }

  groupB(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Group B called before Group A creation \n");
    called = true;
    ((CProxy_tester)testObj).complete_test(CkNumPes());
  }

  void test_em_regular(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Group B's regular entry method called before Group A creation \n");
    ((CProxy_tester)testObj).complete_test(CkNumPes());
  }

  void test_em_inline(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Group B's inline entry method called before Group A creation \n");
    ((CProxy_tester)testObj).complete_test(CkNumPes());
  }
};

// Test Chare with dependence on GroupA
class chareA : public CBase_chareA {
  public:
  chareA() {}

  chareA(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Chare called before Group A \n");
    ((CProxy_tester)testObj).complete_test(1);
  }

  void test_em_regular(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Chare's regular entry method called before Group A \n");
    ((CProxy_tester)testObj).complete_test(1);
  }

  void test_em_inline(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Chare's inline entry method called before Group A \n");
    ((CProxy_tester)testObj).complete_test(1);
  }
};

// Test Chare Array with dependence on GroupA
class arrayA : public CBase_arrayA {
  int arrSize;
  public:
  arrayA(int size) {
    arrSize = size;
  }

  arrayA(int size, CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Chare Array called before Group A creation \n");
    arrSize = size;
    ((CProxy_tester)testObj).complete_test(size);
  }

  void test_em_regular(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Chare Array's regular entry method called before Group A creation\n");
    ((CProxy_tester)testObj).complete_test(arrSize);
  }

  void test_em_inline(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("Chare Array's inline entry method called before Group A \n");
    ((CProxy_tester)testObj).complete_test(arrSize);
  }
};

// Test NodeGroup with dependence with GroupA
class nodeGroupA : public CBase_nodeGroupA {
  public:
  nodeGroupA() {}

  nodeGroupA(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("NodeGroup called before Group A \n");
    ((CProxy_tester)testObj).complete_test(CkNumNodes());
  }

  void test_em_regular(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("NodeGroup's regular entry method called before Group A \n");
    ((CProxy_tester)testObj).complete_test(CkNumNodes());
  }

  void test_em_inline(CProxy_groupA dependGroup) {
    groupA *groupALocal = dependGroup.ckLocalBranch();
    if(groupALocal->called == false)
      CkAbort("NodeGroup's inline entry method called before Group A \n");
    ((CProxy_tester)testObj).complete_test(CkNumNodes());
  }
};

// Test fixed sized message with dependence on Group A
class fixedMessage : public CMessage_fixedMessage {
  public:
  int a;
  double b;
  CProxy_groupA aProxy;
  CProxy_groupB bProxy;
  fixedMessage(CProxy_groupA _aProxy, CProxy_groupB _bProxy) {
    aProxy = _aProxy;
    bProxy = _bProxy;
    a = 20;
    b = 12.88;
  }
};

// Test variable sized message with dependence on Group A
class varMessage : public CMessage_varMessage {
  public:
  int *a;
  double *b;
  CProxy_groupA aProxy;
  CProxy_groupB bProxy;
  varMessage(CProxy_groupA _aProxy, CProxy_groupB _bProxy) {
    aProxy = _aProxy;
    bProxy = _bProxy;

    a = new int[20];
    b = new double[10];

    for(int i=0;i<20;i++) {
      a[i]=i;
      if(i<10)
        b[i]= i*1.3;
    }
  }
};
