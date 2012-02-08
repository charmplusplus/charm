#include "test.decl.h"

CProxy_Main mainhandle;

class Main : public CBase_Main {
  int el, called;
 public:
  Main(CkArgMsg *m);
  void allDone();
};

class MyArray : public CBase_MyArray {
  double data1[10000];
  double data2[10000];
  double a[10000];
  double b[10000];
  int length;
  int iterations;
 public:
  MyArray();
  MyArray(CkMigrateMessage *m) {}
  void ResumeFromSync(void);
  void compute();
  void pup(PUP::er&);
};
