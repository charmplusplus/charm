#ifndef __MATMUL_TEST_H
#define __MATMUL_TEST_H

#include "multiply.h"

class Main : public CBase_Main {
 private:
  CProxy_Workers workers;
  int numChares;
  double startTime;

 public:
  Main(CkArgMsg *m);
  void finishWork(CkReductionMsg *m);
};


class Workers: public CBase_Workers {
 public:
  float *A;
  float *B;
  float *C;

  Workers();
  ~Workers();
  Workers(CkMigrateMessage *msg);
  void beginWork(int size);
  void complete();
};

void randomInit(float *data, int size);

#endif
