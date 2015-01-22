#ifndef __MATMUL_TEST_H
#define __MATMUL_TEST_H

#include "vectorAddConsts.h"

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
 private:
  float *A;
  float *B;
  float *C;

 public:
  Workers();
  ~Workers();
  Workers(CkMigrateMessage *msg);
  void beginWork();
  void complete();
};

void randomInit(float *data, int size);

#endif
