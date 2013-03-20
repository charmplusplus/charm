#include "test.h"

// routine that updates the length of the work done by the object
#define MYROUTINE(xxx)    xxx==4000 ? 1000 : 4000
//#define MYROUTINE(xxx)    xxx

#define MAX_ITER  40

Main::Main(CkArgMsg *m) { // entry point of the program

  el = 12;
  //if (m->argc > 1) el = atoi(m->argv[1]);
  delete m;

  mainhandle=thishandle;
  called = 0;

  CProxy_MyArray arr = CProxy_MyArray::ckNew(el);
  arr.compute();
}

void Main::allDone() {
  if (++called = el) {
    CkExit();
  }
}

MyArray::MyArray() {
  length = 5000;
  iterations = 0;
  usesAtSync = true;
  for (int i=0; i<length; ++i) {
    data1[i] = rand();
    data2[i] = rand();
  }
}

void MyArray::compute() {
  CkPrintf("[%d] computing iteration %d, length %d\n",thisIndex,++iterations,length);
  for (int j=0; j<2000; ++j) {
  for (int i=0; i<length; ++i) {
    a[i] = data1[i] * data2[i];
    b[i] = data1[i] / data2[i];
  }
  }
  AtSync();
}

void MyArray::ResumeFromSync(void) {
  length = MYROUTINE(length);

  // exceptions for the Predictor
  if (iterations == 1) LBTurnPredictorOff();
  if (iterations == 3) LBTurnPredictorOn(0, 25);
  if (iterations == 35) LBChangePredictor(new DefaultFunction());
  if (iterations == 39) {
    LBTurnPredictorOff();
    LBTurnPredictorOn(0,15);
  }

  if (iterations < MAX_ITER) thisProxy[thisIndex].compute();
  else mainhandle.allDone();
};

void MyArray::pup(PUP::er &p) {
  p|length;
  p|iterations;
}

#include "test.def.h"
