#include <stdlib.h>
#include "parallelJacobi.decl.h"

#define ITER    100
#define WORK     1

     
class Main: public CBase_Main{
  
 public: 
  int numElements, numFinished;
  Main(CkArgMsg *m);
  void finished();
};

class Msg: public CMessage_Msg{

 public:
  int dim;
  double* strip;

  Msg(int n, double *inputStrip):dim(n){
    memcpy(strip,inputStrip,n*(sizeof(double)));
  }

};


class MaxMsg: public CMessage_MaxMsg{
  
 public:
  double max;
 
  MaxMsg(double m){
    max = m; 
  }
};

class VoidMsg: public CMessage_VoidMsg {
  public:
    int intVal;

    VoidMsg(): intVal(-1) { }
    VoidMsg(int val): intVal(val) { }
};

class Chunk: public CBase_Chunk {

Chunk_SDAG_CODE

private:
 int  myxdim, myydim,total,counter,iterations;
  double *A, *B;
  double* temp;
  double myMax;
  void resetBoundary();
  void print();
  void doWork();
  void testEnd();

public:
  Chunk(int,int,int);
  Chunk(CkMigrateMessage *m);
  void startWork();
  void workover(CkReductionMsg* msg);

  void processStripfromright(Msg* aMessage);
  void processStripfromleft(Msg* aMessage);

};


