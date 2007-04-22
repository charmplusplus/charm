
#include "amr.h"
#include "GreedyCommLB.h"
#include "GreedyLB.h"
#include "RefineLB.h"
#include "cfd.h"

#define LB_FREQUENCY 15

class CfdProblem:public AmrUserData {
 private:
  int gridW, gridH;
  CfdGrid *grid;
   
  //Split physics
  int stepCount;
 public:
  CfdProblem() {
    stepCount =0;
    Properties* properties = initParam();
    
    grid=new CfdGrid(gridW,gridH,properties);
    initGrid();
  }

  CfdProblem(void* data, int dataSize) {
    
    Properties* properties = initParam();
    grid=new CfdGrid(gridW,gridH,properties);
    stepCount = readParentGrid(data, dataSize);
  }
  
  void init() {
    initBoundary();
    if(stepCount == 0)
      createInitialPressure();
  }
  
  void createInitialPressure() {
    if (myIndex.vec[0] == 0 && myIndex.vec[1] == 0 ) {
      for (int y=0;y<gridH;y++)
	for (int x=0;x<gridW;x++) 
	  grid->at(x,y)->P+=20.0e3;
    }
  }
  Properties* initParam();
  void initGrid();
  void initBoundary();
  int readParentGrid(void* data, int dataSize);
  void startStep();
  void finishStep(bool resample);
  void setBoundaries();
  
  CfdProblem(CkMigrateMessage *m): AmrUserData(m){}
  PUPable_decl(CfdProblem);
  virtual void doComputation(void);
  virtual void **fragmentNborData(void* data, int* sizePtr);
  virtual void **getNborMsgArray(int *sizeptr);
  virtual void store(void* data, int dataSize, int neighborSide);
  virtual void combineAndStore(void **dataArray, int dataSize,int neighborSide);
  virtual bool refineCriterion(void);
  virtual void **fragmentForRefine(int *sizePtr);
  virtual void pup(PUP::er &p);
  ~CfdProblem() {
    delete grid;
  }
};




class main : public Chare {
 public:
  main(CkArgMsg* args) {

    StartUpMsg *msg;
    msg = new StartUpMsg;
    msg->synchInterval = 100;
    msg->depth = 4;
    msg->dimension = 2;
    msg->totalIterations = 399;
    msg->statCollection = 0;
    //CreateCommLB();
    //CreateHeapCentLB();
    CreateRefineLB();
    CProxy_AmrCoordinator::ckNew(msg,0);

  }
};
