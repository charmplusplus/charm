
#include "amr.h"
//#include "GreedyLB.h"
#define LB_FREQUENCY 15

class Jacobi2DAMR:public AmrUserData {
 private:
  int cellSize;
  double **dataGrid;
  double **newDataGrid;
  /*Utility Functions*/
  void copyGrid(void);
  void copyColumn(double *buf , int colNum) {
    for(int i=1; i<=cellSize;i++)
      buf[i-1] = dataGrid[i][colNum];   
  }
  
  void copyRow(double *buf , int rowNum) {
    for(int i=1; i<=cellSize;i++)
      buf[i-1] = dataGrid[rowNum][i];   
  }

  void copyToColumn(double *buf , int colNum) {
    for(int i=1; i<=cellSize;i++)
      dataGrid[i][colNum]=  buf[i-1];   
  }
  
  void copyToRow(double *buf , int rowNum) {
    for(int i=1; i<=cellSize;i++)
      dataGrid[rowNum][i]= buf[i-1];   
  }
  double sumofGrid(void) {
    double sum = 0.0;
    for(int i=1;i<cellSize+1;i++)
      for(int j=1;j<cellSize+1;j++)
	sum += dataGrid[i][j];
    return sum;
  }

 public:
  /*Default Constructor: Called in the initial setup of the tree*/
  Jacobi2DAMR() {
    cellSize = 32;
    dataGrid = new double* [cellSize+2];
    newDataGrid = new double* [cellSize+2];
    for(int i=0;i< cellSize+2;i++) {
      dataGrid[i] = new double [cellSize+2];
      newDataGrid[i] = new double [cellSize +2];
      for(int k = 0;  k < cellSize+2; k++) {
	newDataGrid[i][k]=10.0;
	dataGrid[i][k] = (i+k) *1.0;
      }
    }
  }

  Jacobi2DAMR(int size) {
    cellSize = size;
    dataGrid = new double* [cellSize+2];
    newDataGrid = new double* [cellSize+2];
    for(int i=0;i< cellSize+2;i++) {
      dataGrid[i] = new double [cellSize+2];
      newDataGrid[i] = new double [cellSize +2];
      for(int k = 0; k < cellSize+2; k++) {
	newDataGrid[i][k]= 10.0;
	dataGrid[i][k] = (i+k) *1.0;



      }
    }
  }

  /*This constructor is called after refinement with data from te parent*/
  Jacobi2DAMR(void *data,int dataSize)
  {
    double *indata = (double*) data;
    cellSize = (int) sqrt((double) (dataSize/sizeof(double)));
    //    cellSize = cellSize/sizeof(double);
    dataGrid = new double* [cellSize+2];
    newDataGrid = new double* [cellSize+2];
    for(int i=0;i< cellSize+2;i++) {
      dataGrid[i] = new double [cellSize+2];
      newDataGrid[i] = new double [cellSize +2];
      for(int k = 0;  k < cellSize+2; k++) {
	newDataGrid[i][k]= 10.0;
	if(i== 0 || i == cellSize+1 || k==0 || k== cellSize + 1)
	  dataGrid[i][k] = (i+k) *1.0;
	else
	  dataGrid[i][k] = indata[(i-1) * cellSize + (k-1)];
      }
    }
  
  }

  Jacobi2DAMR(CkMigrateMessage *m): AmrUserData(m){}

  PUPable_decl(Jacobi2DAMR);
  /*Mandatory Library Interface functions*/
  virtual void doComputation(void);
  virtual void **fragmentNborData(void* data, int* sizePtr);
  virtual void **getNborMsgArray(int *sizeptr);
  virtual void store(void* data, int dataSize, int neighborSide);
  virtual void combineAndStore(void **dataArray, int dataSize,int neighborSide);
  virtual bool refineCriterion(void);
  virtual void **fragmentForRefine(int *sizePtr);

  /*If load balancing is required*/
  virtual void pup(PUP::er &p);
  /*Destructor*/
  ~Jacobi2DAMR() {
    for (int i=0; i< cellSize+2;i++)
      delete [] newDataGrid[i];
    delete [] newDataGrid;
    for (int i=0; i< cellSize+2;i++)
      delete [] dataGrid[i];
    delete[] dataGrid;
  }
};

/*Main Chare*/
class main : public Chare {
 public:
  /*Constructor: Library is created from here*/
  main(CkArgMsg* args) {

    StartUpMsg *msg;
    msg = new StartUpMsg;
    msg->synchInterval = 200;
    msg->depth = 2;
    msg->dimension = 2;
    msg-> totalIterations = 500;
    //    CreateGreedyLB();
    CProxy_AmrCoordinator::ckNew(msg,0);

  }
};
