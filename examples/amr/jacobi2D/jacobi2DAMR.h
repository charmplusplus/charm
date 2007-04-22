#include "amr.h"

class Jacobi2DAMR:public AmrUserData {
 private:
  int cellSize;
  double **dataGrid;
  double **newDataGrid;
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
  Jacobi2DAMR() {
    cellSize = 64;
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
  virtual void doComputation(void);
  virtual void **fragmentNborData(void* data, int* sizePtr);
  virtual void **getNborMsgArray(int *sizeptr);
  virtual void store(void* data, int dataSize, int neighborSide);
  virtual void combineAndStore(void *data1, void *data2, int dataSize,int neighborSide);
  virtual bool refineCriterion(void);
  virtual void **fragmentForRefine(int *sizePtr);
  ~Jacobi2DAMR() {
    delete newDataGrid;
    delete dataGrid;
  }
};

class main : public Chare {
 public:
  main(CkArgMsg* args) {
    CProxy_AmrCoordinator::ckNew( new _DMsg);
  }
};
