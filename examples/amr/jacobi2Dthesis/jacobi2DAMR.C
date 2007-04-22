#include "jacobi2DAMR.h"
#include "jacobi2DAMR.decl.h"
/*
********************************************
User Code for 2D
********************************************
*/
void AmrUserDataJacobiInit(void)
{
  PUPable_reg(AmrUserData);
  PUPable_reg(Jacobi2DAMR);
}

AmrUserData* AmrUserData :: createData()
{
  Jacobi2DAMR *instance = new Jacobi2DAMR;
  return (AmrUserData *)instance;
}
  
AmrUserData* AmrUserData :: createData(void *data, int dataSize)
{
   Jacobi2DAMR *instance = new Jacobi2DAMR(data , dataSize);
   return (AmrUserData *) instance;
}

void AmrUserData :: deleteNborData(void* data)
{
  delete [](double *) data;
}

void AmrUserData :: deleteChildData(void* data)
{
  delete [](double *) data;
}

void Jacobi2DAMR :: doComputation(void)
{
  for(int i=1; i <= cellSize ;i++) 
    for(int j=1; j<=cellSize;j++) 
      newDataGrid[i][j] = 0.2 * (dataGrid[i][j-1] + dataGrid[i][j+1] 
         			 +dataGrid[i][j] +dataGrid[i-1][j] +
				  dataGrid[i+1][j]);
  copyGrid();
  
}

void Jacobi2DAMR :: copyGrid(void) 
{
  for(int i=0; i< cellSize+2; i++)
    for(int j=0; j<cellSize+2; j++){
      dataGrid[i][j] = newDataGrid[i][j];
      newDataGrid[i][j] = 10.0;
    }
}

void ** Jacobi2DAMR :: getNborMsgArray(int* sizePtr)
{
  //gives the size of each individual column in bytes
  *sizePtr = cellSize* sizeof(double);
  //since we are using 2D mesh so have an array of size 4
  double ** dataArray = new double* [4];
  for(int i =0;i<4;i++) {
    dataArray[i] = new double[cellSize];
  }
  //To my Right neighbor
  copyColumn(dataArray[0], cellSize);
  //To my left neighbor
  copyColumn(dataArray[1], 1);
  //To my Down neighbor
  copyRow(dataArray[2], cellSize);
  //To my Up neighbor
  copyRow(dataArray[3], 1);
  return (void **) dataArray;
}

void **Jacobi2DAMR :: fragmentNborData(void *data, int* sizePtr)
{
  int elements = (*sizePtr)/sizeof(double);
  int newElements = elements/2;
  double **fragmentedArray = new double* [2];
  double *indata = (double *)data;
  if(elements %2 == 0){
    *sizePtr = newElements * sizeof(double);
    for(int i=0; i<2; i++) {
      fragmentedArray[i] = new double[newElements];
      for(int j=0; j<newElements;j++)
	fragmentedArray[i][j] = indata[i*newElements + j];
    }
  }
  else {
    *sizePtr =( ++newElements)*sizeof(double);
    for(int i=0; i<2; i++) {
      fragmentedArray[i] = new double[newElements];
      for(int j=0; j<newElements-1;j++)
	fragmentedArray[i][j] = indata[i*newElements + j];
    }
    fragmentedArray[1][newElements-1] = indata[elements -1];
    fragmentedArray[0][newElements-1] = (fragmentedArray[0][newElements -2] 
					+ fragmentedArray[1][0])/2;
  }
  return (void **)fragmentedArray;
    
}

void Jacobi2DAMR :: store(void* data , int dataSize , int neighborSide)
{
  if(dataSize/sizeof(double) == cellSize) {
    switch(neighborSide) {
    case 0:
      copyToColumn((double*) data, 0);
      break;
    case 1:
      copyToColumn((double *) data, cellSize+1);
      break;
    case 2:
      copyToRow((double *) data, 0);
      break;
    case 3:
      copyToRow((double *) data, cellSize+1);
      break;
    }
  }
  //  else
  //    CkError("Error: Jacobi::store...wrong sized message size %d size/sizeof(double) %d cellsize %d\n",
  //	    dataSize,dataSize/sizeof(double), cellSize);
}

void Jacobi2DAMR :: combineAndStore(void **dataArray, 
					int dataSize,
					int neighborSide) {
  int size = dataSize /sizeof(double);
  double * buf = new double[2*size];
  double *tmpbuf = buf + size;
  memcpy((void *)buf, dataArray[0], dataSize);
  memcpy((void *)tmpbuf, dataArray[1], dataSize);
  store((void *)buf,(2*dataSize),neighborSide);
  delete []buf;
}

bool Jacobi2DAMR :: refineCriterion(void) 
{
  double average = sumofGrid()/(cellSize*cellSize);
  if(average < 15.0 && cellSize >= 4)
    return true;
  else
    return false;
}

void** Jacobi2DAMR :: fragmentForRefine(int *sizePtr)
{
  int newSize = cellSize/2;
  *sizePtr = newSize*newSize*sizeof(double);

  double ** dataArray = new double* [4];
  for(int i=0;i<4;i++) {
    dataArray[i] = new double[newSize*newSize];
    for(int j=1;j<=newSize;j++){
      for(int k=1;k<=newSize;k++)
	dataArray[i][(j-1)*newSize+(k-1)] = 
		dataGrid[((i/2)%2)*newSize+j][(i%2)*newSize+k];
    }
  }
  return (void **)dataArray;
  
}

void Jacobi2DAMR::pup(PUP::er &p)
{
  AmrUserData::pup(p);
  p(cellSize);

  //have to pup dataGrid and newDataGrid here
  if(p.isUnpacking()) {
    dataGrid = new double*[cellSize+2];
    newDataGrid = new double*[cellSize+2];
    for(int i=0;i<cellSize+2;i++) {
      dataGrid[i] = new double[cellSize+2];
      newDataGrid[i] = new double[cellSize+2];
    }
  }
  for(int i=0; i<cellSize+2;i++) {
    p(dataGrid[i],cellSize+2);
    p(newDataGrid[i],cellSize+2);
  }
}

PUPable_def(AmrUserData);
PUPable_def(Jacobi2DAMR);

#include "jacobi2DAMR.def.h"
