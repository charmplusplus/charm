#include "ckPairCalculator.h"

PairCalculator::PairCalculator(CkMigrateMessage *m) { }
	
PairCalculator::PairCalculator(bool sym, int grainSize, int s, int blkSize,  int op1,  FuncType fn1, int op2,  FuncType fn2, CkCallback cb, CkGroupID gid) 
{
  this->symmetric = sym;
  this->grainSize = grainSize;
  this->S = s;
  this->blkSize = blkSize;
  this->op1 = op1; 
  this->fn1 = fn1; 
  this->op2 = op2; 
  this->fn2 = fn2; 
  this->cb = cb;
  this->N = -1;
  numRecd = 0;
  numExpected = grainSize;
  
  inDataLeft = new complex*[numExpected];
  for (int i = 0; i < numExpected; i++)
    inDataLeft[i] = NULL;
  inDataRight = NULL;
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)) {
    inDataRight = new complex*[numExpected];
    for (int i = 0; i < numExpected; i++)
      inDataRight[i] = NULL;
  }
  outData = new complex[grainSize * grainSize];
  newData = NULL;
  sumPartialCount = 0;
  setMigratable(false);

  CProxy_PairCalcReducer pairCalcReducerProxy(gid); 
  pairCalcReducerProxy.ckLocalBranch()->doRegister(this, symmetric);

#ifdef _DEBUG_
  CkPrintf("     pairCalc [%d %d %d %d] inited\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
#endif
}

void
PairCalculator::pup(PUP::er &p)
{
  ArrayElement4D::pup(p);
  p|numRecd;
  p|grainSize;
  p|numExpected;
  p|S;
  p|blkSize;
  p|op1;
  p|op2;
  if (p.isUnpacking()) {
    outData = new complex[grainSize * grainSize];
    inDataLeft = new complex*[numExpected];
    for (int i = 0; i < numExpected; i++)
      inDataLeft[i] = new complex[N];
    if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
      inDataLeft = new complex*[numExpected];
      for (int i = 0; i < numExpected; i++)
	inDataLeft[i] = new complex[N];
    }
  }
  for (int i = 0; i < numExpected; i++)
    p(inDataLeft[i], N);
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
    for (int i = 0; i < numExpected; i++)
      p(inDataRight[i], N);
  }
}

PairCalculator::~PairCalculator()
{
  delete [] outData;
  for (int i = 0; i < numExpected; i++)
    delete [] inDataLeft[i];
  delete [] inDataLeft;
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
    for (int i = 0; i < numExpected; i++)
      delete [] inDataRight[i];
    delete [] inDataRight;
  }  
  if(!newData)
    delete [] newData;
}

void
PairCalculator::calculatePairs(int size, complex *points, int sender, bool fromRow)
{
#ifdef _DEBUG_
  CkPrintf("     pairCalc[%d %d %d %d] got from [%d %d] with size {%d}, symm=%d, from=%d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  thisIndex.w, sender, size, symmetric, fromRow);
#endif
  int offset = -1;
  complex **inData;

  if (fromRow) {
    offset = sender - thisIndex.x;
    inData = inDataLeft;
  }
  else {
    offset = sender - thisIndex.y;
    inData = inDataRight;
  }
  if(symmetric && thisIndex.x == thisIndex.y){
    inData = inDataLeft;
  }
  N = size;                                                             

  if (!inData[offset]) 
    inData[offset] = new complex[size];
  memcpy(inData[offset], points, size * sizeof(complex));

  numRecd++; 
  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
  
#ifdef _DEBUG_
    CkPrintf("     pairCalc[%d %d %d %d] got expected %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  numExpected);
#endif
    numRecd = 0;
    int i, j, idxOffset;
    if(symmetric && thisIndex.x == thisIndex.y) {
      for (i = 0; i < grainSize; i++)
	for (j = 0; j < grainSize; j++) 
	  outData[i * grainSize + j] = compute_entry(size, inDataLeft[i],
						     inDataLeft[j], op1);
    }     
    else {                                                        
      // compute a square region of the matrix. The correct part of the
      // region will be used by the reduction.
      for (i = 0; i < grainSize; i++)
	for (j = 0; j < grainSize; j++) 
	  outData[i * grainSize + j] = compute_entry(size, inDataLeft[i],
						     inDataRight[j], op1);
    }

    // FIXME: should do 'op2' here!!!

   r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), (CkTwoDoubles*)outData);
    r.contribute(this, sparse_sum_TwoDoubles);

  }
}

void
PairCalculator::acceptEntireResult(int size, complex *matrix){
  acceptEntireResult(size, matrix, cb);
}

void
PairCalculator::acceptEntireResult(int size, complex *matrix, CkCallback cb)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: Accept EntireResult with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif
  CkArrayIndexIndex4D myidx(thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
  acceptResult(size, matrix, thisIndex.x, cb);
  if(symmetric && thisIndex.x != thisIndex.y)
    acceptResult(size, matrix, thisIndex.y, cb);
}


void
PairCalculator::acceptResult(int size, complex *matrix, int rowNum, CkCallback cb)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: Accept Result with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif
  complex *mynewData = new complex[N*grainSize];
  memset(mynewData, 0, sizeof(complex)*N*grainSize);

  int offset = 0, index = thisIndex.y*S + thisIndex.x;
  complex m=complex(0,0);  
  complex zero=complex(0,0);  

  for (int i = 0; i < grainSize; i++) {
    int iSindex=i*S+index;
    int iN=i*N;
    complex *newiNdata=&mynewData[iN];
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[iSindex + j];
      if(!(m==zero))
	for (int p = 0; p < N; p++)
	  if(!(inDataLeft[j][p]==zero))
	    newiNdata[p] += inDataLeft[j][p] * m;
    }
  }
  /*
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[index + j + i*S];
      for (int p = 0; p < N; p++)
	mynewData[p + i*N] += inDataLeft[j][p] * m;
    }
  }
  */
  if(!symmetric){
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z, cb);
  }
  else {
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z, cb);  
    if (rowNum != thisIndex.x){
      CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z, cb);  
    }
  }
  
  delete [] mynewData;
}

void 
PairCalculator::sumPartialResult(int size, complex *result, int offset, CkCallback cb)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: sum result from %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, offset);
#endif

  sumPartialCount++;

  if(!newData){
    newData = new complex[N*grainSize];
  }  
  for(int i=0; i<N*grainSize; i++){
    newData[i] += result[i];  // should be adjusted with offset
  }
  if (sumPartialCount == (S/grainSize)*blkSize) {
    for(int j=0; j<grainSize; j++){
      CkCallback mycb(cb.d.array.ep, CkArrayIndex2D(thisIndex.y+j, thisIndex.w), cb.d.array.id);
      mySendMsg *msg = new (N, 0)mySendMsg(N, newData+j*N); // msg with newData (size N)
      mycb.send(msg);
    }
    sumPartialCount = 0;
    for(int k=0; k<N*grainSize; k++){
      newData[k] = complex(0,0);
    }
  }
}

complex
PairCalculator::compute_entry(int n, complex *psi1, complex *psi2, int op)
{
  // FIXME: should do 'op1' here!!!
  double re=0, im = 0;
  double *ptr1 = (double*)psi1;
  double *ptr2 = (double*)psi2;
  for (int i = 0; i < 2*n; i+=2){
    re += ptr1[i]*ptr2[i] - ptr1[i+1]*ptr2[i+1];
    im += ptr1[i+1]*ptr2[i] + ptr1[i]*ptr2[i+1];
  }
  complex sum(re,im);
  return sum;
}


void
PairCalcReducer::acceptPartialResult(int size, complex* matrix, int fromRow, int fromCol, CkCallback cb){
  

}

void
PairCalcReducer::broadcastEntireResult(int size, complex* matrix, bool symmtype, CkCallback cb){
  for (int i = 0; i < localElements[symmtype].length(); i++)
    (localElements[symmtype])[i]->acceptEntireResult(size, matrix, cb); 
}

void
PairCalcReducer:: doRegister(PairCalculator *elem, bool symmtype){
    localElements[symmtype].push_back(elem);
    numRegistered[symmtype]++;
}

#include "ckPairCalculator.def.h"
