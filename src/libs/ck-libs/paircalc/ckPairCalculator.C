#include "ckPairCalculator.h"

#define PARTITION_SIZE 500

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
  outData = new double[grainSize * grainSize];
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
    outData = new double[grainSize * grainSize];
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

  // once have K left and K right (or just K left if we're diagonal
  // and symmetric) compute ZDOT for the inputs.

  // Because the vectors are not guaranteed contiguous, record each
  // offset so we can iterate through them
  /*
  if(kLefttCount > kUnits && ((kRightCount > kUnits) || (symmetric && thisIndex.x == thisIndex.y)))
    {

      // compute

      // count down kUnits from leftCount starting at kUnit'th element
      for(int i=kUnits;i>0;i--)
	{
	}

      kLeftCount-=kUnits;
      memcpy(kLeftOffset,kLeftOffset+kUnits,kLeftCount);
      if(!symmetric || thisIndex.x != thisIndex.y)
	{
	  kRightCount-=kUnits;
	  memcpy(kRighttOffset,kRightOffset+kUnits,kRightCount);
	}
    }

  */
  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
  
#ifdef _DEBUG_
    CkPrintf("     pairCalc[%d %d %d %d] got expected %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  numExpected);
#endif
    numRecd = 0;
    int i, j, idxOffset;

    if(symmetric && thisIndex.x == thisIndex.y) {
        int size1 = size%PARTITION_SIZE;
        if(size1 > 0) {
	    int start_offset = size-size1;
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] = compute_entry(size1, inDataLeft[i]+start_offset,
                                                               inDataLeft[j]+start_offset, op1);        
        }
        for(size1 = 0; size1 + PARTITION_SIZE < size; size1 += PARTITION_SIZE) {
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] += compute_entry(PARTITION_SIZE, inDataLeft[i]+size1,
                                                                inDataLeft[j]+size1, op1);
        }        
    }     
    else {                                                        
      // compute a square region of the matrix. The correct part of the
      // region will be used by the reduction.
        int size1 = size%PARTITION_SIZE;
        if(size1 > 0) {
	    int start_offset = size-size1;
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] = compute_entry(size1, inDataLeft[i]+start_offset,
                                                               inDataRight[j]+start_offset, op1);        
        }
        for(size1 = 0; size1 + PARTITION_SIZE < size; size1 += PARTITION_SIZE) {
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] += compute_entry(PARTITION_SIZE, inDataLeft[i]+size1,
                                                               inDataRight[j]+size1, op1);      
        }
    }
#if 0
    if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data
	for (i = 0; i < grainSize*grainSize; i++)
	    outData[i] *= 2.0;
    }
#endif 

    // FIXME: should do 'op2' here!!!

   r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), (CkTwoDoubles*)outData);
    r.contribute(this, sparse_sum_double);
  }
}

void
PairCalculator::acceptEntireResult(int size, double *matrix){
  acceptEntireResult(size, matrix, cb);
}

void
PairCalculator::acceptEntireResult(int size, double *matrix, CkCallback cb)
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
PairCalculator::acceptResult(int size, double *matrix, int rowNum, CkCallback cb)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: Accept Result with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif
  complex *mynewData = new complex[N*grainSize];
  memset(mynewData, 0, sizeof(complex)*N*grainSize);

  int offset = 0, index = thisIndex.y*S + thisIndex.x;

  //ASSUMING TMATRIX IS REAL (LOSS OF GENERALITY)
  register double m=0;  
  //  complex zero=complex(0,0);  

/*  int size1 = 0;
  for(size1 = 0; size1 + PARTITION_SIZE < N; size1 += PARTITION_SIZE) {
    for (int i = 0; i < grainSize; i++) {
      int iSindex=i*S+index;
      int iN=i*N;
      complex *newiNdata=&mynewData[iN];
      for (int j = 0; j < grainSize; j++){ 
	//if(matrix[iSindex + j].notzero())
	//  {
	m = matrix[iSindex + j];
	for (int p = size1; p < size1+PARTITION_SIZE; p++)
	  //if(inDataLeft[j][p].notzero())
	  newiNdata[p] += inDataLeft[j][p] * m;
	//}
      }
    }
  }
  if(size1 > N) {
    int start_offset = N-size1;
    for (int i = 0; i < grainSize; i++) {
      int iSindex=i*S+index;
      int iN=i*N;
      complex *newiNdata=&mynewData[iN];
      for (int j = 0; j < grainSize; j++){ 
	m = matrix[iSindex + j];
	for (int p = start_offset; p < N; p++)
	  //if(inDataLeft[j][p].notzero())
	  newiNdata[p] += inDataLeft[j][p] * m;
      }
    }
  }
*/

  //  this one reads better but may take longer
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[index + j + i*S];
      for (int p = 0; p < N; p++)
	mynewData[p + i*N] += inDataLeft[j][p] * m;
    }
  }

  /* revise this to partition the data into S/M objects 
   * add new message and entry method for sumPartial result
   * to avoid message copying.
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
  /*
  // commenting the reduction split out since it doesn't work yet.

  int segments=S/grainSize;
  if(S%grainSize!=0)
    segments+=1;
  int blocksize=grainSize/segments;
  int priority=0xFFFFFFFF;
  for(int segment=0;segment < segments;segment++)
    {  
      //      CkPrintf("[%d %d %d %d]: sending N %d segment %d of %d segments \n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z, N, segment,segments);
      CkArrayIndexIndex4D idx(thisIndex.w,segment*grainSize, thisIndex.y, thisIndex.z);

      if(!symmetric){

	partialResultMsg *msg = new (N*blocksize,8*sizeof(int))partialResultMsg(priority,N*blocksize, cb, mynewData+segment*N*blocksize);
	*((int*)CkPriorityPtr(msg)) = priority;
	CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	thisProxy(idx).sumPartialResult(msg);  
      }
      else {
	partialResultMsg *msg = new (N*blocksize,8*sizeof(int))partialResultMsg(priority,N*blocksize,  cb, mynewData+segment*N*blocksize);
	*((int*)CkPriorityPtr(msg)) = priority;
	CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	thisProxy(idx).sumPartialResult(msg);  
	if (rowNum != thisIndex.x){
	  partialResultMsg *msg = new (N*blocksize,8*sizeof(int))partialResultMsg(priority,N*blocksize, cb, mynewData+segment*N*blocksize);
	  *((int*)CkPriorityPtr(msg)) = priority;
	  CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	  thisProxy(idx).sumPartialResult(msg);  
	}
      }
    }
  */
  delete [] mynewData;
}

void 
PairCalculator::sumPartialResult(partialResultMsg *msg)
{

  //  CkPrintf("[%d %d %d %d]: sum result from grain %d  count %d\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,msg->grain, sumPartialCount);

  int segments=S/grainSize;
  if(S%grainSize!=0)
    segments+=1;
  segments*=blkSize;
  int psumblocksize=grainSize/segments;

  sumPartialCount++;

  if(!newData){
    newData = new complex[N*psumblocksize];
  }  
  for(int i=0; i<N*psumblocksize; i++){
    newData[i] += msg->result[i];  // should be adjusted with offset
  }
  if (sumPartialCount == segments) {
    //    CkPrintf("[%d %d %d %d]: sumPartialCount %d is grainSize %d segments %d, psumblocksize %d, calling back\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,sumPartialCount,grainSize,segments,psumblocksize);
    for(int i=0;i<psumblocksize;i++)
      {
	//	CkPrintf("N %d i %d \n",N,i);
	CkCallback mycb(msg->cb.d.array.ep, CkArrayIndex2D(thisIndex.y/grainSize+i+thisIndex.x/grainSize*psumblocksize, thisIndex.w), msg->cb.d.array.id);
	mySendMsg *outmsg = new (N*psumblocksize,0)mySendMsg(N*psumblocksize,  newData+i*psumblocksize); // msg with newData (size N)
	mycb.send(outmsg);
      }
    sumPartialCount = 0;
    memset(newData,0,N*psumblocksize);
  }
  delete msg;
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
    memset(newData,0,N*grainSize);
    //    for(int k=0; k<N*grainSize; k++)
    //	 newData[k] = complex(0,0);
  }
}


void
PairCalcReducer::acceptPartialResult(int size, complex* matrix, int fromRow, int fromCol, CkCallback cb){
  

}

void
PairCalcReducer::broadcastEntireResult(int size, double* matrix, bool symmtype, CkCallback cb){
  for (int i = 0; i < localElements[symmtype].length(); i++)
    (localElements[symmtype])[i]->acceptEntireResult(size, matrix, cb); 
}

void
PairCalcReducer:: doRegister(PairCalculator *elem, bool symmtype){
    localElements[symmtype].push_back(elem);
    numRegistered[symmtype]++;
}

#include "ckPairCalculator.def.h"
