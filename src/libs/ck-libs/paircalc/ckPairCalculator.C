#include "ckPairCalculator.h"

#define PARTITION_SIZE 500

PairCalculator::PairCalculator(CkMigrateMessage *m) { }
	

PairCalculator::PairCalculator(bool sym, int grainSize, int s, int blkSize,  int op1,  FuncType fn1, int op2,  FuncType fn2, CkCallback cb, CkGroupID gid, CkArrayID cb_aid, int cb_ep, bool conserveMemory) 
{
#ifdef _PAIRCALC_DEBUG_ 
  CkPrintf("[PAIRCALC] [%d %d %d %d] inited\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
#endif 
  this->conserveMemory=conserveMemory;
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
  this->cb_aid = cb_aid;
  this->cb_ep = cb_ep;
  reducer_id = gid;

  numRecd = 0;
  numExpected = grainSize;

  kUnits=5;  //streaming unit only really used in NOGEMM, but could be used under other conditions

#ifdef NOGEMM  
  kLeftOffset= new int[numExpected];
  kRightOffset= new int[numExpected];

  kLeftCount=0;
  kRightCount=0;

  kLeftDoneCount = 0;
  kRightDoneCount = 0;
#endif

  inDataLeft = NULL;
  inDataRight = NULL;

#ifdef _SPARSECONT_ 
  outData = new double[grainSize * grainSize];
  memset(outData, 0 , sizeof(double)* grainSize * grainSize);
#else
  outData = new double[S * S];
  memset(outData, 0 , sizeof(double)* S *S);
#endif

  newData = NULL;
  sumPartialCount = 0;
  setMigratable(false);

  CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id); 
  pairCalcReducerProxy.ckLocalBranch()->doRegister(this, symmetric);
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
  p|fn1;
  p|fn2;
  p|conserveMemory;
#ifdef NOGEMM
  p|kRightCount;
  p|kLeftCount;
  p|kRightDoneCount;
  p|kLeftDoneCount;
#endif
  p|kUnits;
  p|cb_aid;
  p|cb_ep;
  p|reducer_id;
  p|symmetric;
  p|sumPartialCount;
  p|N;
#ifdef NOGEMM
  int rdiff,ldiff;
#endif
  if(p.isPacking())
    {//store offset calculation
#ifdef NOGEMM      
      rdiff=kRightMark-kRightOffset; 
      ldiff=kLeftMark-kLeftOffset;
      p|rdiff;
      p|ldiff;
#endif
    }
  if (p.isUnpacking()) {
#ifdef _SPARSECONT_ 
    outData = new double[grainSize * grainSize];
#else
    outData = new double[S*S];
#endif

    if(N>0)
      inDataLeft = new complex[numExpected*N];
    if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y))
      if(N>0)
	inDataRight = new complex[numExpected*N];

#ifdef NOGEMM
    kRightOffset= new int[numExpected];
    kLeftOffset= new int[numExpected];
    p|rdiff;
    p|ldiff;
    kRightMark=kRightOffset+rdiff;
    kLeftMark=kLeftOffset+ldiff;
#endif
  }
  if(N>0)
    for (int i = 0; i < numExpected; i++)
      p(inDataLeft,numExpected*N);
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
    if(N>0)  
      p(inDataRight, numExpected* N);
  }
#ifdef NOGEMM  
  p(kRightOffset, numExpected);
  p(kLeftOffset, numExpected);
#endif
  //p(cb);      // PUP the callback function: ???
              // How about sparseCont reducer???

#ifdef _PAIRCALC_DEBUG_ 
  CkPrintf("ckPairCalculatorPUP\n");
#endif
}

PairCalculator::~PairCalculator()
{
  if(outData!=NULL)  
    delete [] outData;

  // Since allocation is done in contiguous chunk, 
  //   deletion is done in corresponding way.
  if(inDataLeft!=NULL)
    delete [] inDataLeft;
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y))
    if(inDataRight!=NULL)
      delete [] inDataRight;

  if(newData!=NULL)
    delete [] newData;
#ifdef NOGEMM
  if(kRightOffset!=NULL)
    delete [] kRightOffset;
  if(kLeftOffset!=NULL)
    delete [] kLeftOffset;
#endif
}


void
PairCalculator::calculatePairs_gemm(int size, complex *points, int sender, bool fromRow, bool flag_dp)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     pairCalc[%d %d %d %d] got from [%d %d] with size {%d}, symm=%d, from=%d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  thisIndex.w, sender, size, symmetric, fromRow);
#endif
  
  numRecd++;   // increment the number of received counts
  complex *inData;
  int offset = -1;
  
  if (fromRow) {   // This could be the symmetric diagonal case
    offset = sender - thisIndex.x;
    if (inDataLeft==NULL) 
      { // now that we know N we can allocate contiguous space
	N = size; // N is init here with the size of the data chunk. 
	inDataLeft = new complex[numExpected*N];
      }
    inData = inDataLeft;
  }
  else {
    offset = sender - thisIndex.y;
    if (inDataRight==NULL) 
      { // now that we know N we can allocate contiguous space
	N = size; // N is init here with the size of the data chunk. 
	inDataRight = new complex[numExpected*N];
      }
    inData= inDataRight;
  }

  CkAssert(N==size);
  /* 
   *  NOTE: For this to work the data chunks of the same plane across
   *  all states must be of the same size
   */

  // copy the input into our matrix
  memcpy(&(inData[offset*N]), points, size * sizeof(complex));

  /*
   * Once we have accumulated all rows  we gemm it.
   * (numExpected X N) X (N X numExpected) = (numExpected X numExpected) 
   */
  /* To make this work, we transpose the first matrix (A). 
     In C++ it appears to be: 
   * (ydima X ydimb) = (ydima X xdima) X (xdimb X ydimb)

   * Which would be wrong, this works because we're using fortran
   * BLAS, which has a transposed perspective (column major), so the
   * actual multiplication is:
   *
   * (xdima X xdimb) = (xdima X ydima) X (ydimb X xdimb)
   *
   * Since xdima==xdimb==numExpected==grainSize this gives us the
   * solution matrix we want in one step.
   */

  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
    char transform='N';
    int doubleN=2*N;
    char transformT='T';
    int m_in=numExpected;
    int n_in=numExpected;
    int k_in=doubleN;
    int lda=doubleN;   //leading dimension A
    int ldb=doubleN;   //leading dimension B
    int ldc=numExpected;   //leading dimension C

    double alpha=double(1.0);//multiplicative identity 
    double beta=double(0.0); // C is unset

    double *ldata= reinterpret_cast <double *> (inDataLeft);

    if( numRecd == numExpected * 2) 
      {
	double *rdata= reinterpret_cast <double *> (inDataRight); 
	DGEMM(&transformT, &transform, &m_in, &n_in, &k_in, &alpha, ldata, &lda, rdata, &ldb, &beta, outData, &ldc);
      }
    else if (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)
      {
	DGEMM(&transformT, &transform, &m_in, &n_in, &k_in, &alpha, ldata, &lda, ldata, &ldb, &beta, outData, &ldc);
      }
    numRecd = 0;

    if (flag_dp) {
      if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data
#ifdef _SPARSECONT_
	for (int i = 0; i < grainSize*grainSize; i++)
	  outData[i] *= 2.0;
#else
	for (int i = 0; i < grainSize; i++)
	  for (int j = 0; j < grainSize; j++)		  
	    outData[(i+thisIndex.y)*S + j + thisIndex.x] *= 2.0; 
#endif
      }
    }
#ifdef _SPARSECONT_
    r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
    r.contribute(this, sparse_sum_double);
#else

#if !CONVERSE_VERSION_ELAN
    contribute(S * S *sizeof(double), outData, CkReduction::sum_double);
#else
    //CkPrintf("[%d] ELAN VERSION %d\n", CkMyPe(), symmetric);
    CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id); 
    pairCalcReducerProxy.ckLocalBranch()->acceptContribute(S * S, outData, 
							   cb, !symmetric, symmetric);
#endif //!CONVERSE_VERSION_ELAN

#endif //_SPARSECONT_
  }

}


/* note this is broken now, offset calculations need to be rejiggered to work with the one dimensional allocation scheme*/
void
PairCalculator::calculatePairs(int size, complex *points, int sender, bool fromRow, bool flag_dp)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     pairCalc[%d %d %d %d] got from [%d %d] with size {%d}, symm=%d, from=%d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  thisIndex.w, sender, size, symmetric, fromRow);
#endif

#ifdef NOGEMM
  numRecd++;   // increment the number of received counts

  int offset = -1;
  complex **inData;
  if (fromRow) {   // This could be the symmetric diagnoal case
    offset = sender - thisIndex.x;
    inData = inDataLeft;
#ifdef _PAIRCALC_FIRSTPHASE_STREAM_
    kLeftOffset[kLeftDoneCount + kLeftCount]=offset;
    kLeftCount++;
#endif
  }
  else {
    offset = sender - thisIndex.y;
    inData = inDataRight;
#ifdef _PAIRCALC_FIRSTPHASE_STREAM_
    kRightOffset[kRightDoneCount + kRightCount]=offset;
    kRightCount++;
#endif
  }

  N = size; // N is init here with the size of the data chunk. 
            // Assuming that data chunk of the same plane across all states are of the same size

  if (inData==NULL) 
  { // now that we know N we can allocate contiguous space
    inData = new complex[numExpected*N];
  }
  memcpy(inData[offset*N], points, size * sizeof(complex));


  // once have K left and K right (or just K left if we're diagonal
  // and symmetric) compute ZDOT for the inputs.

  // Because the vectors are not guaranteed contiguous, record each
  // offset so we can iterate through them


#ifdef _PAIRCALC_FIRSTPHASE_STREAM_

  if((kLeftCount >= kUnits 
      && ((kRightCount >= kUnits) || (symmetric && thisIndex.x == thisIndex.y) )) 
     || (numRecd == numExpected * 2) 
     || ((symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)))
      // if enough submatrix has arrived from both left and right matrixes; 
      // or if enough submatrix has arrived from left and this is diagonal one;
      // or if all submatrix has arrived
    {
      // count down kUnits from leftCount starting at kUnit'th element
      //int i, j, idxOffset;
      int leftoffset, rightoffset;

      // Ready to do  kLeftReady in left matrix, and kRightReady in right matrix.
      // Both in multiples of kUnits
      int kLeftReady = kLeftCount - kLeftCount % kUnits;
      int kRightReady;
      if(! (symmetric && thisIndex.x == thisIndex.y))
	  kRightReady = kRightCount - kRightCount % kUnits;

      if (numRecd == numExpected * 2 
	  || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected))
	{ // if all has arrived, then finish whatever is remained
	  kLeftReady = kLeftCount;
	  if(! (symmetric && thisIndex.x == thisIndex.y))
	      kRightReady = kRightCount;
	}

      if(symmetric && thisIndex.x == thisIndex.y) {
	  // if the symmetric but diagonal case

          // NEW left compute with every entry in left (old+new)
  	  int leftoffset1=0, leftoffset2=0;
	  for(int kLeft1=kLeftDoneCount; kLeft1<kLeftDoneCount + kLeftReady; kLeft1++)
	  {
	    leftoffset1=kLeftOffset[kLeft1];
	    for(int kLeft2=0; kLeft2<kLeftDoneCount + kLeftReady; kLeft2++)
	      {
		  leftoffset2 = kLeftOffset[kLeft2];
		  // if(leftoffset1 <= leftoffset2) {
#ifdef _SPARSECONT_
		  outData[leftoffset1 * grainSize + leftoffset2] = 
		    compute_entry(size, &(inDataLeft[leftoffset1*N]), &(inDataLeft[leftoffset2*N]),op1);   
#else
		  outData[(leftoffset1+thisIndex.y)*S + leftoffset2 + thisIndex.x] = 
		    compute_entry(size, &(inDataLeft[leftoffset1*N]), &(inDataLeft[leftoffset2*N]),op1);   
#endif	 
		  //}
	      }
	  }
      }
      else {                                                        
	// compute a square region of the matrix. The correct part of the
	// region will be used by the reduction.

        // NEW left compute with every entry in right
	for(int kLeft=kLeftDoneCount; kLeft<kLeftDoneCount + kLeftReady; kLeft++)
	  {
	    leftoffset = kLeftOffset[kLeft];
	    for(int kRight=0; kRight<kRightDoneCount + kRightReady; kRight++) 
	      {
	       rightoffset = kRightOffset[kRight];
#ifdef _SPARSECONT_
		outData[leftoffset * grainSize + rightoffset] = 
		    compute_entry(size, &(inDataLeft[leftoffset*N]), &(inDataRight[rightoffset*N]),op1);     

#else
		outData[(leftoffset+thisIndex.y)*S + rightoffset + thisIndex.x] = 
		    compute_entry(size, &(inDataLeft[leftoffset*N]), &(inDataRight[rightoffset*N]),op1);    
#endif	    
	      }
	  }

        // OLD left compute with every NEW entry in right
	for(int kLeft=0; kLeft<kLeftDoneCount; kLeft++)
	  {
	    leftoffset = kLeftOffset[kLeft];
	    for(int kRight=kRightDoneCount; kRight<kRightDoneCount + kRightReady; kRight++) 
	      {
	       rightoffset = kRightOffset[kRight];
#ifdef _SPARSECONT_
		outData[leftoffset * grainSize + rightoffset] = 
		    compute_entry(size, &(inDataLeft[leftoffset*N]), &(inDataRight[rightoffset*N]),op1);       

#else
		outData[(leftoffset+thisIndex.y)*S + rightoffset + thisIndex.x] = 
		    compute_entry(size, &(inDataLeft[leftoffset*N]), &(inDataRight[rightoffset*N]),op1);        
#endif	    
	      }
	  }

      }
      // Decrement the undone session count
      kLeftCount -= kLeftReady;
      if(! (symmetric && thisIndex.x == thisIndex.y))
	  kRightCount -= kRightReady;

      // Increment the done session count
      kLeftDoneCount +=kLeftReady;
      if(! (symmetric && thisIndex.x == thisIndex.y))
	  kRightDoneCount += kRightReady;
    }

  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
      numRecd = 0;
      kLeftCount=0;
      kRightCount=0;
      kLeftDoneCount = 0;
      kRightDoneCount = 0;

      if (flag_dp) {
	  if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data
#ifdef _SPARSECONT_
	      for (int i = 0; i < grainSize*grainSize; i++)
		  outData[i] *= 2.0;
#else
	      for (int i = 0; i < grainSize; i++)
		  for (int j = 0; j < grainSize; j++)		  
		      outData[(i+thisIndex.y)*S + j + thisIndex.x] *= 2.0; 
#endif
	  }
      }
#ifdef _SPARSECONT_
      r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
      r.contribute(this, sparse_sum_double);
#else
#if !CONVERSE_VERSION_ELAN
      contribute(S * S *sizeof(double), outData, CkReduction::sum_double);
#else
      //CkPrintf("[%d] ELAN VERSION %d\n", CkMyPe(), symmetric);
      CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id); 
      pairCalcReducerProxy.ckLocalBranch()->acceptContribute(S * S, outData, 
                                                             cb, !symmetric, symmetric);
#endif
#endif
  }

#else 

// Below is the old version, very dusty
#ifdef _SPARSECONT_ 
  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
    //    kLeftCount=kRightCount=0;  
#ifdef _PAIRCALC_DEBUG_
    CkPrintf("     pairCalc[%d %d %d %d] got expected %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  numExpected);
#endif
    numRecd = 0;   // Reset the received count to zero for next iteration

    int i, j, idxOffset;
    if(symmetric && thisIndex.x == thisIndex.y) {
        int size1 = size%PARTITION_SIZE;
        if(size1 > 0) {
	    int start_offset = size-size1;
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] = compute_entry(size1, &(inDataLeft[i*N])+start_offset,
                                                               &(inDataLeft[j*N])+start_offset, op1);        
        }
        for(size1 = 0; size1 + PARTITION_SIZE < size; size1 += PARTITION_SIZE) {
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] += compute_entry(PARTITION_SIZE, &(inDataLeft[i*N])+size1,
                                                                &(inDataLeft[j*N])+size1, op1);
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
                    outData[i * grainSize + j] = compute_entry(size1, &(inDataLeft[i*N])+start_offset,
                                                               &(inDataRight[j*N])+start_offset, op1);        
        }
        for(size1 = 0; size1 + PARTITION_SIZE < size; size1 += PARTITION_SIZE) {
            for (i = 0; i < grainSize; i++)
                for (j = 0; j < grainSize; j++) 
                    outData[i * grainSize + j] += compute_entry(PARTITION_SIZE, &(inDataLeft[i*N])+size1,
                                                               &(inDataRight[j*N])+size1, op1);      
        }
    }
    if (flag_dp) {
	if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data
	    for (i = 0; i < grainSize*grainSize; i++)
		outData[i] *= 2.0;
	}
    }
    // FIXME: should do 'op2' here!!!

    r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
    r.contribute(this, sparse_sum_double);
  }
#endif
#endif

#endif //NOGEMM
}

void
PairCalculator::acceptResult(int size, double *matrix)
{
    acceptResult(size, matrix, NULL);
}

void
PairCalculator::acceptResult(int size, double *matrix1, double *matrix2)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: Accept Result with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif

  bool unitcoef = false;
  if(matrix2==NULL) unitcoef = true;

  complex *mynewData = new complex[N*grainSize];

  complex *othernewData;
  if(symmetric && thisIndex.x != thisIndex.y){
      othernewData = new complex[N*grainSize];
  }

  int offset = 0, index = thisIndex.y*S + thisIndex.x;

  //ASSUMING TMATRIX IS REAL (LOSS OF GENERALITY)
  register double m=0;  


  int m_in=grainSize;
  int n_in=N;
  int k_in=grainSize;
  int matrixSize=grainSize*grainSize;
  complex *amatrix=new complex[matrixSize];
  int incx=1;
  int incy=2;
  complex alpha=complex(1.0,0.0);//multiplicative identity 
  complex beta=complex(0.0,0.0);
  char transform='N';
  char transformT='T';

  index = thisIndex.x*S + thisIndex.y;
  memset(amatrix, 0, matrixSize*sizeof(complex));
  double *localMatrix;
  double * outMatrix;
  
  for(int i=0;i<grainSize;i++){
    localMatrix = (matrix1+index+i*S);
    outMatrix   = reinterpret_cast <double *> (amatrix+i*grainSize);
    //    DCOPY(&grainSize,localMatrix,&incx, outMatrix,&incy);
    // copy in real leaving imaginary as zeros
    for(incx=0,incy=0;incx<grainSize;incx++,incy+=2)
      outMatrix[incy]=localMatrix[incx];
  }

  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix1[index + j + i*S];
#ifdef _PAIRCALC_DEBUG_
      if(m!=amatrix[i*grainSize+j].re){CkPrintf("Dcopy broken in back path: %2.5g != %2.5g \n",
       						m, amatrix[i*grainSize+j].re);}
#endif _PAIRCALC_DEBUG_
    }
  }

  ZGEMM(&transform, &transformT, &n_in, &m_in, &k_in, &alpha, inDataLeft, &n_in, 
        &(amatrix[0]), &k_in, &beta, &(mynewData[0]), &n_in);

  if(!unitcoef){

  beta=complex(1.0,0.0);  // C = alpha*A*B + beta*C

  for(int i=0;i<grainSize;i++){
    localMatrix = (matrix2+index+i*S);
    outMatrix   = reinterpret_cast <double *> (amatrix+i*grainSize);
    // DCOPY(&grainSize,localMatrix,&incx, outMatrix,&incy);
    for(incx=0,incy=0;incx<grainSize;incx++,incy+=2)
      outMatrix[incy]=localMatrix[incx];
  }
#ifdef _PAIRCALC_DEBUG_
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix2[index + j + i*S];

      if(m!=amatrix[i*grainSize+j].re){CkPrintf("Dcopy broken in back path: %2.5g != %2.5g \n",
						m, amatrix[i*grainSize+j]);}
    }
  }
#endif
  ZGEMM(&transform, &transformT, &n_in, &m_in, &k_in, &alpha, inDataRight, &n_in, 
        &(amatrix[0]), &k_in, &beta, &(mynewData[0]), &n_in);
  }

  delete [] amatrix;


  /* revise this to partition the data into S/M objects 
   * add new message and entry method for sumPartial result
   * to avoid message copying.
   */ 

  //original version
#ifndef _PAIRCALC_SECONDPHASE_LOADBAL_
  if(!symmetric){
    CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
  }
  else {
    CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
    if (thisIndex.y != thisIndex.x){   // FIXME: rowNum will alway == thisIndex.x
      CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, othernewData, thisIndex.z);
    }
  }

#else
  int segments=S/grainSize;
  if(S%grainSize!=0)
      segments+=1;
  int blocksize=grainSize/segments;
  int priority=0xFFFFFFFF;
  if(!symmetric){    
    for(int segment=0;segment < segments;segment++)
      {  
	CkArrayIndex4D idx(thisIndex.w, segment*grainSize, thisIndex.y, thisIndex.z);
	partialResultMsg *msg = new (N*blocksize, 8*sizeof(int) )partialResultMsg;
	msg->N=N*blocksize;
	msg->myoffset = segment*blocksize;
	memcpy(msg->result,mynewData+segment*N*blocksize,msg->N*sizeof(complex));
	msg->cb= cb;
	*((int*)CkPriorityPtr(msg)) = priority;
	CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	thisProxy(idx).sumPartialResult(msg);  
      }
  }
  else { // else part is NOT load balanced yet!!!
    CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
    if (thisIndex.y != thisIndex.x){   // FIXME: rowNum will alway == thisIndex.x
      CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, othernewData, thisIndex.z);
    }

/*
    int segmentsSymm[2];
    findSegNumber(thisIndex.x, thisIndex.y, S, grainSize, segmentsSymm);
    for(int segment=0;segment < segmentsSymm[0];segment++)
      {
	  CkArrayIndex4D idx(thisIndex.w, thisIndex.y, thisIndex.y + segment*grainSize, thisIndex.z);
	  partialResultMsg *msg = new (N*blocksize, 8*sizeof(int) )partialResultMsg;
	  msg->N=N*blocksize;
	  msg->myoffset = segment*blocksize;
	  memcpy(msg->result,mynewData+segment*N*blocksize,msg->N*sizeof(complex));
	  msg->cb= cb;
	  *((int*)CkPriorityPtr(msg)) = priority;
	  CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	  thisProxy(idx).sumPartialResult(msg);  
      }
    if (thisIndex.y != thisIndex.x){ 
	for(int segment=0;segment < segmentsSymm[1];segment++)
	{  
	  CkArrayIndex4D idx(thisIndex.w, thisIndex.x, thisIndex.x + segment*grainSize, thisIndex.z);
	  partialResultMsg *msg = new (N*blocksize, 8*sizeof(int) )partialResultMsg;
	  msg->N=N*blocksize;
	  msg->myoffset = segment*blocksize;
	  memcpy(msg->result,mynewData+segment*N*blocksize,msg->N*sizeof(complex));
	  msg->cb= cb;
	  *((int*)CkPriorityPtr(msg)) = priority;
	  CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	  thisProxy(idx).sumPartialResult(msg);  
	}
    }
*/
  }
#endif

  delete [] mynewData;
  if(symmetric && thisIndex.x != thisIndex.y){
      delete [] othernewData;
  }
  if(conserveMemory)
  {
      // clear the right and left they'll get reallocated on the next pass

      delete [] inDataLeft;
      inDataLeft=NULL;
      if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)) {
	delete [] inDataRight;
	inDataRight = NULL;
      }
    }
}

void 
PairCalculator::sumPartialResult(partialResultMsg *msg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: sum result from grain %d  count %d\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,msg->N, sumPartialCount);
#endif

  sumPartialResult(msg->N, msg->result, msg->myoffset);

  delete msg;
}



void 
PairCalculator::sumPartialResult(priorSumMsg *msg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: sum result \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z );
#endif

  sumPartialResult(msg->N, msg->result, 0);

  delete msg;
}


void 
PairCalculator::sumPartialResult(int size, complex *result, int offset)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: sum result from %d, count %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, offset, sumPartialCount);
#endif

  sumPartialCount++;

  if(!newData){
    newData = new complex[size];
    memset(newData,0,size*sizeof(complex));
  }  
  for(int i=0; i<size; i++){
    newData[i] += result[i];
  }
  if (sumPartialCount == (S/grainSize)*blkSize) {
#ifndef _PAIRCALC_SECONDPHASE_LOADBAL_
    for(int j=0; j<grainSize; j++){
      CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j+offset, thisIndex.w), cb_aid);
      mySendMsg *msg = new (N, 0)mySendMsg; // msg with newData (size N)
      memcpy(msg->data, newData+j*N, N * sizeof(complex));
      msg->N=N;
      mycb.send(msg);
    }
#else
    for(int j=0; j<grainSize/(S/grainSize); j++){
      CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j+offset, thisIndex.w), cb_aid);
      mySendMsg *msg = new (N, 0)mySendMsg; // msg with newData (size N)
      memcpy(msg->data, newData+j*N, N * sizeof(complex));
      msg->N=N;
      mycb.send(msg);
    }
#endif
    sumPartialCount = 0;
    memset(newData,0,size*sizeof(complex));
  }
}

// EJB: Shouldn't this be inlined?
void add_double(void *in, void *inout, int *red_size, void *handle) {
    double * matrix1 = (double *)in;
    double * matrix2 = (double *)inout;
    int size = *red_size / sizeof(double);
    
    for(int i = 0; i < size; i ++){
        matrix2[i] += matrix1[i];
    }
}


#if CONVERSE_VERSION_ELAN

typedef void (* ELAN_REDUCER)(void *in, void *inout, int *count, void *handle);

extern "C" void elan_machine_reduce(int nelem, int size, void * data, 
                                    void *dest, ELAN_REDUCER fn, int root);
extern "C" void elan_machine_allreduce(int nelem, int size, void * data, 
                                       void *dest, ELAN_REDUCER fn);
#endif

void PairCalcReducer::acceptContribute(int size, double* matrix, CkCallback cb, 
                                       bool isAllReduce, bool symmtype)
{
    this->isAllReduce = isAllReduce;
    this->size = size;
    this->symmtype = symmtype;
    this->cb = cb;

#if CONVERSE_VERSION_ELAN
    reduction_elementCount ++;
    
    int red_size = size *sizeof(double);
    if(tmp_matrix == NULL) {
        tmp_matrix = matrix;
    }
    else
        add_double(matrix, tmp_matrix, &red_size, NULL);
    
    if(reduction_elementCount == localElements[symmtype].length()) {
        reduction_elementCount = 0;
        
        contribute(sizeof(int),&reduction_elementCount,CkReduction::sum_int);
    }
#else
    CkAbort("Converse Version Is not ELAN, h/w reduction is not supported");
#endif
}


void PairCalcReducer::startMachineReduction() {
#if CONVERSE_VERSION_ELAN
    double * dst_matrix =  NULL;
    
    if(isAllReduce) {
        dst_matrix = new double[size];
        memset(dst_matrix, 0, size * sizeof(double));
        elan_machine_allreduce(size, sizeof(double), tmp_matrix, dst_matrix, add_double);
    }
    else {     
        int pe = CkNumPes()/2; //HACK FOO BAR, GET IT FROM CALLBACK cb
        
        if(pe == CkMyPe()) {
            dst_matrix = new double[size];
            memset(dst_matrix, 0, size * sizeof(double));
        }
        
        elan_machine_reduce(size, sizeof(double), tmp_matrix, dst_matrix, add_double, pe);
    }
    
    if(isAllReduce) {
         broadcastEntireResult(size, dst_matrix,  symmtype);
        delete [] dst_matrix;
    }
    else {
        if(dst_matrix != NULL) {
            cb.send(size *sizeof(double), dst_matrix);
            delete [] dst_matrix;
        }
    }        
    
    tmp_matrix = NULL;
#else
    CkAbort("Converse Version Is not ELAN, h/w reduction is not supported");
#endif
}

void
PairCalcReducer::broadcastEntireResult(int size, double* matrix, bool symmtype){
  for (int i = 0; i < localElements[symmtype].length(); i++)
    (localElements[symmtype])[i]->acceptResult(size, matrix); 
}

void
PairCalcReducer::broadcastEntireResult(int size, double* matrix1, double* matrix2, bool symmtype){
    CkPrintf("On Pe %d -- %d objects\n", CkMyPe(), localElements[symmtype].length());
  for (int i = 0; i < localElements[symmtype].length(); i++)
    (localElements[symmtype])[i]->acceptResult(size, matrix1, matrix2); 
}

void
PairCalcReducer:: doRegister(PairCalculator *elem, bool symmtype){
    localElements[symmtype].push_back(elem);
    numRegistered[symmtype]++;
}

#include "ckPairCalculator.def.h"
