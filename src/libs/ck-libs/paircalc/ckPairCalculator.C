#include "ckPairCalculator.h"

#define PARTITION_SIZE 500

PairCalculator::PairCalculator(CkMigrateMessage *m) { }
	

PairCalculator::PairCalculator(bool sym, int grainSize, int s, int blkSize,  int op1,  FuncType fn1, int op2,  FuncType fn2, CkCallback cb, CkGroupID gid, CkArrayID cb_aid, int cb_ep) 
{
#ifdef _PAIRCALC_DEBUG_ 
  CkPrintf("[PAIRCALC] [%d %d %d %d] inited\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
#endif 

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

  kUnits=5;   // FIXME: hardcoding the streaming factor to 5

  kLeftOffset= new int[numExpected];
  kRightOffset= new int[numExpected];

  kLeftCount=0;
  kRightCount=0;

  kLeftDoneCount = 0;
  kRightDoneCount = 0;

  inDataLeft = new complex*[numExpected];
  for (int i = 0; i < numExpected; i++)
    inDataLeft[i] = NULL;
  inDataRight = NULL;
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)) {
    inDataRight = new complex*[numExpected];
    for (int i = 0; i < numExpected; i++)
      inDataRight[i] = NULL;
  }

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
  p|kRightCount;
  p|kLeftCount;
  p|kRightDoneCount;
  p|kLeftDoneCount;
  p|kUnits;
  p|cb_aid;
  p|cb_ep;
  p|reducer_id;
  p|symmetric;
  p|sumPartialCount;
  p|N;

  int rdiff,ldiff;
  if(p.isPacking())
    {//store offset calculation
      rdiff=kRightMark-kRightOffset; 
      ldiff=kLeftMark-kLeftOffset;
      p|rdiff;
      p|ldiff;
    }
  if (p.isUnpacking()) {
#ifdef _SPARSECONT_ 
    outData = new double[grainSize * grainSize];
#else
    outData = new double[S*S];
#endif
    inDataLeft = new complex*[numExpected];
    if(N>0)
      for (int i = 0; i < numExpected; i++)
	inDataLeft[i] = new complex[N * blkSize];
    if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
      inDataRight = new complex*[numExpected];
      if(N>0)
	for (int i = 0; i < numExpected; i++)
	  inDataRight[i] = new complex[N * blkSize];
    }
    kRightOffset= new int[numExpected];
    kLeftOffset= new int[numExpected];
    p|rdiff;
    p|ldiff;
    kRightMark=kRightOffset+rdiff;
    kLeftMark=kLeftOffset+ldiff;
  }
  if(N>0)
    for (int i = 0; i < numExpected; i++)
      p(inDataLeft[i], N * blkSize);
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
    if(N>0)  
      for (int i = 0; i < numExpected; i++)
	p(inDataRight[i], N * blkSize);
  }
  p(kRightOffset, numExpected);
  p(kLeftOffset, numExpected);

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
  if(inDataLeft[0]!=NULL)
      delete [] inDataLeft[0];
  delete [] inDataLeft;
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
    if(inDataRight[0]!=NULL)
      delete [] inDataRight[0];
    delete [] inDataRight;
  }

  if(newData!=NULL)
    delete [] newData;
  if(kRightOffset!=NULL)
    delete [] kRightOffset;
  if(kLeftOffset!=NULL)
    delete [] kLeftOffset;
}

void
PairCalculator::calculatePairs(int size, complex *points, int sender, bool fromRow, bool flag_dp)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     pairCalc[%d %d %d %d] got from [%d %d] with size {%d}, symm=%d, from=%d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  thisIndex.w, sender, size, symmetric, fromRow);
#endif

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
  /*
  if(symmetric && thisIndex.x == thisIndex.y){
    inData = inDataLeft;
#ifdef _PAIRCALC_FIRSTPHASE_STREAM_
    if(!fromRow)  CkAbort("Wrong call: fromRow flag should only be true here! \n");
    kLeftOffset[kLeftDoneCount + kLeftCount]=offset;
    kLeftCount++;
#endif
  }
  */

  N = size; // N is init here with the size of the data chunk. 
            // Assuming that data chunk of the same plane across all states are of the same size

  if (inData[0]==NULL) 
  { // now that we know N we can allocate contiguous space
    inData[0] = new complex[numExpected*N];
    for(int i=0;i<numExpected;i++)
      inData[i] = inData[0] + i*N;
  }
  memcpy(inData[offset], points, size * sizeof(complex));


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
		    compute_entry(size, inDataLeft[leftoffset1], inDataLeft[leftoffset2],op1);   
#else
		  outData[(leftoffset1+thisIndex.y)*S + leftoffset2 + thisIndex.x] = 
		    compute_entry(size, inDataLeft[leftoffset1], inDataLeft[leftoffset2],op1);   
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
		    compute_entry(size, inDataLeft[leftoffset], inDataRight[rightoffset],op1);     

#else
		outData[(leftoffset+thisIndex.y)*S + rightoffset + thisIndex.x] = 
		    compute_entry(size, inDataLeft[leftoffset], inDataRight[rightoffset],op1);    
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
		    compute_entry(size, inDataLeft[leftoffset], inDataRight[rightoffset],op1);       

#else
		outData[(leftoffset+thisIndex.y)*S + rightoffset + thisIndex.x] = 
		    compute_entry(size, inDataLeft[leftoffset], inDataRight[rightoffset],op1);        
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

// Below is the old version which works
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
}


void
PairCalculator::acceptResult(int size, double *matrix)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: Accept Result with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif

  complex *mynewData = new complex[N*grainSize];

  complex *othernewData;
  if(symmetric && thisIndex.x != thisIndex.y){
      othernewData = new complex[N*grainSize];
  }

  int offset = 0, index = thisIndex.y*S + thisIndex.x;

  //ASSUMING TMATRIX IS REAL (LOSS OF GENERALITY)
  register double m=0;  

#ifdef _PAIRCALC_SECONDPHASE_BLOCKING_  
  // Obsolete Opt: substitute with ZGEMM 
  // Note: This is not correct, need to be fixed or removed!!!
  memset(mynewData, 0, grainSize*N*sizeof(complex));
  int size1 = 0;
  for(size1 = 0; size1 + PARTITION_SIZE < N; size1 += PARTITION_SIZE) {
    for (int i = 0; i < grainSize; i++) {
      int iSindex=i*S+index;
      int iN=i*N;
      complex *newiNdata=&mynewData[iN];
      for (int j = 0; j < grainSize; j++){ 
	m = matrix[iSindex + j];
	for (int p = size1; p < size1+PARTITION_SIZE; p++)
	  newiNdata[p] += inDataLeft[j][p] * m;
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
	  newiNdata[p] += inDataLeft[j][p] * m;
      }
    }
  }
#else

  // replace with zgemm mynewData=inDataLeft * matrix
  // convert matrix to complex
#ifdef _PAIRCALC_USE_ZGEMM_
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
  complex *leftptr = inDataLeft[0];
  complex *rightptr = inDataRight[0];

  index = thisIndex.x*S + thisIndex.y;
  memset(amatrix, 0, matrixSize*sizeof(complex));
  double *localMatrix;
  double * outMatrix;
  for(int i=0;i<grainSize;i++){
    localMatrix = (matrix+index+i*S);
    outMatrix   = (double*)(amatrix+i*grainSize);
    DCOPY(&grainSize,localMatrix,&incx, outMatrix,&incy);
  }

  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[index + j + i*S];
      if(m!=amatrix[i*grainSize+j].re){CkPrintf("Dcopy broken in back path: %2.5g != %2.5g \n",
      						m, amatrix[i*grainSize+j].re);}
    }
  }

  ZGEMM(&transform, &transformT, &n_in, &m_in, &k_in, &alpha, &(inDataLeft[0][0]), &n_in, 
        &(amatrix[0]), &k_in, &beta, &(mynewData[0]), &n_in);
  /*
  if(symmetric && thisIndex.x != thisIndex.y){
    index = thisIndex.x*S + thisIndex.y;
    localMatrix=matrix+index;
    for(int i=0;i<grainSize;i++){
      localMatrix=matrix+index+i*S;
      outMatrix   = (double*)(amatrix+i*grainSize);
      DCOPY(&grainSize,localMatrix,&incx,outMatrix,&incy);
    }
    ZGEMM(&transform, &transform, &n_in, &m_in, &k_in, &alpha, &(inDataRight[0][0]), &n_in, 
	  &(amatrix[0]), &k_in, &beta, &(othernewData[0]), &n_in);
  }
  */

  delete [] amatrix;

#else

  complex *leftptr = inDataLeft[0];
  complex *rightptr = inDataRight[0];

  // Original calculation : without optimize
  memset(mynewData, 0, N*grainSize*sizeof(complex));
  index = thisIndex.y*S + thisIndex.x;
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[index + j + i*S];
      for (int p = 0; p < N; p++){
	mynewData[p + i*N] += leftptr[j*N + p] * m;
      }
    }
  }
  index = thisIndex.x*S + thisIndex.y;
  /*
  if(symmetric && thisIndex.x != thisIndex.y){
      memset(othernewData, 0, N*grainSize*sizeof(complex));
      for (int i = 0; i < grainSize; i++) {
	  for (int j = 0; j < grainSize; j++){ 
	      m = matrix[index + j + i*S];
	      for (int p = 0; p < N; p++)
		  othernewData[p + i*N] += rightptr[j*N + p] * m;
	  }
      }
  }
  */
#endif
#endif

  /* revise this to partition the data into S/M objects 
   * add new message and entry method for sumPartial result
   * to avoid message copying.
   */ 

  //original version
#ifndef _PAIRCALC_SECONDPHASE_LOADBAL_
  if(!symmetric){
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
  }
  else {
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
    if (thisIndex.y != thisIndex.x){   // FIXME: rowNum will alway == thisIndex.x
      CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, othernewData, thisIndex.z);
    }
  }

#else
  int segments=S/grainSize;
  if(S%grainSize!=0)
      segments+=1;
  int blocksize=grainSize/segments;
  int priority=0xFFFFFFFF;
  if(!symmetric){    // Not right in value given!!!
    for(int segment=0;segment < segments;segment++)
      {  
	CkArrayIndexIndex4D idx(thisIndex.w, segment*grainSize, thisIndex.y, thisIndex.z);
	partialResultMsg *msg = new (N*blocksize, 8*sizeof(int) )partialResultMsg;
	msg->N=N*blocksize;
	memcpy(msg->result,mynewData+segment*N*blocksize,msg->N*sizeof(complex));
	msg->cb= cb;
	*((int*)CkPriorityPtr(msg)) = priority;
	CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	thisProxy(idx).sumPartialResult(msg);  
      }
  }
  else { // else part is NOT load balanced yet!!!

    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    priorSumMsg *pmsg = new (N*grainSize, 8*sizeof(int) )priorSumMsg();
    pmsg->N=N*grainSize;
    memcpy(pmsg->result,mynewData, pmsg->N*sizeof(complex));
    pmsg->cb= cb;
    *((int*)CkPriorityPtr(pmsg)) = priority;
    CkSetQueueing(pmsg, CK_QUEUEING_IFIFO); 
    thisProxy(idx).sumPartialResult(pmsg);
    if (thisIndex.y != thisIndex.x){
      CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      priorSumMsg *pmsg = new (N*grainSize, 8*sizeof(int) )priorSumMsg();
      pmsg->N=N*grainSize;
      memcpy(pmsg->result,othernewData, pmsg->N*sizeof(complex));
      pmsg->cb= cb;
      *((int*)CkPriorityPtr(pmsg)) = priority;
      CkSetQueueing(pmsg, CK_QUEUEING_IFIFO); 
      thisProxy(idx).sumPartialResult(pmsg);  
    }
  }
#endif

  delete [] mynewData;
  if(symmetric && thisIndex.x != thisIndex.y){
      delete [] othernewData;
  }
}

void 
PairCalculator::sumPartialResult(partialResultMsg *msg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: sum result from grain %d  count %d\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,msg->N, sumPartialCount);
#endif

  sumPartialResult(N, msg->result, 0);

  delete msg;
}



void 
PairCalculator::sumPartialResult(priorSumMsg *msg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: sum result \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z );
#endif

  sumPartialResult(N, msg->result, 0);

  delete msg;
}


void 
PairCalculator::sumPartialResult(int size, complex *result, int offset)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d]: sum result from %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, offset);
#endif

  sumPartialCount++;

  if(!newData){
    newData = new complex[N*grainSize];
    memset(newData,0,N*grainSize*sizeof(complex));
  }  
  for(int i=0; i<N*grainSize; i++){
    newData[i] += result[i];  
  }
  if (sumPartialCount == (S/grainSize)*blkSize) {
    for(int j=0; j<grainSize; j++){
      CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j, thisIndex.w), cb_aid);
      mySendMsg *msg = new (N, 0)mySendMsg; // msg with newData (size N)
      memcpy(msg->data, newData+j*N, N * sizeof(complex));
      msg->N=N;
      mycb.send(msg);
    }
    sumPartialCount = 0;
    memset(newData,0,N*grainSize*sizeof(complex));
  }
}

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
        //CkPrintf("Calling BroadcastEntire\n");
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
PairCalcReducer:: doRegister(PairCalculator *elem, bool symmtype){
    localElements[symmtype].push_back(elem);
    numRegistered[symmtype]++;
}

#include "ckPairCalculator.def.h"
