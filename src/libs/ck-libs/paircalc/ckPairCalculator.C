#include "ckPairCalculator.h"

#define PARTITION_SIZE 500

//#define _SPARSECONT_ 

PairCalculator::PairCalculator(CkMigrateMessage *m) { }
	
PairCalculator::PairCalculator(bool sym, int grainSize, int s, int blkSize,  int op1,  FuncType fn1, int op2,  FuncType fn2, CkCallback cb, CkGroupID gid, CkArrayID cb_aid, int cb_ep) 
{
    //CkPrintf("Create Pair Calculator\n");

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
  kUnits=5;
  kLeftOffset= new int[numExpected];
  kRightOffset= new int[numExpected];
  kLeftCount=0;
  kRightCount=0;
  kLeftMark=kLeftOffset;
  kRightMark=kRightOffset;
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
  p|kRightCount;
  p|kLeftCount;
  p|kUnits;
  p|cb_aid;
  p|cb_ep;
  p|reducer_id;
  p|symmetric;

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
    for (int i = 0; i < numExpected; i++)
      inDataLeft[i] = new complex[N];
    if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
      inDataRight = new complex*[numExpected];
      for (int i = 0; i < numExpected; i++)
	inDataRight[i] = new complex[N];
    }
    kRightOffset= new int[numExpected];
    kLeftOffset= new int[numExpected];
    p|rdiff;
    p|ldiff;
    kRightMark=kRightOffset+rdiff;
    kLeftMark=kLeftOffset+ldiff;
  }
  for (int i = 0; i < numExpected; i++)
    p(inDataLeft[i], N);
  if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)){
    for (int i = 0; i < numExpected; i++)
      p(inDataRight[i], N);
  }
  CkPrintf("ckPairCalculatorPUP\n");
  p(kRightOffset, numExpected);
  p(kLeftOffset, numExpected);

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
  delete [] kRightOffset;
  delete [] kLeftOffset;
}

void
PairCalculator::calculatePairs(int size, complex *points, int sender, bool fromRow, bool flag_dp)
{
#ifdef _DEBUG_
  CkPrintf("     pairCalc[%d %d %d %d] got from [%d %d] with size {%d}, symm=%d, from=%d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  thisIndex.w, sender, size, symmetric, fromRow);
#endif
  int offset = -1;
  complex **inData;
  if (fromRow) {
    offset = sender - thisIndex.x;
    inData = inDataLeft;
    kLeftMark[kLeftCount++]=offset;
  }
  else {
    offset = sender - thisIndex.y;
    inData = inDataRight;
    kRightMark[kRightCount++]=offset;
  }
  if(symmetric && thisIndex.x == thisIndex.y){
    inData = inDataLeft;
        if(!fromRow)
      { //switch
		kLeftMark[kLeftCount++]=offset;
		kLeftCount--;
      }

  }
  N = size; 

  if (!inData[offset]) 
    { // now that we know N we can allocate contiguous space
      inData[0] = new complex[numExpected*N];
      for(int i=0;i<numExpected;i++)
	inData[i]=inData[0]+i*N;
    }

                                                 
  //  if (inData[offset]==NULL)
  //    inData[offset] = new complex[size];
  memcpy(inData[offset], points, size * sizeof(complex));

  numRecd++;   

  // once have K left and K right (or just K left if we're diagonal
  // and symmetric) compute ZDOT for the inputs.

  // Because the vectors are not guaranteed contiguous, record each
  // offset so we can iterate through them

  if((kLeftCount >= kUnits 
      && ((kRightCount >= kUnits) 
	  || (symmetric && thisIndex.x == thisIndex.y) )) 
     || (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y 
					&& numRecd==numExpected)))
    {

      // compute
      // count down kUnits from leftCount starting at kUnit'th element
      int i, j, idxOffset;
      int iunits=(kLeftCount < kUnits)? kLeftCount : kUnits;
      int junits=(kRightCount < kUnits)? kRightCount : kUnits;
      if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected))
	{ //finish 
	  iunits=kLeftCount;
	  junits=kRightCount;
	}
      if(symmetric && thisIndex.x == thisIndex.y) {
	for(int kth=0;kth<iunits;kth++)
	  {
	    i=kLeftMark[kth];
	    for(int jkth=0;jkth<iunits;jkth++)
	      {
		j=kLeftMark[jkth];
#ifdef  _SPARSECONT_	
		outData[i * grainSize + j] = 
#else
		    outData[(i+thisIndex.y)*S + j + thisIndex.x] = 
#endif
                    compute_entry(size, inDataLeft[i],
                                  inDataLeft[j], op1);        
	      }
	  }
      }
      else {                                                        
	// compute a square region of the matrix. The correct part of the
	// region will be used by the reduction.
	for(int kth=0;kth<iunits;kth++)
	  {
	    i=kLeftMark[kth];
	    for(int jkth=0;jkth<junits;jkth++)
	      {
		j=kRightMark[jkth];
#ifdef _SPARSECONT_
		outData[i * grainSize + j] =
#else
		    outData[(i+thisIndex.y)*S + j + thisIndex.x] = 
#endif	    
                    compute_entry(size, inDataLeft[i],
                                  inDataRight[j],op1);        
	      }
	  }
      }
      // move mark past used vectors
      kLeftCount-=iunits;
      kLeftMark+=iunits;
      if(!symmetric || thisIndex.x != thisIndex.y)
	{
	  kRightCount-=junits;
	  kRightMark+=junits;
	}
    }

  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
      numRecd = 0;
      kLeftCount=0;
      kRightCount=0;
      kLeftMark=kLeftOffset;
      kRightMark=kRightOffset;

      if (flag_dp) {
	  if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data
	      for (int i = 0; i < grainSize*grainSize; i++)
		  outData[i] *= 2.0;
	  }
      }
#ifdef _SPARSECONT_
      r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
      r.contribute(this, sparse_sum_double);
#else
#if 1 //!CONVERSE_VERSION_ELAN
      contribute(S * S *sizeof(double), outData, CkReduction::sum_double);
#else

      CkPrintf("ELAN VERSION\n");
      CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id); 
      pairCalcReducerProxy.ckLocalBranch()->acceptContribute(S * S, outData, cb,
                                                             false, symmetric);
#endif
#endif
  }
  
  /*
  if (numRecd == numExpected * 2 || (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)) {
    //    kLeftCount=kRightCount=0;  
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
    if (flag_dp) {
    if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data
	for (i = 0; i < grainSize*grainSize; i++)
	    outData[i] *= 2.0;
    }
    }
    // FIXME: should do 'op2' here!!!

   r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), (CkTwoDoubles*)outData);
    r.contribute(this, sparse_sum_double);
  }
  */
}

void
PairCalculator::acceptEntireResult(int size, double *matrix)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: Accept EntireResult with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif
  CkArrayIndexIndex4D myidx(thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z);
  acceptResult(size, matrix, thisIndex.x);
}


void
PairCalculator::acceptResult(int size, double *matrix, int rowNum)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: Accept Result with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, size);
#endif
  complex *mynewData = new complex[N*grainSize];
  memset(mynewData, 0, sizeof(complex)*N*grainSize);

  complex *othernewData;
  if(symmetric && thisIndex.x != thisIndex.y){
      othernewData = new complex[N*grainSize];
      memset(othernewData, 0, sizeof(complex)*N*grainSize);
  }

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

  // replace with zgemm mynewData=inDataLeft * matrix
  // convert matrix to complex
#if USE_ZGEMM
  int m_in=grainSize;
  int n_in=N;
  int k_in=grainSize;
  int matrixSize=grainSize*grainSize;
  complex *amatrix=new complex[matrixSize];
  int incx=1;
  int incy=2;
  double *localMatrix=matrix+index;
  for(int i=0;i<grainSize;i++)
    dcopy_(&grainSize,localMatrix+i*grainSize,&incx,(double *) (amatrix+i*grainSize),&incy);

  complex alpha=complex(1.0,0.0);//multiplicative identity 
  complex beta=complex(0.0,0.0);
  //  char transform='N';
  char transform='N';
  // hack inData into contiguous  space
  /*
  complex *one=new complex[N*grainSize];
  for (int j = 0; j < grainSize; j++) 
    memcpy(one+j*N,inDataLeft[j],N*sizeof(complex));
  zgemm_(&transform, &transform, &n_in, &m_in, &k_in, &alpha, one, &n_in, amatrix, &k_in, &beta, mynewData, &n_in);
  */
  zgemm_(&transform, &transform, &n_in, &m_in, &k_in, &alpha, &(inDataLeft[0][0]), &n_in, &(amatrix[0]), &k_in, &beta, &(mynewData[0]), &n_in);
  /*
  complex *tdata=new complex[N*grainSize];
  memset(tdata, 0, sizeof(complex)*N*grainSize);
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[index + j + i*S];
      for (int p = 0; p < N; p++)
	tdata[p + i*N] += inDataLeft[j][p] * m;
    }
  }
  if(tdata[0].re!=mynewData[0].re)
    CkPrintf("%d %d: %f  %f \n",0,0,tdata[0].re,mynewData[0].re);
  delete [] tdata;
  */
  if(symmetric && thisIndex.x != thisIndex.y){
    index = thisIndex.x*S + thisIndex.y;
    localMatrix=matrix+index;
    for(int i=0;i<grainSize;i++)
      dcopy_(&grainSize,localMatrix+i*grainSize,&incx,(double *) (amatrix+i*grainSize),&incy);
    // ahh if only we had contiguous data
    /*
    for (int j = 0; j < grainSize; j++) 
      memcpy(one+j*N,inDataRight[j],N*sizeof(complex));
    zgemm_(&transform, &transform, &n_in, &m_in, &k_in, &alpha, one, &n_in, amatrix, &k_in, &beta, othernewData, &n_in);
    */
    zgemm_(&transform, &transform, &n_in, &m_in, &k_in, &alpha, &(inDataRight[0][0]), &n_in, &(amatrix[0]), &k_in, &beta, &(othernewData[0]), &n_in);
  }
  //  delete [] one;
  delete [] amatrix;
#else
  //  this one reads better but may take longer
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      m = matrix[index + j + i*S];
      for (int p = 0; p < N; p++)
	mynewData[p + i*N] += inDataLeft[j][p] * m;
    }
  }
  index = thisIndex.x*S + thisIndex.y;
  if(symmetric && thisIndex.x != thisIndex.y){
      for (int i = 0; i < grainSize; i++) {
	  for (int j = 0; j < grainSize; j++){ 
	      m = matrix[index + j + i*S];
	      for (int p = 0; p < N; p++)
		  othernewData[p + i*N] += inDataRight[j][p] * m;
	  }
      }
  }
#endif
  /* revise this to partition the data into S/M objects 
   * add new message and entry method for sumPartial result
   * to avoid message copying.
   */ 

  //original version
  
/*
  if(!symmetric){
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z, cb);
  }
  else {
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z, cb);
    if (rowNum != thisIndex.x){
      CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, othernewData, thisIndex.z, cb);
                                                                                
    }
  }
*/

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
  else 
  {
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    priorSumMsg *pmsg = new (N*grainSize, 8*sizeof(int) )priorSumMsg();
    pmsg->N=N*grainSize;
    memcpy(pmsg->result,mynewData, pmsg->N*sizeof(complex));
    pmsg->cb= cb;
    *((int*)CkPriorityPtr(pmsg)) = priority;
    CkSetQueueing(pmsg, CK_QUEUEING_IFIFO); 
    thisProxy(idx).sumPartialResult(pmsg);
    if (rowNum != thisIndex.x){
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
  delete [] mynewData;
  if(symmetric && thisIndex.x != thisIndex.y){
      delete [] othernewData;
  }
}

void 
PairCalculator::sumPartialResult(partialResultMsg *msg)
{

//    CkPrintf("[%d %d %d %d]: sum result from grain %d  count %d\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,msg->N, sumPartialCount);

  int segments=S/grainSize;
  if(S%grainSize!=0)
    segments+=1;
  segments*=blkSize;
  int psumblocksize=grainSize/segments;
  if(symmetric)
    segments+=thisIndex.x;
  sumPartialCount++;

  if(!newData){
    newData = new complex[N*psumblocksize];
  }  
  for(int i=0; i<N*psumblocksize; i++){
    newData[i] += msg->result[i];  // should be adjusted with offset
  }
  if (sumPartialCount == segments) {
      //CkPrintf("[%d %d %d %d]: sumPartialCount %d is grainSize %d segments %d, psumblocksize %d, calling back\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,sumPartialCount,grainSize,segments,psumblocksize);
    for(int i=0;i<psumblocksize;i++)
      {
	  // CkPrintf("[%d %d %d %d]: sending to [%d %d]  \n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,thisIndex.y+i+thisIndex.x/grainSize*psumblocksize,thisIndex.w);
	CkCallback mycb(cb_ep, CkArrayIndex2D(/*thisIndex.y/grainSize+i+thisIndex.x/grainSize*psumblocksize*/thisIndex.y+i+thisIndex.x/grainSize*psumblocksize, thisIndex.w), cb_aid);
          mySendMsg *outmsg = new (N,0)mySendMsg; // msg with newData (size N)
          memcpy(outmsg->data, newData+i*psumblocksize, N * sizeof(complex));
          outmsg->N = N;
          mycb.send(outmsg);
      }
    sumPartialCount = 0;
    memset(newData,0,N*psumblocksize*sizeof(complex));
  }
  delete msg;
}



void 
PairCalculator::sumPartialResult(priorSumMsg *msg)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: sum result \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z );
#endif

  sumPartialCount++;

  if(!newData){
    newData = new complex[N*grainSize];
  }  
  for(int i=0; i<N*grainSize; i++){
    newData[i] += msg->result[i];  // should be adjusted with offset
  }
  if (sumPartialCount == (S/grainSize)*blkSize) {
    for(int j=0; j<grainSize; j++){
        CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j, thisIndex.w), cb_aid);
        mySendMsg *outmsg = new (N, 0)mySendMsg; // msg with newData (size N)
        memcpy(outmsg->data, newData+j*N, N * sizeof(complex));
        outmsg->N = N;
        
      mycb.send(outmsg);
    }
    sumPartialCount = 0;
    memset(newData,0,N*grainSize*sizeof(complex));
    //    for(int k=0; k<N*grainSize; k++)
    //	 newData[k] = complex(0,0);
  }
  delete msg;
}


void 
PairCalculator::sumPartialResult(int size, complex *result, int offset)
{
#ifdef _DEBUG_
  CkPrintf("[%d %d %d %d]: sum result from %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, offset);
#endif

  sumPartialCount++;

  if(!newData){
    newData = new complex[N*grainSize];
    memset(newData,0,N*grainSize*sizeof(complex));
  }  
  for(int i=0; i<N*grainSize; i++){
    newData[i] += result[i];  // should be adjusted with offset
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
    //    for(int k=0; k<N*grainSize; k++)
    //	 newData[k] = complex(0,0);
  }
}


void
PairCalcReducer::acceptPartialResult(int size, complex* matrix, int fromRow, int fromCol){
}


void add_double(void *in, void *inout, int *size, void *handle) {
    double * matrix1 = (double *)in;
    double * matrix2 = (double *)inout;

    for(int i = 0; i < *size; i ++){
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
#if CONVERSE_VERSION_ELAN
    reduction_elementCount ++;
    
    if(tmp_matrix == NULL) {
        tmp_matrix = new double[size];
        memset(matrix, 0, size * sizeof(double));
    }
    
    add_double(matrix, tmp_matrix, &size, NULL);

    if(reduction_elementCount == localElements[symmtype].length()) {
        reduction_elementCount = 0;
        double * dst_matrix =  NULL;
        
        if(isAllReduce) {
            dst_matrix = new double[size];
            memset(matrix, 0, size * sizeof(double));
            elan_machine_allreduce(size, sizeof(double), tmp_matrix, dst_matrix, add_double);
        }
        else {     
            CkPrintf("HERE\n");
            int pe = CkNumPes()/2; //HACK FOO BAR, GET IT FROM CALLBACK cb
            
            if(pe == CkMyPe()) {
                dst_matrix = new double[size];
                memset(matrix, 0, size * sizeof(double));
            }
            
            elan_machine_reduce(size, sizeof(double), tmp_matrix, dst_matrix, add_double, pe);
        }
        
        if(isAllReduce) {
            broadcastEntireResult(size, dst_matrix,  symmtype);
            delete [] dst_matrix;
        }
        else {
            cb.send(size *sizeof(double), dst_matrix);
            if(dst_matrix)
                delete [] dst_matrix;
        }        
        
        delete [] tmp_matrix;
        tmp_matrix = NULL;
    }
#else
    CkAbort("Converse Version Is not ELAN, h/w reduction is not supported");
#endif
}

void
PairCalcReducer::broadcastEntireResult(int size, double* matrix, bool symmtype){
  for (int i = 0; i < localElements[symmtype].length(); i++)
    (localElements[symmtype])[i]->acceptEntireResult(size, matrix); 
}

void
PairCalcReducer:: doRegister(PairCalculator *elem, bool symmtype){
    localElements[symmtype].push_back(elem);
    numRegistered[symmtype]++;
}

#include "ckPairCalculator.def.h"
