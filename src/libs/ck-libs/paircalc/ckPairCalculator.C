#include "ckPairCalculator.h"

#define PARTITION_SIZE 500

PairCalculator::PairCalculator(CkMigrateMessage *m) { }
	
PairCalculator::PairCalculator(bool sym, int grainSize, int s, int blkSize,  int op1,  FuncType fn1, int op2,  FuncType fn2, CkCallback cb, CkGroupID gid) 
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

  //outData = new double[grainSize * grainSize];
  outData = new double[S * S];
  memset(outData, 0 , sizeof(double)* S *S);

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
  p|kRightCount;
  p|kLeftCount;
  p|kUnits;
  int rdiff,ldiff;
  if(p.isPacking())
    {//store offset calculation
      rdiff=kRightMark-kRightOffset; 
      ldiff=kLeftMark-kLeftOffset;
      p|rdiff;
      p|ldiff;
    }
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
    inData[offset] = new complex[size];
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
		outData[(i+thisIndex.y)*S + j + thisIndex.x] = 
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
		j=kRightOffset[jkth];
		outData[(i+thisIndex.y)*S + j + thisIndex.x] = 
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
      //r.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), (CkTwoDoubles*)outData);
      //r.contribute(this, sparse_sum_double);
      
      contribute(S * S *sizeof(double), outData, CkReduction::sum_double);
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
  */
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
}


void
PairCalculator::acceptResult(int size, double *matrix, int rowNum, CkCallback cb)
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
/*    for(int segment=0;segment < segments;segment++){

      //      CkPrintf("[%d %d %d %d]: sending N %d segment %d of %d segments \n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z, N, segment,segments);
      CkArrayIndexIndex4D idx(thisIndex.w, segment*grainSize, thisIndex.y, thisIndex.z);

      //	partialResultMsg *msg = new (N*blocksize, 8*sizeof(int) )partialResultMsg(N*blocksize, mynewData+segment*N*blocksize, priority, cb);
      //	partialResultMsg *msg = new (N*blocksize, 0)partialResultMsg(N*blocksize, mynewData+segment*N*blocksize, priority, cb);
      
      partialResultMsg *msg = new (N*blocksize, 0)partialResultMsg;
      msg->N=N*blocksize;
      memcpy(msg->result, mynewData+segment*N*blocksize, N*blocksize*sizeof(complex));
      msg->priority=priority;
      msg->cb=cb;
      //	*((int*)CkPriorityPtr(msg)) = priority;
      //	CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
      thisProxy(idx).sumPartialResult(msg);  
      
      if (rowNum != thisIndex.x){
        CkArrayIndexIndex4D idx(thisIndex.w, segment*grainSize, thisIndex.x, thisIndex.z);
	if(thisIndex.x < segment*grainSize && segment < segmants/2) {
	    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
	}
	else {
	    CkArrayIndexIndex4D idx(thisIndex.w, thisIndex.x, thisIndex.x, thisIndex.z);
	}

	//	partialResultMsg *msg = new (N*blocksize, 8*sizeof(int) )partialResultMsg(N*blocksize, mynewData+segment*N*blocksize, priority, cb);
	//	partialResultMsg *msg = new (N*blocksize, 0)partialResultMsg(N*blocksize, mynewData+segment*N*blocksize, priority, cb);
      
	partialResultMsg *msg1 = new (N*blocksize, 0)partialResultMsg;
	msg1->N=N*blocksize;
	memcpy(msg1->result, othernewData+segment*N*blocksize, N*blocksize*sizeof(complex));
	msg1->priority=priority;
	msg1->cb=cb;
	//	*((int*)CkPriorityPtr(msg)) = priority;
	//	CkSetQueueing(msg, CK_QUEUEING_IFIFO); 
	thisProxy(idx).sumPartialResult(msg1);  
      }
    }
*/
/*
    CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z, cb);
    if (rowNum != thisIndex.x){
      CkArrayIndexIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, othernewData, thisIndex.z, cb);

    }                                                                             */
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
	  CkCallback mycb(msg->cb.d.array.ep, CkArrayIndex2D(/*thisIndex.y/grainSize+i+thisIndex.x/grainSize*psumblocksize*/thisIndex.y+i+thisIndex.x/grainSize*psumblocksize, thisIndex.w), msg->cb.d.array.id);
	
          mySendMsg *outmsg = new (N*psumblocksize,0)mySendMsg; // msg with newData (size N)
          memcpy(outmsg->data, newData+i*psumblocksize, N*psumblocksize * sizeof(complex));
          outmsg->N = N*psumblocksize;
          
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
      CkCallback mycb(msg->cb.d.array.ep, CkArrayIndex2D(thisIndex.y+j, thisIndex.w), msg->cb.d.array.id);
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
PairCalculator::sumPartialResult(int size, complex *result, int offset, CkCallback cb)
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
      CkCallback mycb(cb.d.array.ep, CkArrayIndex2D(thisIndex.y+j, thisIndex.w), cb.d.array.id);
      mySendMsg *msg = new (N, 0)mySendMsg; // msg with newData (size N)
      memcpy(msg->data, newData+j*N, N * sizeof(complex));
      mycb.send(msg);
    }
    sumPartialCount = 0;
    memset(newData,0,N*grainSize*sizeof(complex));
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
