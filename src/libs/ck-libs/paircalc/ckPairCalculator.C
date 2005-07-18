#include "ckPairCalculator.h"

#define PARTITION_SIZE 500
/***************************************************************************
 * This is a matrix multiply library with extra frills to communicate the  *
 * results back to gspace or the calling ortho char as directed by the     *
 * callback.                                                               *
 *                                                                         *
 * The extra complications are for parallelization and the multiplication  *
 * of the forces and energies.                                             *
 *                                                                         *
 * In normal use the calculator is created.  Then forces are sent to it    *
 * and multiplied in a big dgemm.  Then this result is reduced to the      *
 * answer matrix and shipped back.  The received left and/or right data is *
 * retained for the backward pass which is triggered by the finishPairCalc *
 * call.  This carries in another set of matrices for multiplication.      *
 * The results are again reduced and cast back.  Thus terminating the life *
 * cycle of the data in the pair calculator.  As the calculator will be    *
 * reused again throughout each iteration the calculators themselves are   *
 * only created once.                                                      *
 *                                                                         *
 * The elan code is a specialized machine reduction/broadcast which runs   *
 * much faster on lemieux, and theoretically on any other elan machine     *
 * It operates by one dummy reduction which is used to indicate that all   *
 * calculators on a PE machine have reported in.  Then the machine         *
 * reduction is triggered.                                                 *
 *                                                                         *
 * The paircalculator is a 4 dimensional array.  Those dimensions are:     *
 *            w: gspace state plane (the second index of the 2D gspace)    *
 *            x: coordinate offset within plane (a factor of grainsize)    *
 *            y: coordinate offset within plane (a factor of grainsize)    *
 *            z: blocksize, unused always 0                                *
 *       So, for an example grainsize of 64 for a 128x128 problem:         *
 *        S/grainsize gives us a 2x2 decomposition.                        *
 *        1st quadrant ranges from [0,0]   to [63,63]    index [w,0,0,0]   *
 *        2nd quadrant ranges from [0,64]  to [63,127]   index [w,0,64,0]  *
 *        3rd quadrant ranges from [64,0]  to [127,63]   index [w,64,0,0]  *
 *        4th quadrant ranges from [64,64] to [127,127]  index [w,64,64,0] *
 *                                                                         *
 *       0   64   127                                                      *
 *     0 _________                                                         *
 *       |   |   |                                                         *
 *       | 1 | 2 |                                                         *
 *    64 ---------                                                         *
 *       |   |   |                                                         *
 *       | 3 | 4 |                                                         *
 *   127 ---------                                                         *
 *                                                                         *
 *                                                                         *
 *                                                                         *
 * Further complication arises from the fact that each plane is a          *
 * cross-section of a sphere.  So the actual data is sparse and is         *
 * represented by a contiguous set of the nonzero elements.  This is       * 
 * commonly referred to as N or size within the calculator.                *
 *                                                                         *
 ***************************************************************************/

ComlibInstanceHandle mcastInstanceCP;

CkReduction::reducerType sumMatrixDoubleType;

void registersumMatrixDouble(void)
{ 
  sumMatrixDoubleType=CkReduction::addReducer(sumMatrixDouble);
}


// sum up the contribution for each plane of the grain
CkReductionMsg *sumMatrixDouble(int nMsg, CkReductionMsg **msgs)
{
  double *ret=(double *)msgs[0]->getData();
  // all pieces are grainSize* grainSize
  double *inmatrix;
  int offset;
  int size=msgs[0]->getSize()/sizeof(double);

  for(int i=1; i<nMsg;i++)
    {
      inmatrix=(double *) msgs[i]->getData();
      for(int d=0;d<size;d++)
	ret[d]+=inmatrix[d];
    }
  return CkReductionMsg::buildNew(size*sizeof(double),ret);
}


PairCalculator::PairCalculator(CkMigrateMessage *m) { }

PairCalculator::PairCalculator(bool sym, int grainSize, int s, int blkSize,  int op1,  FuncType fn1, int op2,  FuncType fn2, CkCallback cb, CkGroupID gid, CkArrayID cb_aid, int cb_ep, bool conserveMemory, bool lbpaircalc, CkCallback lbcb, redtypes _cpreduce, bool gspacesum)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[PAIRCALC] [%d %d %d %d] inited lb %d \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,lbpaircalc);
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
  existsLeft=false;
  existsRight=false;
  existsOut=false;
  existsNew=false;
  numRecd = 0;
  newelems=0;
  numExpected = grainSize;
  this->cb_lb=lbcb;
  this->gspacesum=gspacesum;
  cpreduce=_cpreduce;
  resumed=true;
  kUnits=grainSize;  //streaming unit only really used in NOGEMM, but could be used under other conditions

  inDataLeft = NULL;
  inDataRight = NULL;

  outData = NULL;

  newData = NULL;
  sumPartialCount = 0;
  usesAtSync=true;
  if(lbpaircalc)
    setMigratable(true);
  else
    setMigratable(false);

  CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id);
  CkArrayIndex4D indx4(thisIndex.w,thisIndex.x, thisIndex.y, thisIndex.z);
  CkArrayID myaid=thisProxy.ckGetArrayID();
  IndexAndID iandid(indx4,myaid);
  pairCalcReducerProxy.ckLocalBranch()->doRegister(iandid, symmetric);

}

void
PairCalculator::pup(PUP::er &p)
{
  ArrayElement4D::pup(p);
  p|symmetric;
  p|resumed;
  p|numRecd;
  p|grainSize;
  p|numExpected;
  p|S;
  p|blkSize;
  p|N;
  p|kUnits;
  p|op1;
  p|op2;
  p|fn1;
  p|fn2;
  p|sumPartialCount;
  p|conserveMemory;
  p|lbpaircalc;
  p|cb;
  p|cpreduce;
  p|cb_aid;
  p|cb_ep;
  p|reducer_id;
  p|existsLeft;
  p|existsRight;
  p|existsOut;
  p|existsNew;
  p|cb_lb;
  p|newelems;
  p|gspacesum;
  p|mCastGrpId;
  p|cookie;
  if (p.isUnpacking()) {
    if(existsNew)
      newData= new complex[newelems];
    else
      newData=NULL;
    if(existsOut)
      {
	outData= new double[grainSize*grainSize];
	// just in case we migrated with a reduction in progress
	// this is the only way to restore the sparseContiguous reducer state
	// if we didn't, no harm done.
	if(cpreduce==sparsecontiguous)
	  {
	    sparseRed.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
	  }
      }
    else
      outData=NULL;
    if(existsLeft)
      inDataLeft = new double[2*numExpected*N];
    else
      inDataLeft=NULL;
    if(existsRight)
      inDataRight = new double[2*numExpected*N];
    else
      inDataRight=NULL;

  }
  if(existsLeft)
    p((void*) inDataLeft, numExpected * N * 2* sizeof(double));
  if(existsRight)
    p((void*) inDataRight, numExpected* N * 2* sizeof(double));
  if(existsNew)
    p((void*) newData, newelems* sizeof(complex));
  if(existsOut)
    p((void*) outData, grainSize*grainSize* sizeof(double));

#ifdef _PAIRCALC_DEBUG_
  if (p.isUnpacking())
    {
      CkPrintf("[%d,%d,%d,%d,%d] pup unpacking on %d resumed=%d memory %d\n",thisIndex.w,thisIndex.x, thisIndex.y, thisIndex.z,symmetric,CkMyPe(),resumed, CmiMemoryUsage());
      CkPrintf("[%d,%d,%d,%d,%d] pupped : %d %d %d %d %d %d %d %d %d fn1 fn2 %d %d %d %d %d cb cb_aid %d reducer_id sparsRed %d %d cb_lb inDataLeft inDataRight outData newData %d %d\n",thisIndex.w,thisIndex.x, thisIndex.y, thisIndex.z,symmetric, numRecd, numExpected, grainSize, S, blkSize, N, kUnits, op1, op2, sumPartialCount,symmetric, conserveMemory, lbpaircalc, cpreduce, cb_ep, existsLeft, existsRight, gspacesum, resumed);

    }
  else
    CkPrintf("[%d,%d,%d,%d,%d] pup called on %d\n",thisIndex.w,thisIndex.x, thisIndex.y, thisIndex.z,symmetric,CkMyPe());
#endif


}

PairCalculator::~PairCalculator()
{

#ifdef _PAIRCALC_DEBUG_
      CkPrintf("[%d,%d,%d,%d,%d] destructs on [%d]\n",thisIndex.w,thisIndex.x,thisIndex.y, thisIndex.z, symmetric, CkMyPe());
#endif
  if(outData!=NULL)
    delete [] outData;

  if(inDataLeft!=NULL)
    delete [] inDataLeft;

  if(inDataRight!=NULL)
    delete [] inDataRight;

  if(newData!=NULL)
    delete [] newData;
  // redundant paranoia
  outData=NULL;
  inDataRight=NULL;
  inDataLeft=NULL;
  newData=NULL;
  existsLeft=false;
  existsRight=false;
  existsNew=false;
  existsOut=false;
}






// initialize the section cookie and the reduction client
void PairCalculator::initGRed(initGRedMsg *msg)
{
  CkGetSectionInfo(cookie,msg);
  cb=msg->cb;
  mCastGrpId=msg->mCastGrpId;
  cpreduce=section;
  //  thisProxy.ckSetReductionClient(&cb);  //probably redundant
  delete msg;
}

void PairCalculator::ResumeFromSync() {
  if(!resumed){
  resumed=true;
#ifdef _PAIRCALC_DEBUG_
      CkPrintf("[%d,%d,%d,%d,%d] resumes from sync\n",thisIndex.w,thisIndex.x,thisIndex.y, thisIndex.z, symmetric);
#endif
#ifndef _PAIRCALC_SLOW_FAT_SIMPLE_CAST_
      CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id);
      CkArrayIndex4D indx4(thisIndex.w,thisIndex.x, thisIndex.y, thisIndex.z);
      CkArrayID myaid=thisProxy.ckGetArrayID();
      IndexAndID iandid(indx4,myaid);
      pairCalcReducerProxy.ckLocalBranch()->doRegister(iandid, symmetric);
#endif
  }
}
void
PairCalculator::calculatePairs_gemm(calculatePairsMsg *msg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf(" symm=%d    pairCalc[%d %d %d %d] got from [%d %d] with size {%d}, from=%d, count=%d, resumed=%d\n", symmetric, thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,  thisIndex.w, msg->sender, msg->size, msg->fromRow, numRecd,resumed);
#endif
  if(!resumed)
    {
      ResumeFromSync();
    }
  numRecd++;   // increment the number of received counts
  int offset = -1;
  if (msg->fromRow) {   // This could be the symmetric diagonal case
    offset = msg->sender - thisIndex.x;
    if (!existsLeft)
      { // now that we know N we can allocate contiguous space
	CkAssert(inDataLeft==NULL);
	existsLeft=true;
	N = msg->size; // N is init here with the size of the data chunk.
	inDataLeft = new double[numExpected*N*2];
	memset(inDataLeft,0,numExpected*N*2*sizeof(double));
#ifdef _PAIRCALC_DEBUG_
	CkPrintf("[%d,%d,%d,%d,%d] Allocated Left %d * %d *2 \n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric, numExpected,N);

#endif
      }
    CkAssert(N==msg->size);
    CkAssert(offset<numExpected);
    memcpy(&(inDataLeft[offset*N*2]), msg->points, N * 2 *sizeof(double));
#ifdef _PAIRCALC_DEBUG_
    CkPrintf("[%d,%d,%d,%d,%d] Copying into offset*N %d * %d N *2 %d points start %g end %g\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z, symmetric, offset, N, N*2,msg->points[0].re, msg->points[N-1].im);
#endif
#ifdef _PAIRCALC_DEBUG_STUPID_PARANOID_
    CkPrintf("copying in \n");
    double re;
    double im;
    for(int i=0;i<N;i++)
      {
	re=msg->points[i].re;
	im=msg->points[i].im;
	CkPrintf("CL %d %g %g\n",i,re,im);
	if(fabs(re)>0.0)
	  CkAssert(fabs(re)>1.0e-300);
	if(fabs(im)>0.0)
	  CkAssert(fabs(im)>1.0e-300);
      }
#endif
  }
  else {
    offset = msg->sender - thisIndex.y;
    if (!existsRight)
      { // now that we know N we can allocate contiguous space
	CkAssert(inDataRight==NULL);
	existsRight=true;
	N = msg->size; // N is init here with the size of the data chunk.
	inDataRight = new double[numExpected*N*2];
	memset(inDataRight,0,numExpected*N*2*sizeof(double));
#ifdef _PAIRCALC_DEBUG_
	CkPrintf("[%d,%d,%d,%d,%d] Allocated right %d * %d *2 \n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric,numExpected,N);
#endif
      }
    CkAssert(N==msg->size);
    CkAssert(offset<numExpected);
    memcpy(&(inDataRight[offset*N*2]), msg->points, N * 2 *sizeof(double));
#ifdef _PAIRCALC_DEBUG_
    CkPrintf("[%d,%d,%d,%d,%d] Copying into offset*N %d * %d N *2 %d points start %g end %g\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric,offset,N, N*2,msg->points[0].re, msg->points[N-1].im);
#endif
#ifdef _PAIRCALC_DEBUG_STUPID_PARANOID_
    CkPrintf("copying in \n");
    double re;
    double im;
    for(int i=0;i<N;i++)
      {
	re=msg->points[i].re;
	im=msg->points[i].im;
	CkPrintf("CR %d %g %g\n",i,re,im);
	if(fabs(re)>0.0)
	  CkAssert(fabs(re)>1.0e-300);
	if(fabs(im)>0.0)
	  CkAssert(fabs(im)>1.0e-300);
      }
#endif
  }


  /*
   *  NOTE: For this to work the data chunks of the same plane across
   *  all states must be of the same size
   */

  // copy the input into our matrix


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
    if(!existsOut){
      CkAssert(outData==NULL);
      existsOut=true;
      outData = new double[grainSize * grainSize];
      memset(outData, 0 , sizeof(double)* grainSize * grainSize);
#ifdef _PAIRCALC_DEBUG_
       CkPrintf("[%d,%d,%d,%d,%d] Allocated outData %d * %d\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric,grainSize, grainSize);
#endif
    }


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
#ifndef CMK_OPTIMIZE
    double StartTime=CmiWallTimer();
#endif
#ifdef _PAIRCALC_DEBUG_
    CkPrintf("[%d,%d,%d,%d,%d] gemming %c %c %d %d %d %f A %d B %d %f C %d\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric,transformT,transform, m_in, n_in, k_in, alpha, lda, ldb, beta, ldc);
#endif
    if( numRecd == numExpected * 2)
      {
#ifdef _PAIRCALC_DEBUG_PARANOID_
	char filename[80];
	sprintf(filename, "fwlmdata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
	FILE *loutfile = fopen(filename, "w");
	fprintf(loutfile,"[%d,%d,%d,%d,%d] inDataLeft\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
	for(int i=0;i<N*2;i++)
	  for(int j=0;j<numExpected;j++)
	    fprintf(loutfile,"%d %d %g \n",i,j,inDataLeft[i*numExpected+j]);
	fclose(loutfile);
	sprintf(filename, "fwrmdata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
	FILE *routfile = fopen(filename, "w");
	fprintf(routfile,"[%d,%d,%d,%d,%d] inDataRight\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
	for(int i=0;i<N*2;i++)
	  for(int j=0;j<numExpected;j++)
	    fprintf(routfile,"%d %d %g\n",i,j,inDataRight[i*numExpected+j]);
	fclose(routfile);
#endif

	DGEMM(&transformT, &transform, &m_in, &n_in, &k_in, &alpha, inDataRight, &lda, inDataLeft, &ldb, &beta, outData, &ldc);
	// switching solves a transposition problem
	//	  DGEMM(&transformT, &transform, &m_in, &n_in, &k_in, &alpha, inDataLeft, &lda, inDataRight, &ldb, &beta, outData, &ldc);
      }
    else if (symmetric && thisIndex.x==thisIndex.y && numRecd==numExpected)
      {
#ifdef _PAIRCALC_DEBUG_PARANOID_
	char filename[80];
	sprintf(filename, "fwlmdata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
	FILE *loutfile = fopen(filename, "w");
	fprintf(loutfile,"[%d,%d,%d,%d,%d] inDataLeft\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
	for(int i=0;i<N*2;i++)
	  for(int j=0;j<numExpected;j++)
	    fprintf(loutfile,"%d %d %g \n",i,j,inDataLeft[i*numExpected+j]);
	fclose(loutfile);
#endif
	DGEMM(&transformT, &transform, &m_in, &n_in, &k_in, &alpha, inDataLeft, &lda, inDataLeft, &ldb, &beta, outData, &ldc);
      } 

#ifndef CMK_OPTIMIZE
    traceUserBracketEvent(210, StartTime, CmiWallTimer());
#endif

    numRecd = 0; 

    if (msg->flag_dp) {
      if(thisIndex.w != 0) {   // Adjusting for double packing of incoming data

	for (int i = 0; i < grainSize*grainSize; i++)
	  outData[i] *= 2.0;
      }
    }
#ifdef _PAIRCALC_DEBUG_PARANOID_
    char filename[80];
    sprintf(filename, "fwgmodata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
    FILE *outfile = fopen(filename, "w");


    fprintf(outfile,"[%d,%d,%d,%d,%d] outData=C\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
    for(int i=0;i<grainSize;i++)
      {
	for(int j=0;j<grainSize;j++)
	  fprintf(outfile," %g",outData[i*grainSize+j]);
	  fprintf(outfile,"\n");
      }
    fclose(outfile);
#endif

#ifdef _PAIRCALC_VALID_OUT_
    CkPrintf("[PAIRCALC] [%d %d %d %d %d] forward gemm out %.10g %.10g %.10g %.10g \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,symmetric, outData[0],outData[1],outData[grainSize*grainSize-2],outData[grainSize*grainSize-1]);
#endif
#ifndef CMK_OPTIMIZE
    StartTime=CmiWallTimer();
#endif


    if(cpreduce==machine){
	//CkPrintf("[%d] ELAN VERSION %d\n", CkMyPe(), symmetric);
	CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id);
	pairCalcReducerProxy.ckLocalBranch()->acceptContribute(S * S, outData,
							       cb, !symmetric, symmetric, thisIndex.x, thisIndex.y, grainSize);
#ifdef _ELAN_PAIRCALC_DEBUG_
	CkPrintf("[PAIRCALC] [%d %d %d %d %d] called acceptContribute \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,symmetric);
#endif

    }
    else if(cpreduce==sparsecontiguous)
      {
	sparseRed.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
	sparseRed.contribute(this, sparse_sum_double);
      }
    else //section reduction
      {
	CkMulticastMgr *mcastGrp=CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
	mcastGrp->contribute(grainSize*grainSize*sizeof(double), outData, sumMatrixDoubleType, cookie, cb);
      }

#ifndef CMK_OPTIMIZE
    traceUserBracketEvent(220, StartTime, CmiWallTimer());
#endif
  }

}

void
PairCalculator::acceptResultSlow(acceptResultMsg *msg)
{
  acceptResult(msg->size, msg->matrix, NULL);
}

void
PairCalculator::acceptResultSlow(acceptResultMsg2 *msg)
{
  acceptResult(msg->size, msg->matrix1, msg->matrix2);
}

void
PairCalculator::acceptResult(acceptResultMsg *msg)
{
  acceptResult(msg->size, msg->matrix, NULL);
}

void
PairCalculator::acceptResult(acceptResultMsg2 *msg)
{
  acceptResult(msg->size, msg->matrix1,msg->matrix2);
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
  CkPrintf("[%d %d %d %d %d]: Accept Result with size %d\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, symmetric, size);
#endif

  bool unitcoef = false;
  if(matrix2==NULL) unitcoef = true;

  complex *mynewData = new complex[N*grainSize];
  
  complex *othernewData;

  if(symmetric && (thisIndex.x != thisIndex.y)){
    othernewData = new complex[N*grainSize];
    memset(othernewData,0,N*grainSize* sizeof(complex));
  }

  int offset = 0, index = thisIndex.y*S + thisIndex.x;

  if(!symmetric)
    index = thisIndex.x*S + thisIndex.y;

  //ASSUMING TMATRIX IS REAL (LOSS OF GENERALITY)
  register double m=0;
  bool makeLocalCopy=false;
  double *amatrix=NULL;

  if(S!=grainSize && size!=S)
    makeLocalCopy=true;
  
  if(S==grainSize)// all at once no malloc
    {
      amatrix=matrix1+index;
    }
  else
    { // you were sent the correct section only
      amatrix=matrix1;
    }

  int matrixSize=grainSize*grainSize;
  double *localMatrix;
  double *outMatrix;

  if(makeLocalCopy)
    {
#ifdef _PAIRCALC_DEBUG_PARANOID_
      char ifilename[80];
      sprintf(ifilename, "bwim1data.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
      FILE *ofile = fopen(ifilename, "w");
      fprintf(ofile,"[%d,%d,%d,%d,%d] IM1\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
      for(int i=0;i<S*S;i++)
	{
	  fprintf(ofile,"%.10g\n",matrix1[i]);
	}
      fclose(ofile);
#endif
      // copy grainSize chunks at S offsets
      amatrix=new double[matrixSize];
      memset(amatrix,0,matrixSize *sizeof(double));
      CkAssert(size>=matrixSize);
      for(int i=0;i<grainSize;i++){
	localMatrix = (matrix1+index+i*S);
	outMatrix   = (double *) (amatrix+i*grainSize);
	memcpy(outMatrix,localMatrix,grainSize*sizeof(double));
      }
    }


#ifdef _PAIRCALC_DEBUG_
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){
      m = matrix1[index + j + i*S];
      if(m!=amatrix[i*grainSize+j]){CkPrintf("copy broken in back path: %2.5g != %2.5g \n", m, amatrix[i*grainSize+j]);}
    }
  }
#endif //_PAIRCALC_DEBUG_

  int m_in=grainSize;
  int n_in=N*2;
  int k_in=grainSize;
  double *mynewDatad= reinterpret_cast <double *> (mynewData);
  double alphad(1.0);
  double betad(0.0);
  char transform='N';
  char transformT='T';

#ifdef _PAIRCALC_DEBUG_PARANOID_
  char filename[80];
  sprintf(filename, "bwlmodata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
  FILE *outfile = fopen(filename, "w");


  fprintf(outfile,"[%d,%d,%d,%d,%d] LM\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
  for(int i=0;i<N*grainSize;i++)
    {
      fprintf(outfile," %g %g",inDataLeft[i]);
      fprintf(outfile,"\n");
    }
  fclose(outfile);
  sprintf(filename, "bwmdata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
  FILE *moutfile = fopen(filename, "w");
  fprintf(moutfile,"[%d,%d,%d,%d,%d] amatrix\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
  for (int i = 0; i < grainSize; i++) {
    for (int j = 0; j < grainSize; j++){ 
      fprintf(moutfile,"%.10g\n",amatrix[i*grainSize+j]);
    }
  }
  fclose(moutfile);
#endif
#ifndef CMK_OPTIMIZE
  double StartTime=CmiWallTimer();
#endif
  if(symmetric)
    DGEMM(&transform, &transform, &n_in, &m_in, &k_in, &alphad, inDataLeft, &n_in,  amatrix, &k_in, &betad, mynewDatad, &n_in);
  else
    DGEMM(&transform, &transformT, &n_in, &m_in, &k_in, &alphad, inDataLeft, &n_in,  amatrix, &k_in, &betad, mynewDatad, &n_in);

#ifndef CMK_OPTIMIZE
  traceUserBracketEvent(230, StartTime, CmiWallTimer());
#endif

  if(symmetric && (thisIndex.x !=thisIndex.y) && existsRight)
    {
#ifndef CMK_OPTIMIZE
      StartTime=CmiWallTimer();
#endif
      double *othernewDatad= reinterpret_cast <double *> (othernewData);
      //this isn't right
      DGEMM(&transform, &transformT, &n_in, &m_in, &k_in, &alphad, inDataRight, &n_in,  amatrix, &k_in, &betad, othernewDatad, &n_in);
#ifndef CMK_OPTIMIZE
      traceUserBracketEvent(250, StartTime, CmiWallTimer());
#endif
    }


#ifdef _PAIRCALC_DEBUG_PARANOID_
  sprintf(filename, "bwgmodata.%d_%d_%d_%d_%d", thisIndex.w,thisIndex.x, thisIndex.y,thisIndex.z, symmetric);
  FILE *ooutfile = fopen(filename, "w");


  fprintf(ooutfile,"[%d,%d,%d,%d,%d] outData=C\n",thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric);
  for(int i=0;i<N*grainSize;i++)
    {
      fprintf(ooutfile," %g %g",mynewData[i].re,mynewData[i].im);
      fprintf(ooutfile,"\n");
    }
  fclose(ooutfile);
#endif


  if(!unitcoef){ // CG non minimization case
    // FIXME: we need to figure out section arrangement for this
    if(S!=grainSize)
      for(int i=0;i<grainSize;i++){
	localMatrix = (matrix2+index+i*S);
	outMatrix   = amatrix+i*grainSize;
	memcpy(outMatrix,localMatrix,grainSize*sizeof(double));
      }
    else
      amatrix=matrix2+index;

#ifdef _PAIRCALC_DEBUG_
    for (int i = 0; i < grainSize; i++) {
      for (int j = 0; j < grainSize; j++){
	m = matrix2[index + j + i*S];

	if(m!=amatrix[i*grainSize+j]){CkPrintf("Dcopy broken in back path: %2.5g != %2.5g \n",
					       m, amatrix[i*grainSize+j]);}
      }
    }
#endif
    double *rightd=reinterpret_cast <double *> (inDataRight);
    // C = alpha*A*B + beta*C
#ifndef CMK_OPTIMIZE
    StartTime=CmiWallTimer();
#endif
    betad=1.0;
    DGEMM(&transform, &transformT, &n_in, &m_in, &k_in, &alphad, inDataRight, &n_in,  amatrix, &k_in, &betad, mynewDatad, &n_in);
#ifndef CMK_OPTIMIZE
    traceUserBracketEvent(240, StartTime, CmiWallTimer());
#endif
  } // end  CG case

  if(makeLocalCopy)
    delete [] amatrix;
  // else we didn't allocate one

#ifdef _PAIRCALC_VALID_OUT_
  CkPrintf("[PAIRCALC] [%d %d %d %d %d] backward gemm out %.10g %.10g %.10g %.10g \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,symmetric, mynewDatad[0],mynewDatad[1],mynewData[N*grainSize-1].re,mynewData[N*grainSize-1].im);
#endif

  //original version
#ifndef _PAIRCALC_SECONDPHASE_LOADBAL_
  if(!symmetric){
    CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
  }
  else {
    CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
    thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);

    if (thisIndex.y != thisIndex.x){ 
      CkArrayIndex4D idxo(thisIndex.w, 0, thisIndex.x, thisIndex.z);
      thisProxy(idxo).sumPartialResult(N*grainSize, othernewData, thisIndex.z);
    }

  }
#else
  // partition the result and send it in segments for better balancing

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
  else if (gspacesum){
    // we just send these to gspace where acceptNewPsi will put them together
    // this should be fairly load balanced as long as gspace is balanced
    /*
      CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
      thisProxy(idx).sumPartialResult(N*grainSize, mynewData, thisIndex.z);
    */

    //we have an N*grainsize block
    //we must communicate with all planes
    int offset=thisIndex.z;  //zero for now but should be considered
    int priority=0xFFFFFFFF;
    if (thisIndex.y != thisIndex.x)
      { 
	for(int j=0;j<grainSize;j++)
	  {

	    //	CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j+offset, thisIndex.w), cb_aid);

	    CkCallback mycb(cb_ep, CkArrayIndex2D(j+thisIndex.x ,thisIndex.w), cb_aid);
	    partialResultMsg *msg = new (N, 8*sizeof(int) )partialResultMsg;
	    msg->N=N;
	    msg->myoffset = j; // someplace in the state
	    memcpy(msg->result,othernewData+j*N,msg->N*sizeof(complex));
	    *((int*)CkPriorityPtr(msg)) = priority;
	    CkSetQueueing(msg, CK_QUEUEING_IFIFO);
#ifdef _PAIRCALC_DEBUG_
	    CkPrintf("sending partial of size %d offset %d to [%d %d]\n",N,j,thisIndex.y+j,thisIndex.w);
#endif
	    mycb.send(msg);

	  }
      }
    for(int j=0;j<grainSize;j++)
      {

	//	CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j+offset, thisIndex.w), cb_aid);

	CkCallback mycb(cb_ep, CkArrayIndex2D(j+thisIndex.y ,thisIndex.w), cb_aid);
	partialResultMsg *msg = new (N, 8*sizeof(int) )partialResultMsg;
	msg->N=N;
	msg->myoffset = j; // someplace in the state
	memcpy(msg->result,mynewData+j*N,msg->N*sizeof(complex));
	*((int*)CkPriorityPtr(msg)) = priority;
	CkSetQueueing(msg, CK_QUEUEING_IFIFO);
#ifdef _PAIRCALC_DEBUG_
	CkPrintf("sending partial of size %d offset %d to [%d %d]\n",N,j,thisIndex.y+j,thisIndex.w);
#endif
	mycb.send(msg);

      }

  }
  else //sum in paircalc
    {
      CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.y, thisIndex.z);
      partialResultMsg *msg = new (N*grainSize, 8*sizeof(int) )partialResultMsg;
      msg->N=N*grainSize;
      msg->myoffset = thisIndex.z;
      memcpy(msg->result,mynewData,msg->N*sizeof(complex));
      msg->cb= cb;
      *((int*)CkPriorityPtr(msg)) = priority;
      CkSetQueueing(msg, CK_QUEUEING_IFIFO);
      thisProxy(idx).sumPartialResult(msg);
      if (thisIndex.y != thisIndex.x){   // FIXME: rowNum will alway == thisIndex.x
	CkArrayIndex4D idx(thisIndex.w, 0, thisIndex.x, thisIndex.z);
	partialResultMsg *msg = new (N*grainSize, 8*sizeof(int) )partialResultMsg;
	msg->N=N*grainSize;
	msg->myoffset = thisIndex.z;
	memcpy(msg->result,othernewData,msg->N*sizeof(complex));
	msg->cb= cb;
	*((int*)CkPriorityPtr(msg)) = priority;
	CkSetQueueing(msg, CK_QUEUEING_IFIFO);
	thisProxy(idx).sumPartialResult(msg);
      }
    }
#endif

  delete [] mynewData;

  if(symmetric && thisIndex.x != thisIndex.y)
    delete [] othernewData;


  if(conserveMemory)
    {
      // clear the right and left they'll get reallocated on the next pass
      delete [] inDataLeft;
      inDataLeft=NULL;
      if(!symmetric || (symmetric&&thisIndex.x!=thisIndex.y)) {
	delete [] inDataRight;
	inDataRight = NULL;
      }
      existsLeft=false;
      existsRight=false;
      if(outData!=NULL)
	{
	  delete [] outData;
	  outData = NULL;
	  existsOut=false;
	}
    }

}

void
PairCalculator::sumPartialResult(partialResultMsg *msg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d %d]: sum result from grain %d  count %d\n", thisIndex.w,thisIndex.x,thisIndex.y,thisIndex.z,symmetric,msg->N, sumPartialCount);
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
  CkPrintf("[%d %d %d %d %d]: sum result from %d, count %d\n",  thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, symmetric, offset, sumPartialCount);
#endif

  sumPartialCount++;

  if(!existsNew){
    CkAssert(newData==NULL);
    newData = new complex[N*grainSize];
    newelems=N*grainSize;
    memset(newData,0,N*grainSize*sizeof(complex));
    existsNew=true;
  }
  CkAssert(size<=N*grainSize);
  for(int i=0; i<size; i++){
    newData[i].re += result[i].re;
    newData[i].im += result[i].im;
  }
  int countExpect = (S/grainSize)*blkSize;
  //  if(symmetric) 
  //    countExpect = 2*(S/grainSize-1)+1;
  //  if(symmetric) countExpect = thisIndex.y/grainSize + 1;

#ifdef _PAIRCALC_DEBUG_
  CkPrintf("[%d %d %d %d %d]: Have %d of Expected %d messages\n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z, symmetric, sumPartialCount,countExpect);

#endif
  if (sumPartialCount == countExpect) {

#ifndef _PAIRCALC_SECONDPHASE_LOADBAL_
    for(int j=0; j<grainSize; j++){
      CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j, thisIndex.w), cb_aid);
      mySendMsg *msg = new (N, 0)mySendMsg; // msg with newData (size N)
      memcpy(msg->data, newData+j*N, N * sizeof(complex));
      msg->N=N;
#ifdef _PAIRCALC_VALID_OUT_
      CkPrintf("[PAIRCALC] [%d %d %d %d %d] bw ps to %d %d %.10g %.10g %.10g %.10g \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,symmetric, thisIndex.y+j+offset, thisIndex.x,msg->data[0].re,msg->data[0].im,msg->data[N-1].re,msg->data[N-1].im);
#endif
      mycb.send(msg);
    }
#else //load balance output
    if(!symmetric)
	for(int j=0; j<grainSize/(S/grainSize); j++){
	    CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j+offset, thisIndex.w), cb_aid);
	    mySendMsg *msg = new (N, 0) mySendMsg; // msg with newData (size N)
	    memcpy(msg->data, newData+j*N, N * sizeof(complex));
	    msg->N=N;
#ifdef _PAIRCALC_VALID_OUT_
	    CkPrintf("[PAIRCALC] [%d %d %d %d %d] bw ps to %d %d %.10g %.10g %.10g %.10g \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,symmetric, thisIndex.y+j+offset, thisIndex.x,msg->data[0].re,msg->data[0].im,msg->data[N-1].re,msg->data[N-1].im);
#endif
	    mycb.send(msg);
	}
    else //shouldn't be in here as we sent it directly to gspace
      	for(int j=0; j<grainSize; j++){
	    CkCallback mycb(cb_ep, CkArrayIndex2D(thisIndex.y+j+offset, thisIndex.w), cb_aid); 
	    mySendMsg *msg = new (N, 0)mySendMsg; // msg with newData (size N)
	    memcpy(msg->data, newData+j*N, N * sizeof(complex));
	    msg->N=N;

#ifdef _PAIRCALC_VALID_OUT_
	    CkPrintf("[PAIRCALC] [%d %d %d %d %d] bw ps to %d %d %.10g %.10g %.10g %.10g \n", thisIndex.w, thisIndex.x, thisIndex.y, thisIndex.z,symmetric, thisIndex.y+j+offset, thisIndex.x,msg->data[0].re,msg->data[0].im,msg->data[N-1].re,msg->data[N-1].im);
#endif
	    mycb.send(msg);
	}

#endif //end load balancing

    sumPartialCount = 0;
    memset(newData,0,N*grainSize*sizeof(complex));
  } // expected arrived
}

// EJB: Shouldn't this be inlined?
inline void add_double(void *in, void *inout, int *red_size, void *handle) {
    double * matrix1 = (double *)in;
    double * matrix2 = (double *)inout;
    int size = *red_size / sizeof(double);

    for(int i = 0; i < size; i ++){
        matrix2[i] += matrix1[i];
    }
    return;
}


#if CONVERSE_VERSION_ELAN

typedef void (* ELAN_REDUCER)(void *in, void *inout, int *count, void *handle);

extern "C" void elan_machine_reduce(int nelem, int size, void * data,
                                    void *dest, ELAN_REDUCER fn, int root);
extern "C" void elan_machine_allreduce(int nelem, int size, void * data,
                                       void *dest, ELAN_REDUCER fn);
#endif

void PairCalcReducer::acceptContribute(int size, double* matrix, CkCallback cb,
                                       bool isAllReduce, bool symmtype, int offx, int offy, int grainSize)
{
    this->isAllReduce = isAllReduce;
    this->size = size;
    this->symmtype = symmtype;
    this->cb = cb;

#if CONVERSE_VERSION_ELAN
    reduction_elementCount ++;
#ifdef _ELAN_PAIRCALC_DEBUG_
    CkPrintf("Accept contribute called on %d count is %d out of %d\n",CkMyPe(),reduction_elementCount, localElements[symmtype].length());
#endif
    int red_size = size *sizeof(double);
    if(tmp_matrix == NULL) {
        tmp_matrix = new double[size];
	memset(tmp_matrix,0,red_size);
    }
    int off=offx*grainSize+offy;
    for(int i = 0; i < grainSize*grainSize ; i ++){
        tmp_matrix[i+off] += matrix[i];
    }

    if(reduction_elementCount == localElements[symmtype].length()) {

#ifdef _ELAN_PAIRCALC_DEBUG_
    CkPrintf("[%d] Contributing %d \n",CkMyPe(),reduction_elementCount);

#endif
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
#ifdef _ELAN_PAIRCALC_DEBUG_
    CkPrintf("Starting machine reduction %d\n",CkMyPe());
#endif
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
    delete [] tmp_matrix;
    tmp_matrix = NULL;
#ifdef _ELAN_PAIRCALC_DEBUG_
    CkPrintf("Leaving machine reduction %d\n",CkMyPe());
#endif
#else
    CkAbort("Converse Version Is not ELAN, h/w reduction is not supported");
#endif


}

void
PairCalcReducer::broadcastEntireResult(entireResultMsg *imsg)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("bcasteres:On Pe %d -- %d objects\n", CkMyPe(), localElements[imsg->symmetric].length());
#endif
  acceptResultMsg *msg= new (size,0) acceptResultMsg;
  msg->size=imsg->size;
  memcpy(msg->matrix,imsg->matrix,imsg->size* sizeof(double));
  for (int i = 0; i < localElements[imsg->symmetric].length(); i++)
    {
#ifdef _PAIRCALC_DEBUG_
      CkPrintf("call accept on :");
      (localElements[imsg->symmetric])[i].dump();
#endif
      CkSendMsgArrayInline(CkIndex_PairCalculator::__idx_acceptResult_acceptResultMsg, msg, (localElements[imsg->symmetric])[i].id, (localElements[imsg->symmetric])[i].idx, CK_MSG_KEEP);

    }
  delete imsg;
}


void
PairCalcReducer::broadcastEntireResult(int size, double* matrix, bool symmtype)
{
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("bcasteres:On Pe %d -- %d objects\n", CkMyPe(), localElements[symmtype].length());
#endif
  acceptResultMsg *msg= new (size,0) acceptResultMsg;
  msg->size=size;
  memcpy(msg->matrix,matrix,size* sizeof(double));
  for (int i = 0; i < localElements[symmtype].length(); i++)
    {
#ifdef _PAIRCALC_DEBUG_
      CkPrintf("call accept on :");
      (localElements[symmtype])[i].dump();
#endif
      CkSendMsgArrayInline(CkIndex_PairCalculator::__idx_acceptResult_acceptResultMsg, msg, (localElements[symmtype])[i].id, (localElements[symmtype])[i].idx, CK_MSG_KEEP);

    }
  
  //  delete msg;


}

void
PairCalcReducer::broadcastEntireResult(int size, double* matrix1, double* matrix2, bool symmtype){
    CkPrintf("On Pe %d -- %d objects\n", CkMyPe(), localElements[symmtype].length());
    acceptResultMsg2 *msg= new (size,size,0) acceptResultMsg2;
    msg->size=size;
    memcpy(msg->matrix1,matrix1,size* sizeof(double));
    memcpy(msg->matrix2,matrix2,size* sizeof(double));
    for (int i = 0; i < localElements[symmtype].length(); i++)
      {
	CkSendMsgArrayInline(CkIndex_PairCalculator::__idx_acceptResult_acceptResultMsg2, msg, (localElements[symmtype])[0].id, (localElements[symmtype])[i].idx,CK_MSG_KEEP);
      }
    //    delete msg;

}

void
PairCalcReducer::broadcastEntireResult(entireResultMsg2 *imsg)
{
    CkPrintf("On Pe %d -- %d objects\n", CkMyPe(), localElements[imsg->symmetric].length());
    acceptResultMsg2 *msg= new (imsg->size,imsg->size,0) acceptResultMsg2;
    msg->size=imsg->size;
    memcpy(msg->matrix1,imsg->matrix1,imsg->size* sizeof(double));
    memcpy(msg->matrix2,imsg->matrix2,imsg->size* sizeof(double));
    for (int i = 0; i < localElements[imsg->symmetric].length(); i++)
      {
	CkSendMsgArrayInline(CkIndex_PairCalculator::__idx_acceptResult_acceptResultMsg2, msg, (localElements[imsg->symmetric])[0].id, (localElements[imsg->symmetric])[i].idx,CK_MSG_KEEP);
      }
    delete imsg;

}




/* note this is probably broken now, offset calculations need to be rejiggered to work with the one dimensional allocation scheme*/
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
		  outData[leftoffset1 * grainSize + leftoffset2] =
		    compute_entry(size, &(inDataLeft[leftoffset1*N]), &(inDataLeft[leftoffset2*N]),op1);
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
		outData[leftoffset * grainSize + rightoffset] =
		    compute_entry(size, &(inDataLeft[leftoffset*N]), &(inDataRight[rightoffset*N]),op1);
	      }
	  }

        // OLD left compute with every NEW entry in right
	for(int kLeft=0; kLeft<kLeftDoneCount; kLeft++)
	  {
	    leftoffset = kLeftOffset[kLeft];
	    for(int kRight=kRightDoneCount; kRight<kRightDoneCount + kRightReady; kRight++)
	      {
	       rightoffset = kRightOffset[kRight];
		outData[leftoffset * grainSize + rightoffset] =
		    compute_entry(size, &(inDataLeft[leftoffset*N]), &(inDataRight[rightoffset*N]),op1);
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

	      for (int i = 0; i < grainSize*grainSize; i++)
		  outData[i] *= 2.0;
	  }
      }
      sparseRed.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
      sparseRed.contribute(this, sparse_sum_double);

#if CONVERSE_VERSION_ELAN
      //CkPrintf("[%d] ELAN VERSION %d\n", CkMyPe(), symmetric);
      CProxy_PairCalcReducer pairCalcReducerProxy(reducer_id);
      pairCalcReducerProxy.ckLocalBranch()->acceptContribute(S * S, outData,
                                                             cb, !symmetric, symmetric);
#endif

  }

#else

// Below is the old version, very dusty
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

    sparseRed.add((int)thisIndex.y, (int)thisIndex.x, (int)(thisIndex.y+grainSize-1), (int)(thisIndex.x+grainSize-1), outData);
    sparseRed.contribute(this, sparse_sum_double);
  }
#endif

#endif //NOGEMM
}



#include "ckPairCalculator.def.h"


