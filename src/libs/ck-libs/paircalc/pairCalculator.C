
#include "ckPairCalculator.h"
#include "pairCalculator.h"
extern ComlibInstanceHandle mcastInstanceCP;


void createPairCalculator(bool sym, int s, int grainSize, int numZ, int* z, int op1, 
                          FuncType f1, int op2, FuncType f2, CkCallback cb, 
                          PairCalcID* pcid, int cb_ep, CkArrayID cb_aid, 
                          int comlib_flag, CkGroupID *mapid, int flag_dp, bool conserveMemory, bool lbpaircalc, CkCallback lbcb, CkGroupID mCastGrpId, bool gspacesum) {

  traceRegisterUserEvent("calcpairDGEMM", 210);
  traceRegisterUserEvent("calcpairContrib", 220);
  traceRegisterUserEvent("acceptResultDGEMM1", 230);
  traceRegisterUserEvent("acceptResultDGEMM2", 240);

  //CkPrintf("create pair calculator %d, %d\n", s, grainSize);

  CProxy_PairCalcReducer pairCalcReducerProxy = CProxy_PairCalcReducer::ckNew();

  CkCallback rcb = CkCallback(CkIndex_PairCalcReducer::__idx_startMachineReduction_void,
                              pairCalcReducerProxy.ckGetGroupID());
  
  pairCalcReducerProxy.ckSetReductionClient(&rcb); 

  // FIXME: to choose grain size and block size instead of hard-coding
  int blkSize = 1;
  
  CkArrayOptions options;
  CProxy_PairCalculator pairCalculatorProxy;
#ifdef CONVERSE_VERSION_ELAN
  bool machreduce=(s/grainSize * numZ* blkSize>=CkNumNodes()) ? true: false;
#else
  bool machreduce=false;
#endif  

  if(!mapid) {
      pairCalculatorProxy = CProxy_PairCalculator::ckNew();
  }
  else {
      options.setMap(*mapid);
      pairCalculatorProxy = CProxy_PairCalculator::ckNew(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, conserveMemory, lbpaircalc, lbcb, machreduce, gspacesum, options);
  }

  pairCalculatorProxy.ckSetReductionClient(&cb);  

  int proc = 0, n_paircalc = 0;
  /*
    CharmStrategy * pstrat = new PipeBroadcastStrategy(USE_HYPERCUBE, pairCalculatorProxy.ckGetArrayID());
  */
  CharmStrategy *bstrat = new BroadcastStrategy(USE_HYPERCUBE);

  ComlibInstanceHandle bcastInstance = CkGetComlibInstance();
  bcastInstance.setStrategy(bstrat);
  CharmStrategy *multistrat = new DirectMulticastStrategy(pairCalculatorProxy.ckGetArrayID());

  mcastInstanceCP = CkGetComlibInstance();
  mcastInstanceCP.setStrategy(multistrat);
  pcid->Init(pairCalculatorProxy.ckGetArrayID(), pairCalcReducerProxy.ckGetGroupID(), grainSize, blkSize, s, sym, comlib_flag, bcastInstance, flag_dp, conserveMemory, lbpaircalc, mcastInstanceCP, mCastGrpId, gspacesum);
  if(mapid) {
    if(sym){ 
          for(int numX = 0; numX < numZ; numX += blkSize){
              for (int s1 = 0; s1 < s; s1 += grainSize) {
                  for (int s2 = s1; s2 < s; s2 += grainSize) {
                      for (int c = 0; c < blkSize; c++) {
                          pairCalculatorProxy(z[numX],s1,s2,c).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, conserveMemory, lbpaircalc, lbcb,machreduce, gspacesum );
                          n_paircalc++;
                      }
                  }
              }
          }
      }
      else {   // for non-symmetric
          for(int numX = 0; numX < numZ; numX += blkSize){
              for (int s1 = 0; s1 < s; s1 += grainSize) {
                  for (int s2 = 0; s2 < s; s2 += grainSize) {
                      for (int c = 0; c < blkSize; c++) {
                          pairCalculatorProxy(z[numX],s1,s2,c).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, conserveMemory, lbpaircalc,lbcb,machreduce, gspacesum );
                          n_paircalc++;
                          proc++;
                          if (proc >= CkNumPes()) proc = 0;
                      }
                  }
              }
          }
      }   
      
  }
  else {
      if(sym){
          for(int numX = 0; numX < numZ; numX += blkSize){
              for (int s1 = 0; s1 < s; s1 += grainSize) {
                  for (int s2 = s1; s2 < s; s2 += grainSize) {
                      for (int c = 0; c < blkSize; c++) {
                          pairCalculatorProxy(z[numX],s1,s2,c).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, conserveMemory, lbpaircalc, lbcb, machreduce, gspacesum, proc);
                          n_paircalc++;
                          proc++;
                          if (proc >= CkNumPes()) proc = 0;
                      }
                  }
              }
          }
      }
      else {   // for non-symmetric
          for(int numX = 0; numX < numZ; numX += blkSize){
              for (int s1 = 0; s1 < s; s1 += grainSize) {
                  for (int s2 = 0; s2 < s; s2 += grainSize) {
                      for (int c = 0; c < blkSize; c++) {
                          pairCalculatorProxy(z[numX],s1,s2,c).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, conserveMemory, lbpaircalc, lbcb,  machreduce, gspacesum, proc);
                          n_paircalc++;
                          proc++;
                          if (proc >= CkNumPes()) proc = 0;
                      }
                  }
              }
          }
      }   
  }

  pairCalculatorProxy.doneInserting();

#ifdef _PAIRCALC_DEBUG_
  CkPrintf("    Finished init {grain=%d, sym=%d, blk=%d, Z=%d, S=%d}\n", grainSize, sym, blkSize, numZ, s);
#endif
}

// Deposit data and start calculation
void startPairCalcLeft(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     Calc Left ptr %d\n", ptr);
#endif
  CkArrayID pairCalculatorID = (CkArrayID)pcid->Aid; 
  CProxy_PairCalculator pairCalculatorProxy(pairCalculatorID);
  int s1, s2, x, c;
  int grainSize = pcid->GrainSize;
  int blkSize =  pcid->BlkSize;
  int S = pcid->S;
  int symmetric = pcid->Symmetric;
  bool flag_dp = pcid->isDoublePacked;

  bool conserveMemory = pcid->conserveMemory;
  x = myZ;
  s1 = (myS/grainSize) * grainSize;

  int numElems;
  //create multicast proxy array section list 
  if(!(pcid->existsLproxy||pcid->existsLNotFromproxy)){
    if(symmetric){
      CkArrayIndexMax *elems=new CkArrayIndexMax[blkSize*S/grainSize];
      CkArrayIndexMax *elemsfromrow=new CkArrayIndexMax[blkSize*S/grainSize];
      // 1 proxy for left and 1 for right
      int erowcount=0;
      int ecount=0;
      CkArrayIndex4D idx(x,0,0,0);
      for (c = 0; c < blkSize; c++)
	for(s2 = 0; s2 < S; s2 += grainSize){
	  if(s1 <= s2)
	    {
	      idx.index[1]=s1;
	      idx.index[2]=s2;
	      idx.index[3]=c;
	      elemsfromrow[erowcount++]=idx;
	    }
	  else // swap s1 : s2
	    {
	      idx.index[1]=s2;
	      idx.index[2]=s1;
	      idx.index[3]=c;
	      elems[ecount++]=idx;
	    }
	}
      if(ecount)
	{

	  pcid->proxyLNotFrom = CProxySection_PairCalculator::ckNew(pairCalculatorID, elems, ecount); 
	  pcid->existsLNotFromproxy=true;	  
	  if(pcid->useComlib && _PC_COMMLIB_MULTI_)
	    {
	      ComlibDelegateProxy(&(pcid->proxyLNotFrom));
	      ComlibInitSectionID(pcid->proxyLNotFrom.ckGetSectionID());
	    }
	  else
	    {
	      CkMulticastMgr *mcastGrp = CProxy_CkMulticastMgr(pcid->mCastGrpId).ckLocalBranch();       
	      pcid->proxyLNotFrom.ckSectionDelegate(mcastGrp);
	    }
	}
      if(erowcount)
	{
	  pcid->proxyLFrom  = CProxySection_PairCalculator::ckNew(pairCalculatorID, elemsfromrow, erowcount); 
	  pcid->existsLproxy=true;	  
	  if(pcid->useComlib && _PC_COMMLIB_MULTI_)
	    {
	      ComlibDelegateProxy(&(pcid->proxyLFrom));
	      ComlibInitSectionID(pcid->proxyLFrom.ckGetSectionID());
	    }
	  else
	    {
	      CkMulticastMgr *mcastGrp = CProxy_CkMulticastMgr(pcid->mCastGrpId).ckLocalBranch(); 
	      pcid->proxyLFrom.ckSectionDelegate(mcastGrp);

	    }
	}
      delete [] elemsfromrow;
      delete [] elems;
    }
    else { //just make left here, right will be taken care of in startRight
#ifdef _PAIRCALC_DEBUG_
      CkPrintf("initializing multicast proxy in %d %d \n",x,s1);
#endif
      int erowcount=0;
      CkArrayIndexMax *elemsfromrow=new CkArrayIndexMax[blkSize*S/grainSize];
      CkArrayIndex4D idx(x,s1,0,0);
      for (c = 0; c < blkSize; c++)
	for(s2 = 0; s2 < S; s2 += grainSize){
	  idx.index[2]=s2;
	  idx.index[3]=c;
	  elemsfromrow[erowcount++]=idx;
#ifdef _PAIRCALC_DEBUG_
	  CkPrintf("Add Section ID %d: %d,%d,%d,%d \n",erowcount, idx.index[0],idx.index[1],idx.index[2],idx.index[3]);
#endif
	}
      CkAssert(erowcount>0);
#ifdef _PAIRCALC_DEBUG_
      for(int count = 0; count < erowcount; count ++) {
	CkArrayIndex4D idx4d;

	idx4d.nInts = elemsfromrow[count].nInts;
	idx4d.data()[0] = elemsfromrow[count].data()[0];
	idx4d.data()[1] = elemsfromrow[count].data()[1];
	
	CkPrintf("DEBUG ID %d: %d,%d,%d,%d \n",count, idx4d.index[0], idx4d.index[1], idx4d.index[2], idx4d.index[3]);          
      }
#endif      
      pcid->proxyLFrom = CProxySection_PairCalculator::ckNew(pairCalculatorID, elemsfromrow, erowcount); 
      pcid->existsLproxy=true;      
      if(pcid->useComlib &&_PC_COMMLIB_MULTI_)
	{
	  ComlibDelegateProxy(&(pcid->proxyLFrom));
	  ComlibInitSectionID(pcid->proxyLFrom.ckGetSectionID());
	}
      else
	{
	  CkMulticastMgr *mcastGrp = CProxy_CkMulticastMgr(pcid->mCastGrpId).ckLocalBranch(); 
	  pcid->proxyLFrom.ckSectionDelegate(mcastGrp);
	}
      delete [] elemsfromrow;
    }

  }

  //use proxy to send
  if(pcid->useComlib && _PC_COMMLIB_MULTI_) {
#ifdef _PAIRCALC_DEBUG_
    CkPrintf("%d Calling Begin Iteration\n", CkMyPe());
#endif
    pcid->minst.beginIteration();
  }
#ifdef _PAIRCALC_DEBUG_PARANOID_
  double re;
  double im;
  for(int i=0;i<n;i++)
    {
      re=ptr[i].re;
      im=ptr[i].im;
      if(fabs(re)>0.0)
	CkAssert(fabs(re)>1.0e-300);
      if(fabs(im)>0.0)
	CkAssert(fabs(im)>1.0e-300);
    }
#endif
  if(pcid->existsLproxy)
    {
      calculatePairsMsg *msgfromrow=new ( n,0 ) calculatePairsMsg;
      msgfromrow->init(n, myS, true, flag_dp, ptr);
      pcid->proxyLFrom.calculatePairs_gemm(msgfromrow);
    }
  if(pcid->existsLNotFromproxy)
    { //symmetric
      calculatePairsMsg *msg= new ( n,0 ) calculatePairsMsg;
      msg->init(n, myS, false, flag_dp, ptr);   
      pcid->proxyLNotFrom.calculatePairs_gemm(msg);
    }

#ifdef _PAIRCALC_DEBUG_
  CkPrintf("Send from [%d %d]\n",x,s1);
#endif

  if(pcid->useComlib && _PC_COMMLIB_MULTI_) {
#ifdef _PAIRCALC_DEBUG_
    CkPrintf("%d Calling End Iteration\n", CkMyPe());
#endif
    pcid->minst.endIteration();
  }
}


void isAtSyncPairCalc(PairCalcID* pcid){
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     lbsync symm=%d\n", pcid->Symmetric);
#endif
  //nuke the register will rebuild in ResumeFromSync
  CkGroupID pairCalcReducerID = (CkArrayID)pcid->Gid; 
  CProxy_PairCalcReducer pairCalcReducerProxy(pairCalcReducerID);
  pairCalcReducerProxy.clearRegister();
  CkArrayID pairCalculatorID = (CkArrayID)pcid->Aid; 
  CProxy_PairCalculator pairCalculatorProxy(pairCalculatorID);
  pairCalculatorProxy.lbsync();
}

void startPairCalcRight(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     Calc Right symm=%d\n", pcid->Symmetric);
#endif
  CkArrayID pairCalculatorID = (CkArrayID)pcid->Aid; 
  CProxy_PairCalculator pairCalculatorProxy(pairCalculatorID);

  int s1, s2, x, c;
  int grainSize = pcid->GrainSize;
  int blkSize =  pcid->BlkSize;
  int S = pcid->S;
  bool symmetric = pcid->Symmetric;
  bool flag_dp = pcid->isDoublePacked;

  CkAssert(symmetric == false);
  
  x = myZ;
  s2 = (myS/grainSize) * grainSize;
  //create multicast proxy list 
  if(!pcid->existsRproxy)
    {
#ifdef _PAIRCALC_DEBUG_
      CkPrintf("initializing multicast proxy in %d %d \n",x,s2);
#endif
      int ecount=0;
      CkArrayIndexMax *elems=new CkArrayIndexMax[blkSize*S/grainSize];
      CkArrayIndex4D idx(x,0,s2,0);
      for (c = 0; c < blkSize; c++)
	for(s1 = 0; s1 < S; s1 += grainSize){

	  idx.index[1]=s1;
	  idx.index[3]=c;
	  elems[ecount++] = idx;
#ifdef _PAIRCALC_DEBUG_
	  CkPrintf("Add Section ID %d: %d,%d,%d,%d \n",ecount, idx.index[0],idx.index[1],idx.index[2],idx.index[3]);
#endif
	}

#ifdef _PAIRCALC_DEBUG_      
      for(int count = 0; count < ecount; count ++) {
	CkArrayIndex4D idx4d;
	idx4d.nInts = elems[count].nInts;
	idx4d.data()[0] = elems[count].data()[0];
	idx4d.data()[1] = elems[count].data()[1];
	
	CkPrintf("DEBUG ID %d: %d,%d,%d,%d \n",count, idx4d.index[0], idx4d.index[1], idx4d.index[2], idx4d.index[3]);          
      }
#endif
      if(ecount)
	{
	  pcid->proxyRNotFrom = CProxySection_PairCalculator::ckNew(pairCalculatorID, elems, ecount); 
	  pcid->existsRproxy=true;      
	  if(pcid->useComlib && _PC_COMMLIB_MULTI_)
	    {
	      ComlibDelegateProxy(&(pcid->proxyRNotFrom));
	      ComlibInitSectionID(pcid->proxyRNotFrom.ckGetSectionID());
	    }
	  else
	    {
	      CkMulticastMgr *mcastGrp = CProxy_CkMulticastMgr(pcid->mCastGrpId).ckLocalBranch(); 
	      pcid->proxyRNotFrom.ckSectionDelegate(mcastGrp);
	    }
	}
      delete [] elems;
    }
  //  calculatePairsMsg *msg= new ( n,0 ) calculatePairsMsg(n,myS,false,flag_dp,ptr);
  if(pcid->existsRproxy)
    {
#ifdef _DEBUG_PAIRCALC_PARANOID_
  double re;
  double im;
  for(int i=0;i<n;i++)
    {
      re=ptr[i].re;
      im=ptr[i].im;
      if(fabs(re)>0.0)
	CkAssert(fabs(re)>1.0e-300);
      if(fabs(im)>0.0)
	CkAssert(fabs(im)>1.0e-300);
    }
#endif
      calculatePairsMsg *msg= new ( n,0 ) calculatePairsMsg;
      msg->init(n,myS,false,flag_dp,ptr);
      if(pcid->useComlib & _PC_COMMLIB_MULTI_) {
#ifdef _PAIRCALC_DEBUG_
	CkPrintf("%d Calling Begin Iteration\n", CkMyPe());
#endif
	pcid->minst.beginIteration();
      }
      //mcastInstanceCP.beginIteration();
      pcid->proxyRNotFrom.calculatePairs_gemm(msg);

      if(pcid->useComlib && _PC_COMMLIB_MULTI_) {
#ifdef _PAIRCALC_DEBUG_
	CkPrintf("%d Calling End Iteration\n", CkMyPe());
#endif
	pcid->minst.endIteration();
      }
      
#ifdef _PAIRCALC_DEBUG_
      CkPrintf("Send from [%d 0 %d]\n",x,s2);
#endif
    }
}

void finishPairCalc(PairCalcID* pcid, int n, double *ptr) {
    finishPairCalc2(pcid, n, ptr, NULL);
}

void finishPairCalc2(PairCalcID* pcid, int n, double *ptr1, double *ptr2) {
#ifdef _PAIRCALC_DEBUG_
  CkPrintf("     Calc Finish 2\n");
#endif

  CkGroupID pairCalcReducerID = (CkArrayID)pcid->Gid; 
  CProxy_PairCalcReducer pairCalcReducerProxy(pairCalcReducerID);

  ComlibInstanceHandle bcastInstance = pcid->cinst;

  if(pcid->useComlib && _PC_COMMLIB_MULTI_) {
      ComlibDelegateProxy(&pairCalcReducerProxy);
      bcastInstance.beginIteration();
  }

  if(ptr2==NULL){
      pairCalcReducerProxy.broadcastEntireResult(n, ptr1, pcid->Symmetric);
  }
  else {
      pairCalcReducerProxy.broadcastEntireResult(n, ptr1, ptr2, pcid->Symmetric);
  }
  if(pcid->useComlib && _PC_COMMLIB_MULTI_) {
      bcastInstance.endIteration();
  }
}

void startPairCalcLeftAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){
    
}

void startPairCalcRightAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){

}
