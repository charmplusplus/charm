
#include "ckPairCalculator.h"
#include "pairCalculator.h"

void createPairCalculator(bool sym, int s, int grainSize, int numZ, int* z, int op1, 
                          FuncType f1, int op2, FuncType f2, CkCallback cb, 
                          PairCalcID* pcid, int cb_ep, CkArrayID cb_aid, 
                          int comlib_flag, CkGroupID *mapid, int flag_dp) {

  //CkPrintf("create pair calculator %d, %d\n", s, grainSize);

  CProxy_PairCalcReducer pairCalcReducerProxy = CProxy_PairCalcReducer::ckNew();
  pairCalcReducerProxy.ckSetReductionClient(&cb); 

  // FIXME: to choose grain size and block size instead of hard-coding
  int blkSize = 1;
  
  CkArrayOptions options;
  CProxy_PairCalculator pairCalculatorProxy;
  if(!mapid) {
      pairCalculatorProxy = CProxy_PairCalculator::ckNew();
  }
  else {
      options.setMap(*mapid);
      pairCalculatorProxy = CProxy_PairCalculator::ckNew(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, options);
  }

  pairCalculatorProxy.ckSetReductionClient(&cb);  

  int proc = 0, n_paircalc = 0;
  /*
  Strategy * pstrat = new PipeBroadcastStrategy(USE_HYPERCUBE, pairCalculatorProxy.ckGetArrayID());
  */
  Strategy *bstrat = new BroadcastStrategy(USE_TREE);

  ComlibInstanceHandle bcastInstance = CkGetComlibInstance();
  bcastInstance.setStrategy(bstrat);

  pcid->Init(pairCalculatorProxy.ckGetArrayID(), pairCalcReducerProxy.ckGetGroupID(), grainSize, blkSize, s, sym, comlib_flag, bcastInstance._instid, bcastInstance._dmid, flag_dp);
  
  
  if(mapid) {
      if(sym){
          for(int numX = 0; numX < numZ; numX += blkSize){
              for (int s1 = 0; s1 < s; s1 += grainSize) {
                  for (int s2 = s1; s2 < s; s2 += grainSize) {
                      for (int c = 0; c < blkSize; c++) {
                          pairCalculatorProxy(CkArrayIndexIndex4D(z[numX],s1,s2,c)).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep);
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
                          pairCalculatorProxy(CkArrayIndexIndex4D(z[numX],s1,s2,c)).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep);
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
                          pairCalculatorProxy(CkArrayIndexIndex4D(z[numX],s1,s2,c)).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, proc);
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
                          pairCalculatorProxy(CkArrayIndexIndex4D(z[numX],s1,s2,c)).
                              insert(sym, grainSize, s, blkSize, op1, f1, op2, f2, cb, pairCalcReducerProxy.ckGetGroupID(), cb_aid, cb_ep, proc);
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

#ifdef _DEBUG_
  CkPrintf("    Finished init {grain=%d, sym=%d, blk=%d, Z=%d, S=%d}\n", grainSize, sym, blkSize, numZ, s);
#endif
}

// Deposit data and start calculation
void startPairCalcLeft(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){
#ifdef _DEBUG_
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

  x = myZ;
  s1 = (myS/grainSize) * grainSize;
  if(symmetric){
    for (c = 0; c < blkSize; c++)
      for(s2 = 0; s2 < S; s2 += grainSize){
	if(s1 <= s2)
	  pairCalculatorProxy(CkArrayIndexIndex4D(x, s1, s2, c)).calculatePairs(n, ptr, myS, true, flag_dp);
	else
	  pairCalculatorProxy(CkArrayIndexIndex4D(x, s2, s1, c)).calculatePairs(n, ptr, myS, false, flag_dp);
      }
  }
  else {
    for (c = 0; c < blkSize; c++)
      for(s2 = 0; s2 < S; s2 += grainSize){
	pairCalculatorProxy(CkArrayIndexIndex4D(x, s1, s2, c)).calculatePairs(n, ptr, myS, true, flag_dp);
      }
  }
}

void startPairCalcRight(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){
#ifdef _DEBUG_
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
  for (c = 0; c < blkSize; c++)
    for(s1 = 0; s1 < S; s1 += grainSize){
      pairCalculatorProxy(CkArrayIndexIndex4D(x, s1, s2, c)).calculatePairs(n, ptr, myS, false, flag_dp);
    }
}

void finishPairCalc(PairCalcID* pcid, int n, double *ptr) {
#ifdef _DEBUG_
  CkPrintf("     Calc Finish\n");
#endif

  CkGroupID pairCalcReducerID = (CkArrayID)pcid->Gid; 
  CProxy_PairCalcReducer pairCalcReducerProxy(pairCalcReducerID);

  ComlibInstanceHandle bcastInstance = ComlibInstanceHandle(pcid->instid, pcid->dmid);

  if(pcid->useComlib) {
      ComlibDelegateProxy(&pairCalcReducerProxy);
      bcastInstance.beginIteration();
  }

  pairCalcReducerProxy.broadcastEntireResult(n, ptr, pcid->Symmetric);
  
  if(pcid->useComlib) {
      bcastInstance.endIteration();
  }
}

void startPairCalcLeftAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){
    
}

void startPairCalcRightAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ){

}
