
#ifndef _pairCalculator_h
#define _pairCalculator_h
#include "ckPairCalculator.h"


/* a place to keep the section proxies for the reduction */
class SProxP 
{
 public:
  CkArrayIndex3D idx;
  CProxySection_PairCalculator sectProxy;
  SProxP (CkArrayIndex3D _idx, CProxySection_PairCalculator _proxy) : idx(_idx), sectProxy(_proxy)
    {}
  SProxP () : idx(0,0,0)
    {}
};
PUPbytes(SProxP);
class PairCalcID {
 public:
  CkArrayID Aid;
  CkGroupID Gid;
  int GrainSize;
  int BlkSize;
  int S; 
  bool Symmetric;

  ComlibInstanceHandle cinst;
  ComlibInstanceHandle minst;

  bool useComlib;
  bool isDoublePacked;
  bool conserveMemory;
  bool lbpaircalc;
  bool existsLproxy;
  bool existsLNotFromproxy;
  bool existsRproxy;
  bool gspacesum;
  CProxySection_PairCalculator proxyLFrom;
  CProxySection_PairCalculator proxyLNotFrom;
  CProxySection_PairCalculator proxyRNotFrom;
  CkGroupID mCastGrpId;
  CkVec < SProxP > sections;
  PairCalcID() {}
  ~PairCalcID() {}

  void Init(CkArrayID aid, CkGroupID gid, int grain, int blk, int s, bool sym, bool _useComlib,  ComlibInstanceHandle h, bool _dp, bool _conserveMemory, bool _lbpaircalc, ComlibInstanceHandle m , CkGroupID _mCastGrpId, bool _gspacesum ) {
      
    Aid = aid;
    Gid = gid;
    GrainSize = grain;
    BlkSize = blk;
    S = s;
    Symmetric = sym;
    useComlib = _useComlib;
    conserveMemory = _conserveMemory;
    cinst = h;
    minst = m;
    existsRproxy=false;
    existsLproxy=false;
    existsLNotFromproxy=false;
    isDoublePacked = _dp;
    lbpaircalc=_lbpaircalc;
    mCastGrpId=_mCastGrpId;
    gspacesum=_gspacesum;
  }
  void resetProxy()
    {
      if(useComlib && _PC_COMMLIB_MULTI_)
	{
	  if(existsLNotFromproxy)
	    ComlibResetSectionProxy(&proxyLNotFrom);
	  if(existsRproxy)
	    ComlibResetSectionProxy(&proxyRNotFrom);
	  if(existsLproxy)
	    ComlibResetSectionProxy(&proxyLFrom);
	}
      else
	{
	  CkMulticastMgr *mcastGrp = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();       
	  if(existsRproxy)
	    {
	      mcastGrp->resetSection(proxyRNotFrom);
	    }
	  if(existsLproxy)
	    {
	      mcastGrp->resetSection(proxyLFrom);
	    }
	  if(existsLNotFromproxy)
	    {
	      mcastGrp->resetSection(proxyLNotFrom);
	    }
	}

    }
  void pup(PUP::er &p) {
    p|Aid;
    p|Gid;
    p|GrainSize;
    p|BlkSize;
    p|S;
    p|Symmetric;
    p|cinst;
    p|minst;
    p|useComlib;
    p|isDoublePacked;
    p|conserveMemory;
    p|lbpaircalc;
    p|existsLproxy;
    p|existsLNotFromproxy;
    p|existsRproxy;
    p|gspacesum;
    p|proxyLFrom;
    p|proxyLNotFrom;
    p|proxyRNotFrom;
    p|mCastGrpId;
    p|sections;
  }

};

void createPairCalculator(bool sym, int w, int grainSize, int numZ, int* z, int op1, FuncType f1, int op2, FuncType f2, CkCallback cb, PairCalcID* aid, int ep, CkArrayID cbid, int flag, CkGroupID *gid, int flag_dp, bool conserveMemory, bool lbpaircalc, CkCallback lbcb, CkGroupID mCastGrpId, bool gspacesum);

void startPairCalcLeft(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRight(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);
void makeLeftTree(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void makeRightTree(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

extern "C" void finishPairCalc(PairCalcID* aid, int n, double *ptr);

extern "C" void finishPairCalc2(PairCalcID* pcid, int n, double *ptr1, double *ptr2);

void startPairCalcLeftAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRightAndFinish(PairCalcID* pcid, int n, complex* ptr, int myS, int myZ);

void isAtSyncPairCalc(PairCalcID* pcid);

/* These are the classic no multicast version for comparison and debugging */
void startPairCalcLeftSlow(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

void startPairCalcRightSlow(PairCalcID* aid, int n, complex* ptr, int myS, int myZ);

#endif
