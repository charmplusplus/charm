
#include "check.decl.h"
#include "ckmulticast.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Check checkGroup;
/* readonly */ CkGroupID mCastGrpId;


#define BRANCHING_FACTOR 3


struct sectionBcastMsg : public CkMcastBaseMsg, public CMessage_sectionBcastMsg {

   int k;
   sectionBcastMsg(int _k) : k(_k) {} 
   void pup(PUP::er &p){
	  CMessage_sectionBcastMsg::pup(p);
	  p|k;
   }
};

class Main : public CBase_Main {
   int sum;
   public:
   Main(CkArgMsg* msg){
	  ckout<<"Numpes: "<<CkNumPes()<<endl;
	  checkGroup = CProxy_Check::ckNew();
	  mCastGrpId = CProxy_CkMulticastMgr::ckNew();
	  checkGroup.createSection();
	  sum = 0;
	  mainProxy = thisProxy;
   }
   Main(CkMigrateMessage* msg){}
   void done(int k){
	  ckout<<"Sum : "<<k<<endl;
	  CkExit();
   }
};


class Check : public CBase_Check {
   CProxySection_Check secProxy;
   CkSectionInfo cookie;
   public:
   Check() {}
   Check(CkMigrateMessage* msg) {}		
   void createSection(){
	  int numpes = CkNumPes(), step=1;
	  int me = CkMyPe();
	  if(CkMyPe() == 0){   //root
		 CkVec<int> elems; 
		 for(int i=0; i<numpes; i+=step){
			elems.push_back(i);
			ckout<<i<<" : "<<endl;
		 }
		 secProxy = CProxySection_Check(checkGroup.ckGetGroupID(), elems.getVec(), elems.size()); 
		 CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
		 secProxy.ckSectionDelegate(mCastGrp);
		 mCastGrp->setReductionClient(secProxy, new CkCallback(CkReductionTarget(Main,done), mainProxy));
		 sectionBcastMsg *msg = new sectionBcastMsg(1);
		 secProxy.recvMsg(msg);
	  }
   }

   void recvMsg(sectionBcastMsg *msg){
	  ckout<<"sectionBcastMsg received  - "<<CkMyPe()<<endl;
	  int me = msg->k;
	  CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
	  CkGetSectionInfo(cookie, msg);
	  mCastGrp->contribute(sizeof(int), &me, CkReduction::sum_int, cookie);
	  CkFreeMsg(msg);
   }
};


#include "check.def.h"
