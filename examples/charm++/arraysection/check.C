
#include "check.decl.h"
#include "ckmulticast.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Check checkArray;
/* readonly */ int numchares;

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
		numchares = atoi(msg->argv[1]);
		ckout<<" numchares: "<<numchares<<endl;
		checkArray = CProxy_Check::ckNew(numchares);
		checkArray.createSection();
		sum = 0;
		mainProxy = thisProxy;
	}
	//Main(CkMigrateMessage* msg){}
	//void pup(PUP::er &p){}
	void done(int k){
		ckout<<"sum: "<<k<<endl;
		CkExit();
	}
};


class Check : public CBase_Check {
	CProxySection_Check secProxy;
	public:
	Check() {}
	Check(CkMigrateMessage* msg) {}		
	void createSection(){
		if(thisIndex == 0){
			CkVec<CkArrayIndex1D> elems;    // add array indices
			for (int i=0; i<numchares; i+=2)
				elems.push_back(CkArrayIndex1D(i));
			secProxy = CProxySection_Check(checkArray.ckGetArrayID(), elems.getVec(), elems.size(), 4);
			//Use setReductionClient or alternatively use callback
			//secProxy.setReductionClient(new CkCallback(CkReductionTarget(Main,done), mainProxy));
 			sectionBcastMsg *msg = new sectionBcastMsg(1);
			secProxy.recvMsg(msg);
		}
	}

	void recvMsg(sectionBcastMsg *msg){
		ckout<<"ArrayIndex: "<<thisIndex<<" - "<<CkMyPe()<<endl;
		int k = msg->k;
		CkSectionInfo cookie;
		CkGetSectionInfo(cookie, msg);
		CkCallback cb(CkReductionTarget(Main,done), mainProxy);
		CProxySection_Check::contribute(sizeof(int), &k, CkReduction::sum_int, cookie, cb);
		CkFreeMsg(msg);
	}
};


#include "check.def.h"
