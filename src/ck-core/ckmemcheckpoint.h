#ifndef _CKCHECKPT_
#define _CKCHECKPT_

#include "CkMemCheckpoint.decl.h"

extern CkGroupID ckCheckPTGroupID;

class CkArrayCheckPTReqMessage: public CMessage_CkArrayCheckPTReqMessage {
public: 
  CkArrayCheckPTReqMessage()  {}
};

class CkArrayCheckPTMessage: public CMessage_CkArrayCheckPTMessage {
public:
	CkArrayID  aid;
	CkGroupID  locMgr;
	CkArrayIndexMax index;
	double *packData;
	int bud1, bud2;
	int len;
	int cp_flag;          // 1: from checkpoint 0: from recover
};

class CkProcCheckPTMessage: public CMessage_CkProcCheckPTMessage {
public:
	int pe;
	int failedpe;
	int cur_restart_phase;
	int len;
	char *packData;
};

// table entry
class CkMemCheckPTInfo {
   friend class CkMemCheckPT;
private:
   CkArrayID aid;
   CkGroupID locMgr;
   CkArrayIndexMax index;
   int pNo;   //another buddy
   CkArrayCheckPTMessage* ckBuffer; 
public:
   CkMemCheckPTInfo(CkArrayID a, CkGroupID loc, CkArrayIndexMax idx, int no):
            aid(a), locMgr(loc), index(idx), pNo(no), ckBuffer(NULL)  {}
   ~CkMemCheckPTInfo() { if (ckBuffer) delete ckBuffer; }
   void updateBuffer(CkArrayCheckPTMessage *data) { 
	if (ckBuffer) delete ckBuffer;
	ckBuffer = data;
   }
};

class CkMemCheckPT: public CBase_CkMemCheckPT {
public:
  CkMemCheckPT();
  CkMemCheckPT(CkMigrateMessage *m):CBase_CkMemCheckPT(m) { }
  ~CkMemCheckPT();
  void pup(PUP::er& p);
  void doItNow(int sp, CkCallback &);
  void restart(int failedPe);
  void removeArrayElements();
  void createEntry(CkArrayID aid, CkGroupID loc, CkArrayIndexMax index, int buddy);
  void recvData(CkArrayCheckPTMessage *);
  void recvProcData(CkProcCheckPTMessage *);
  void cpFinish();
  void recoverBuddies();
  void recoverArrayElements();
  void quiescence(CkCallback &);
  void resetReductionMgr();
  void finishUp();
  void inmem_restore(CkArrayCheckPTMessage *m);
  void resetLB(int diepe);
public:
  static CkCallback  cpCallback;

  int inRestarting;
private:
  CkVec<CkMemCheckPTInfo *> ckTable;

  int recvCount, peCount;
  int cpStarter;
  CkVec<int> failedPes;
  int thisFailedPe;
private:
  inline int iFailed() { return isFailed(CkMyPe()); }
  int isFailed(int pe);
  void failed(int pe);

  void sendProcData();
};

// called by user applications
//void CkRegisterRestartCallback(CkCallback *cb);
void CkStartCheckPoint(CkCallback &cb);

int CkInRestart();

#endif
