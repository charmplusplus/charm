#ifndef _CK_MEM_CHECKPT_
#define _CK_MEM_CHECKPT_

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
	CkArrayIndex index;
	double *packData;
	int bud1, bud2;
	int len;
	int cp_flag;          // 1: from checkpoint 0: from recover
};


class CkProcCheckPTMessage: public CMessage_CkProcCheckPTMessage {
public:
	int pe;
	int reportPe;		// chkpt starter
	int failedpe;
	int cur_restart_phase;
	int len;
	int pointer;
	char *packData;
};

// table entry base class
class CkCheckPTInfo {
   friend class CkMemCheckPT;
protected:
   CkArrayID aid;
   CkGroupID locMgr;
   CkArrayIndex index;
   int pNo;   //another buddy
public:
   CkCheckPTInfo();
   CkCheckPTInfo(CkArrayID a, CkGroupID loc, CkArrayIndex idx, int pno):
                  aid(a), locMgr(loc), index(idx), pNo(pno)   {}
   virtual ~CkCheckPTInfo() {}
   virtual void updateBuffer(CkArrayCheckPTMessage *data) = 0;
   virtual CkArrayCheckPTMessage * getCopy() = 0;
   virtual void updateBuddy(int b1, int b2) = 0;
   virtual int getSize() = 0;
};

/// memory or disk checkpointing
#define CkCheckPoint_inMEM   1
#define CkCheckPoint_inDISK  2

class CkCheckPTEntry{
  CkArrayCheckPTMessage **data;
  char * fname;
public:
  int bud1, bud2;
  int where;
  void init(int _where, int idx)
  {
    data = new CkArrayCheckPTMessage*[2];
    data[0] = NULL;
    data[1] = NULL;
    where = _where;
    if(where == CkCheckPoint_inDISK)
    {
#if CMK_USE_MKSTEMP
      fname = new char[64];
#if CMK_CONVERSE_MPI
      sprintf(fname, "/tmp/ckpt%d-%d-%d-XXXXXX",CmiMyPartition(), CkMyPe(), idx);
#else
      sprintf(fname, "/tmp/ckpt%d-%d-XXXXXX", CkMyPe(), idx);
#endif
      if(mkstemp(fname)<0)
	{
	  CmiAbort("mkstemp fail in checkpoint");
	}
#else
      fname=tmpnam(NULL);
#endif
    }
  }

  void updateBuffer(int pointer, CkArrayCheckPTMessage * msg)
  {
    if(where == CkCheckPoint_inDISK)
    {
      envelope *env = UsrToEnv(msg);
      CkUnpackMessage(&env);
      data[pointer] = (CkArrayCheckPTMessage *)EnvToUsr(env);
      FILE *f = fopen(fname,"wb");
      PUP::toDisk p(f);
      CkPupMessage(p, (void **)&msg);
      // delay sync to the end because otherwise the messages are blocked
  //    fsync(fileno(f));
      fclose(f);
      bud1 = msg->bud1;
      bud2 = msg->bud2;
      delete msg;
    }else
    {
      CmiAssert(where == CkCheckPoint_inMEM);
      CmiAssert(msg!=NULL);
      delete data[pointer];
      data[pointer] = msg;
      bud1 = msg->bud1;
      bud2 = msg->bud2;
    }
  }
  
  CkArrayCheckPTMessage * getCopy(int pointer)
  {
    if(where == CkCheckPoint_inDISK)
    {
      CkArrayCheckPTMessage *msg;
      FILE *f = fopen(fname,"rb");
      PUP::fromDisk p(f);
      CkPupMessage(p, (void **)&msg);
      fclose(f);
      msg->bud1 = bud1;				// update the buddies
      msg->bud2 = bud2;
      return msg;
    }else
    {
      CmiAssert(where == CkCheckPoint_inMEM);
      if (data[pointer] == NULL) {
        CmiPrintf("[%d] recoverArrayElements: element does not have checkpoint data.", CkMyPe());
        CmiAbort("Abort!");
      }
      return (CkArrayCheckPTMessage *)CkCopyMsg((void **)&data[pointer]);
    }
  }
};


class CkMemCheckPT: public CBase_CkMemCheckPT {
public:
  CkMemCheckPT(int w);
  CkMemCheckPT(CkMigrateMessage *m):CBase_CkMemCheckPT(m) {};
  virtual ~CkMemCheckPT();
  void pup(PUP::er& p);
  inline int BuddyPE(int pe);
  void doItNow(int sp, CkCallback &);
  void restart(int diePe);
  void removeArrayElements();
  void createEntry(CkArrayID aid, CkGroupID loc, CkArrayIndex index, int buddy);
  void recvData(CkArrayCheckPTMessage *);
  void gotData();
  void recvProcData(CkProcCheckPTMessage *);
  void cpFinish();
  void syncFiles(void);
  void report();
  void recoverBuddies();
  void recoverEntry(CkArrayCheckPTMessage *msg);
  void recoverArrayElements();
  void quiescence(CkCallback &);
  void resetReductionMgr();
  void finishUp();
  void gotReply();
  void inmem_restore(CkArrayCheckPTMessage *m);
  void updateLocations(int n, CkGroupID *g, CkArrayIndex *idx, CmiUInt8 *id, int nowOnPe);
  void resetLB(int diepe);
  int  isFailed(int pe);
  void pupAllElements(PUP::er &p);
  void startArrayCheckpoint();
  void recvArrayCheckpoint(CkArrayCheckPTMessage *m);
  void recoverAll(CkArrayCheckPTMessage * msg, CkVec<CkGroupID> * gmap=NULL, CkVec<CkArrayIndex> * imap=NULL);
public:
  static CkCallback  cpCallback;

  static int inRestarting;
  static int inCheckpointing;
  static int inLoadbalancing;
  static double startTime;
  static char*  stage;

private:
  CkVec<CkCheckPTInfo *> ckTable;
  CkCheckPTEntry chkpTable[2];

  int recvCount, peCount;
  int expectCount, ackCount;
  int recvChkpCount;//expect to receive both the processor checkpoint and array checkpoint from buddy PE
  /// the processor who initiate the checkpointing
  int cpStarter;
  CkVec<int> failedPes;
  int thisFailedPe;

    /// to use memory or disk checkpointing
  int    where;
private:
  void initEntry();
  inline int isMaster(int pe);

  void failed(int pe);
  int  totalFailed();

  void sendProcData();
};

// called in initCharm
void CkMemRestart(const char *, CkArgMsg *);

// called by user applications
// to start a checkpointing
void CkStartMemCheckpoint(CkCallback &cb);

// true if inside a restarting phase
extern "C" int CkInRestarting(); 
extern "C" int CkInLdb(); 
extern "C" void CkSetInLdb(); 
extern "C" void CkResetInLdb();

extern "C" int CkHasCheckpoints();

extern "C" void CkDieNow();

#endif
