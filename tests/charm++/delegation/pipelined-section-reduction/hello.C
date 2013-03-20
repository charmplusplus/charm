#include <stdio.h>
#include "charm++.h"
#include "ckmulticast.h"

#include "hello.decl.h"

CkChareID mid;
CProxy_Hello arr;
CkGroupID mCastGrpId;
CProxySection_Hello *mcast;

int NumElements;			// readonly
int CompleteMsgSize;
int NumFrags;
int NumReductions;
int SectionSize;

void client(CkSectionInfo sid, void *param, int dataSize, void *data);

class HiMsg : public CkMcastBaseMsg, public CMessage_HiMsg
{
public:
  int *data;
};


class myReductionCounter {
public:
  int reductionNo;
  int reductionsRemaining;
public:
  myReductionCounter(): reductionNo(0), reductionsRemaining(0) {}
};
PUPbytes(myReductionCounter);

class main : public CBase_main
{
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
    if(m->argc < 6) {
      CkAbort("Usage: hello <nElements> <completeMsgSize> <nFrags> <nReductioins> <sectionSize>\n");
    }
    NumElements = atoi(m->argv[1]);
    CompleteMsgSize = atoi(m->argv[2]);
    NumFrags = atoi(m->argv[3]);
    NumReductions = atoi(m->argv[4]);
    SectionSize = atoi(m->argv[5]);

    delete m;
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),NumElements);
    mid = thishandle;

    arr = CProxy_Hello::ckNew(NumElements);

    mCastGrpId = CProxy_CkMulticastMgr::ckNew();

    arr[0].start();

  };

  void maindone(void)
  {
    static int count = 0;
    count ++;
    CkPrintf("All done\n");
    CkExit();
  };
};


class Hello : public CBase_Hello
{
private:
  CkSectionInfo sid;
  int init;
  myReductionCounter cnt;
  CProxySection_Hello mcp;
  int sectionSize;
  double startTime;
  double endTime;
  int redNo;

public:
  Hello()
  {
    //CkPrintf("Hello %d created\n",thisIndex);
    init = 0;
    redNo = 0;
  }

  Hello(CkMigrateMessage *m) {}

  void start()
  {
    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

    sectionSize = SectionSize;
    if (sectionSize > NumElements) sectionSize = NumElements;
    CkArrayIndex *al = new CkArrayIndex[sectionSize];
    for (int i=0; i<sectionSize; i++) {
      al[i] = CkArrayIndex1D(i);
    }
    mcp = CProxySection_Hello::ckNew(thisArrayID, al, sectionSize);
    mcp.ckSectionDelegate(mg);
    delete [] al;
    cnt.reductionsRemaining=NumReductions;
    cnt.reductionNo=0;
    CkCallback *cb = new CkCallback(CkIndex_Hello::cb_client(NULL), CkArrayIndex1D(0), thisProxy);
    mg->setReductionClient(mcp, cb);

    HiMsg *hiMsg = new (2, 0) HiMsg;
    hiMsg->data[0] = 22;
    hiMsg->data[1] = 28;
    startTime = CkWallTimer ();
    mcp.SayHi(hiMsg);
  }
  
  void cb_client(CkReductionMsg *msg)
  {
    endTime = CkWallTimer ();
    int dataSize = msg->getSize();
    void *data = msg->getData();
    CmiPrintf("%e\n", endTime-startTime); 

    // check correctness
    int result;
    int redno = msg->getRedNo();
    result = 0;
    for (int i=0; i<sectionSize; i++) result+=i;

    if (*(int *)data != result) {
      CmiPrintf("Expected: %d acual:%d\n", result, *(int *)data);
      CmiAbort("reduction result is wrong!");
    }
  
    cnt.reductionsRemaining--;
    if (cnt.reductionsRemaining<=0) {
      CProxy_main mproxy(mid);
      mproxy.maindone();
      cnt.reductionNo++;
    }
    else {
      HiMsg *hiMsg = new (2, 0) HiMsg;
      hiMsg->data[0] = 22;
      hiMsg->data[1] = 28;
      startTime = CkWallTimer ();  
      mcp.SayHi(hiMsg);
      cnt.reductionNo++;
    }
    delete msg;
  }

  void SayHi(HiMsg *m)
  {
    redNo ++;
    CmiAssert(m->data[0] == 22 && m->data[1] == 28);

    CkGetSectionInfo(sid, m);

    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
    int dataSize = (int)(CompleteMsgSize);
    int* data = new int [dataSize];
    int fragSize = dataSize/NumFrags;
    CkAssert (0 != fragSize);
    for (int i=0; i<dataSize; i++) {
      data [i] = thisIndex;
    }
    CkCallback cb(CkIndex_Hello::cb_client(NULL), CkArrayIndex1D(0), thisProxy);
    mg->contribute(dataSize*sizeof(int), data,CkReduction::sum_int, sid, cb, fragSize*sizeof(int));
//    data[0] = thisIndex+2;
//    data[1] = thisIndex+2;
//    mg->contribute(2*sizeof(int), &data,CkReduction::max_int, sid, sizeof(int));
//    data[0] = thisIndex+1;
//    data[1] = thisIndex+1;
//    mg->contribute(2*sizeof(int), &data,CkReduction::product_int, sid, sizeof(int));
    delete m;
    if (1)
      ckMigrate((CkMyPe()+1)%CkNumPes());
  }

  void pup(PUP::er &p) {
    p|sid;
    p(init);
    p|cnt;
    p|mcp;
    p|sectionSize;
    p|startTime;
    p|endTime;
    p|redNo;
  }
};

#include "hello.def.h"
