#include <stdio.h>
#include "charm++.h"
#include "ckmulticast.h"

#include "hello.decl.h"

CkChareID mid;
CProxy_Hello arr;
CkGroupID mCastGrpId;
CProxySection_Hello *mcast;
CProxySection_Hello mcp;

#define SECTIONSIZE  5
#define REDUCE_TIME  100

class HiMsg : public CkMcastBaseMsg, public CMessage_HiMsg
{
public:
  int *data;
//	HiMsg(int n) {data=n;}
};


typedef struct {
  int reductionNo;
  int reductionsRemaining;
} myReductionCounter;

class main : public Chare
{
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
    if(m->argc < 2) {
      CkAbort("Usage: hello <nElements>\n");
    }
    int nElements = atoi(m->argv[1]);
    delete m;
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mid = thishandle;

    arr = CProxy_Hello::ckNew(nElements);

    mCastGrpId = CProxy_CkMulticastMgr::ckNew();

    arr[0].start();

  };

  void maindone(void)
  {
    static int count = 0;
    count ++;
//    if (count < SECTIONSIZE*3) return;
    CkPrintf("All done\n");
    CkExit();
  };
};

void client(CkSectionCookie sid, void *param, int dataSize, void *data)
{
  myReductionCounter *c=(myReductionCounter *)param;
  CmiPrintf("RESULT [%d]: %d\n", c->reductionNo, *(int *)data); 
  // check correctness
  int result;
  if (c->reductionNo%3 == 0) {
    result = 0;
    for (int i=0; i<SECTIONSIZE; i++) result+=i;
  }
  else if (c->reductionNo%3 == 2) {
    result = 1;
    for (int i=1; i<SECTIONSIZE+1; i++) result*=i;
  }
  else {
    result = SECTIONSIZE+1;
  }
  if (*(int *)data != result) {
    CmiAbort("wrong!");
  }

  c->reductionsRemaining--;
  if (c->reductionsRemaining<=0) {
    CProxy_main mproxy(mid);
    mproxy.maindone();
    c->reductionNo++;
  }
  else {
    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
    if (c->reductionNo % 32 == 0)
    mg->rebuild(mcp.ckGetSectionCookie());

    if (c->reductionNo%3 == 0) {
    HiMsg *hiMsg = new (1, 0) HiMsg;
    hiMsg->data[0] = 18+c->reductionNo;
    mcp.SayHi(hiMsg);
    }
    c->reductionNo++;
  }
}

class Hello : public ArrayElement1D
{
private:
  CkSectionCookie sid;
  int init;

public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
    init = 0;
  }

  Hello(CkMigrateMessage *m) {}

  void start()
  {
CmiPrintf("start\n");
    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

    CkArrayIndexMax al[SECTIONSIZE];
    for (int i=0; i<SECTIONSIZE; i++) {
      al[i] = CkArrayIndex1D(i);
    }
#if 0
    mcast = new CProxySection_Hello(thisArrayID, al, SECTIONSIZE, mCastGrpId);
#endif
    mcp = CProxySection_Hello::ckNew(thisArrayID, al, SECTIONSIZE);
    mcp.ckDelegate(mCastGrpId);

    mg->setSection(mcp);

#if 0
    mg->setSection(sid, thisArrayID, al, SECTIONSIZE);
    sid.create(thisArrayID);
    for (int i=0; i<SECTIONSIZE; i++) 
      sid.addMember(CkArrayIndex1D(i));
    mg->setSection(sid);
#endif

    myReductionCounter *c=new myReductionCounter;
    c->reductionsRemaining=REDUCE_TIME;
    c->reductionNo=0;
    mg->setReductionClient(mcp, client, c);

    HiMsg *hiMsg = new (1, 0) HiMsg;
    hiMsg->data[0] = 17;
    mcp.SayHi(hiMsg);
  }
  
  void SayHi(HiMsg *m)
  {
//    CkPrintf("[%d] Hi[%d] from element %d\n",CmiMyPe(), m->data[0],thisIndex);

    CkGetSectionCookie(sid, m);
//CmiPrintf("[%d] SayHi: sid on %d %p\n", CmiMyPe(), sid.pe, sid.val);

    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

    int data = thisIndex;
    mg->contribute(sizeof(int), &data,CkReduction::sum_int, sid);
    data = thisIndex+2;
    mg->contribute(sizeof(int), &data,CkReduction::max_int, sid);
    data = thisIndex+1;
    mg->contribute(sizeof(int), &data,CkReduction::product_int, sid);
    delete m;
  }
};

#include "hello.def.h"
