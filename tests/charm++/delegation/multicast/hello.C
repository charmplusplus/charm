#include <stdio.h>
#include "charm++.h"
#include "ckmulticast.h"

#include "hello.decl.h"

CkChareID mid;
CProxy_Hello arr;
CkGroupID mCastGrpId;
CProxySection_Hello *mcast;

#define SECTIONSIZE  5
#define REDUCE_TIME  1000

void client(CkSectionCookie sid, void *param, int dataSize, void *data);

class HiMsg : public CkMcastBaseMsg, public CMessage_HiMsg
{
public:
  int *data;
//	HiMsg(int n) {data=n;}
};


class myReductionCounter {
public:
  int reductionNo;
  int reductionsRemaining;
public:
  myReductionCounter(): reductionNo(0), reductionsRemaining(0) {}
};
PUPbytes(myReductionCounter);

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


class Hello : public CBase_Hello
{
private:
  CkSectionCookie sid;
  int init;
  myReductionCounter cnt;
  CProxySection_Hello mcp;

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
    mcp.ckDelegate(mg);

    mg->setSection(mcp);

#if 0
    mg->setSection(sid, thisArrayID, al, SECTIONSIZE);
    sid.create(thisArrayID);
    for (int i=0; i<SECTIONSIZE; i++) 
      sid.addMember(CkArrayIndex1D(i));
    mg->setSection(sid);
#endif

    cnt.reductionsRemaining=REDUCE_TIME;
    cnt.reductionNo=0;
//    mg->setReductionClient(mcp, client, cnt);
    CkCallback *cb = new CkCallback(CkIndex_Hello::cb_client(NULL), CkArrayIndex1D(0), thisProxy);
    mg->setReductionClient(mcp, cb);

    HiMsg *hiMsg = new (1, 0) HiMsg;
    hiMsg->data[0] = 17;
    mcp.SayHi(hiMsg);
  }
  
//  void cb_client(CkSectionCookie sid, void *param, int dataSize, void *data)
  void cb_client(CkReductionMsg *msg)
  {
    int dataSize = msg->getSize();
    void *data = msg->getData();
    CmiPrintf("RESULT [%d]: %d\n", cnt.reductionNo, *(int *)data); 

    // check correctness
    int result;
    int redno = msg->getRedNo();
    if (redno%3 == 0) {
      result = 0;
      for (int i=0; i<SECTIONSIZE; i++) result+=i;
    }
    else if (redno%3 == 2) {
      result = 1;
      for (int i=1; i<SECTIONSIZE+1; i++) result*=i;
    }
    else {
      result = SECTIONSIZE+1;
    }
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
#if 0
      CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
      if (cnt->reductionNo % 32 == 0)
        mg->rebuild(mcp.ckGetSectionCookie());
#endif
  
      if (cnt.reductionNo%3 == 0) {
        HiMsg *hiMsg = new (1, 0) HiMsg;
        hiMsg->data[0] = 18+cnt.reductionNo;
        mcp.SayHi(hiMsg);
      }
      cnt.reductionNo++;
    }
    delete msg;
  }

  void SayHi(HiMsg *m)
  {
//    CkPrintf("[%d] Hi[%d] from element %d\n",CmiMyPe(), m->data[0],thisIndex);

    CkGetSectionCookie(sid, m);
//CmiPrintf("[%d] SayHi: sid on %d %p\n", CmiMyPe(), sid.pe, sid.val);

    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

    int data = thisIndex;
    CkCallback cb(CkIndex_Hello::cb_client(NULL), CkArrayIndex1D(0), thisProxy);
    mg->contribute(sizeof(int), &data,CkReduction::sum_int, sid, cb);
    data = thisIndex+2;
    mg->contribute(sizeof(int), &data,CkReduction::max_int, sid);
    data = thisIndex+1;
    mg->contribute(sizeof(int), &data,CkReduction::product_int, sid);
    delete m;
    if (1)
    ckMigrate((CkMyPe()+1)%CkNumPes());
  }

  void pup(PUP::er &p) {
    ArrayElement1D::pup(p);//Call superclass
    p|sid;
    p(init);
    p|cnt;
    p|mcp;
  }
};

#if 0
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
#endif

#include "hello.def.h"
