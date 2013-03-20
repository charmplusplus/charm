#include <stdio.h>
#include "charm++.h"
#include "ckmulticast.h"

#include "hello.decl.h"

CkChareID mid;
CProxy_Hello arr;
CkGroupID mCastGrpId;
CProxySection_Hello *mcast;
int nElements;			// readonly
int sectionSize;		// readonly

#define SECTIONSIZE  6
#define REDUCE_TIME  100

void client(CkSectionInfo sid, void *param, int dataSize, void *data);

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
PUPbytes(myReductionCounter)

class main : public CBase_main
{
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
    if(m->argc < 2) {
      CkAbort("Usage: hello <nElements>\n");
    }
    nElements = atoi(m->argv[1]);
    delete m;
    CkPrintf("Running Hello on %d processors for %d elements\n",
	     CkNumPes(),nElements);
    mid = thishandle;

    sectionSize = SECTIONSIZE;
    if (sectionSize > nElements) sectionSize = nElements;

    arr = CProxy_Hello::ckNew(nElements);

    mCastGrpId = CProxy_CkMulticastMgr::ckNew(3);   // new factor

    arr[0].start();

  };

  void maindone(void)
  {
    static int count = 0;
    count ++;
//    if (count < sectionSize*3) return;
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

public:
  Hello()
  {
    CkPrintf("Hello %d created\n",thisIndex);
    init = 0;
  }

  Hello(CkMigrateMessage *m) {}

    // only on index 0
  void start()
  {
    CkAssert(thisIndex == 0);
CmiPrintf("start %d elements\n", nElements);
    CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();

    // create section proxy
//    CkVec<CkArrayIndex1D> al;
//    for (int i=0; i<sectionSize; i++)
//      al.push_back(CkArrayIndex1D(i));
//    mcp = CProxySection_Hello::ckNew(thisArrayID, al.getVec(), al.size());
#if 0
    mcast = new CProxySection_Hello(thisArrayID, al, sectionSize, mCastGrpId);
#endif
    mcp = CProxySection_Hello::ckNew(thisArrayID, 0, sectionSize-1, 1);
    mcp.ckSectionDelegate(mg);

#if 0
    mcp.ckDelegate(mg);
    mg->setSection(mcp);
#endif

#if 0
    mg->setSection(sid, thisArrayID, al, sectionSize);
    sid.create(thisArrayID);
    for (int i=0; i<sectionSize; i++) 
      sid.addMember(CkArrayIndex1D(i));
    mg->setSection(sid);
#endif

    cnt.reductionsRemaining=REDUCE_TIME;
    cnt.reductionNo=0;
//    mg->setReductionClient(mcp, client, cnt);
    CkCallback *cb = new CkCallback(CkIndex_Hello::cb_client(NULL), CkArrayIndex1D(0), thisProxy);
    mg->setReductionClient(mcp, cb);

    HiMsg *hiMsg = new (2, 0) HiMsg;
    hiMsg->data[0] = 22;
    hiMsg->data[1] = 28;
    mcp.SayHi(hiMsg);
  }
  
//  void cb_client(CkSectionInfo sid, void *param, int dataSize, void *data)
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
      for (int i=0; i<sectionSize; i++) result+=i;
    }
    else if (redno%3 == 2) {
      result = 1;
      for (int i=1; i<sectionSize+1; i++) result*=i;
    }
    else {
      result = sectionSize+1;
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
        mg->rebuild(mcp.ckGetSectionInfo());
#endif
  
      if (cnt.reductionNo%3 == 0) {
        HiMsg *hiMsg = new (2, 0) HiMsg;
        //hiMsg->data[0] = 18+cnt.reductionNo;
        hiMsg->data[0] = 22;
        hiMsg->data[1] = 28;
        mcp.SayHi(hiMsg);
      }
      cnt.reductionNo++;
    }
    delete msg;
  }

  void SayHi(HiMsg *m)
  {
    // CkPrintf("[%d] Hi[%d] from element %d\n",CmiMyPe(), m->data[0],thisIndex);
    CmiAssert(m->data[0] == 22 && m->data[1] == 28);

    CkGetSectionInfo(sid, m);
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
    p|sid;
    p(init);
    p|cnt;
    p|mcp;
#if 1
    if (p.isUnpacking() && thisIndex == 0) {
      CkMulticastMgr *mg = CProxy_CkMulticastMgr(mCastGrpId).ckLocalBranch();
      mg->resetSection(mcp);
      CkCallback *cb = new CkCallback(CkIndex_Hello::cb_client(NULL), CkArrayIndex1D(0), thisProxy);
      mg->setReductionClient(mcp, cb);
    }
#endif
  }
};

#if 0
void client(CkSectionInfo sid, void *param, int dataSize, void *data)
{
  myReductionCounter *c=(myReductionCounter *)param;
  CmiPrintf("RESULT [%d]: %d\n", c->reductionNo, *(int *)data); 
  // check correctness
  int result;
  if (c->reductionNo%3 == 0) {
    result = 0;
    for (int i=0; i<sectionSize; i++) result+=i;
  }
  else if (c->reductionNo%3 == 2) {
    result = 1;
    for (int i=1; i<sectionSize+1; i++) result*=i;
  }
  else {
    result = sectionSize+1;
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
    mg->rebuild(mcp.ckGetSectionInfo());

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
