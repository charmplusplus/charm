#include "charm++.h"
#include "envelope.h"

#include "ckmulticast.h"

#define MAXMCASTCHILDREN  2

class IndexPos;

typedef CkQ<multicastGrpMsg *> multicastGrpMsgBuf;
typedef CkVec<CkArrayIndexMax>  arrayIndexList;
typedef CkVec<CkSectionCookie>  sectionIdList;
typedef CkVec<IndexPos>  arrayIndexPosList;
typedef CkVec<ReductionMsg *>  reductionMsgs;

class IndexPos {
public:
  CkArrayIndexMax idx;
  int  pe;
public:
  IndexPos() {}
  IndexPos(int i): idx(i), pe(i) {}
  IndexPos(CkArrayIndexMax i, int p): idx(i), pe(p) {};
};

class reductionInfo {
public:
  int lcounter;
  int ccounter;
  int gcounter;   // total elem collected
  redClientFn storedClient;
  void *storedClientParam;
  int redNo;
  reductionMsgs  msgs;
  reductionMsgs  futureMsgs;
public:
  reductionInfo(): lcounter(0), ccounter(0), gcounter(0), 
		   storedClientParam(NULL), redNo(0) {}
};

#define COOKIE_NOTREADY 0
#define COOKIE_READY    1
#define COOKIE_OLD      2

// BOC entry for one array section
class mCastEntry {
public:
  CkSectionCookie parentGrp;	// spanning tree parent
  sectionIdList children;
  arrayIndexList allElem;	// only useful on root
  arrayIndexList localElem;
  int pe;			// should always be mype
  CkSectionCookie rootSid;
  multicastGrpMsgBuf msgBuf;
  mCastEntry *oldc, *newc;
  // for reduction
  reductionInfo red;
  char needRebuild;

private:
  char flag;
public:
  mCastEntry(): flag(COOKIE_NOTREADY), oldc(NULL), newc(NULL), needRebuild(0){}
  mCastEntry(mCastEntry *);
  inline int hasParent() { return parentGrp.val?1:0; }
  inline int isObsolete() { return (flag == COOKIE_OLD); }
  inline void setObsolete() { flag=COOKIE_OLD; }
  inline int notReady() { return (flag == COOKIE_NOTREADY); }
  inline void setReady() { flag=COOKIE_READY; }
  inline void incReduceNo();
};

class cookieMsg: public CMessage_cookieMsg {
public:
  CkSectionCookie cookie;
public:
  cookieMsg() {};
  cookieMsg(CkSectionCookie m): cookie(m) {};
};


// setup message
class multicastSetupMsg: public CMessage_multicastSetupMsg {
public:
  int  nIdx;
  CkArrayIndexMax *arrIdx;
  int      *lastKnown;
  CkSectionCookie parent;
  CkSectionCookie rootSid;
  int redNo;
};

// message send in spanning tree
class multicastGrpMsg: public CkMcastBaseMsg, public CMessage_multicastGrpMsg {
};

class ReductionMsg: public CMessage_ReductionMsg {
public:
  int dataSize;
  char *data;
  CkReduction::reducerType reducer;
  CkSectionCookie sid;
  char flag;  // 1: come from array elem 2: come from BOC
  int redNo;
  int gcounter;
  char rebuilt;
public:
  static ReductionMsg* buildNew(int NdataSize,void *srcData,
		  CkReduction::reducerType reducer=CkReduction::invalid);
};


extern void CkPackMessage(envelope **pEnv);


mCastEntry::mCastEntry (mCastEntry *old): 
flag(COOKIE_NOTREADY), oldc(NULL), newc(NULL)
{
  parentGrp = old->parentGrp;
  for (int i=0; i<old->allElem.length(); i++)
    allElem.push_back(old->allElem[i]);
  pe = old->pe;
  red.storedClient = old->red.storedClient;
  red.storedClientParam = old->red.storedClientParam;
  red.redNo = old->red.redNo;
  needRebuild = 0;
}

inline void mCastEntry::incReduceNo()
{
  red.redNo ++;
  for (mCastEntry *next = newc; next; next=next->newc) next->red.redNo++;
}

// call setup to return a sectionid.
void CkMulticastMgr::setSection(CkSectionCookie &_id, CkArrayID aid, CkArrayIndexMax *al, int n)
{
  mCastEntry *entry = new mCastEntry;
  for (int i=0; i<n; i++)
    entry->allElem.push_back(al[i]);
//  entry->aid = aid;
  _id.aid = aid;
  _id.val = entry;		// allocate table for this section
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].init(_id);
}

void CkMulticastMgr::setSection(CkSectionCookie &id)
{
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].init(id);
}

void CkMulticastMgr::setSection(CProxySection_ArrayElement &proxy)
{
  CkSectionCookie &_id = proxy.ckGetSectionCookie();
  mCastEntry *entry = new mCastEntry;

  const CkArrayIndexMax *al = proxy.ckGetArrayElements();
  for (int i=0; i<proxy.ckGetNumElements(); i++) {
    entry->allElem.push_back(al[i]);
  }
  _id.aid = proxy.ckGetArrayID();
  _id.val = entry;		// allocate table for this section
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].init(_id);
}

void CkMulticastMgr::init(CkSectionCookie s)
{
  mCastEntry *entry = (mCastEntry *)s.val; 
  int n = entry->allElem.length();
//CmiPrintf("init: %d\n", n);
  multicastSetupMsg *msg = new (n, n, 0) multicastSetupMsg;
  msg->nIdx = n;
  msg->parent = CkSectionCookie();
  msg->rootSid = s;
  msg->redNo = entry->red.redNo;
  CkArray *array = CProxy_ArrayBase(s.aid).ckLocalBranch();
  for (int i=0; i<n; i++) {
    msg->arrIdx[i] = entry->allElem[i];
    int ape = array->lastKnown(entry->allElem[i]);
    msg->lastKnown[i] = ape;
  }
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  // sync call to seup
  cookieMsg *cookiemsg = mCastGrp[CmiMyPe()].setup(msg);
  delete cookiemsg;

  // clear msg buffer
  while (!entry->msgBuf.isEmpty()) {
     multicastGrpMsg *newmsg = entry->msgBuf.deq();
//CmiPrintf("[%d] release buffer %p %d\n", CmiMyPe(), newmsg, newmsg->ep);
//    newmsg->cookie.val = entry;
    mCastGrp[CmiMyPe()].recvMsg(newmsg);
  }
  // release reduction msgs
  releaseFutureReduceMsgs(entry);
}

void CkMulticastMgr::teardown(CkSectionCookie cookie)
{
  int i;
  mCastEntry *sect = (mCastEntry *)cookie.val;

  sect->setObsolete();

  releaseBufferedReduceMsgs(sect);

  CProxy_CkMulticastMgr mp(thisgroup);
  for (i=0; i<sect->children.length(); i++) {
    mp[sect->children[i].pe].teardown(sect->children[i]);
  }

}

void CkMulticastMgr::freeup(CkSectionCookie cookie)
{
  mCastEntry *sect = (mCastEntry *)cookie.val;

  while (sect) {
    CProxy_CkMulticastMgr mp(thisgroup);
    for (int i=0; i<sect->children.length(); i++) {
      CkSectionCookie &s = sect->children[i];
      mp[s.pe].freeup(s);
    }
    // free cookie
CmiPrintf("[%d] Free up on %p\n", CmiMyPe(), sect);
    mCastEntry *oldc= sect->oldc;
    delete sect;
    sect = oldc;
  }
}

cookieMsg * CkMulticastMgr::setup(multicastSetupMsg *msg)
{
  int i,j;
  mCastEntry *entry;
  if (msg->parent.pe == CmiMyPe()) entry = (mCastEntry *)msg->rootSid.val; //sid.val;
  else entry = new mCastEntry;
  entry->pe = CmiMyPe();
  entry->rootSid = msg->rootSid;
  entry->parentGrp = msg->parent;
  entry->red.redNo = msg->redNo;
//CmiPrintf("[%d] setup: redNo: %d\n", CmiMyPe(), entry->red.redNo);

  int numpes = CmiNumPes();
  arrayIndexPosList *lists = new arrayIndexPosList[numpes];
  for (i=0; i<msg->nIdx; i++) {
    // msg->arrIdx[i] is local ?
    int lastKnown = msg->lastKnown[i];
    if (lastKnown == CmiMyPe()) {
      entry->localElem.insertAtEnd(msg->arrIdx[i]);
    }
    else {
      lists[lastKnown].push_back(IndexPos(msg->arrIdx[i], lastKnown));
    }
  }
  // divide into MAXMCASTCHILDREN slots
  int numchild = 0;
  int num = 0;
  for (i=0; i<numpes; i++) {
    if (i==CmiMyPe()) continue;
    if (lists[i].length()) num++;
  }
  if (MAXMCASTCHILDREN <= 0) numchild = num;
  else numchild = num<MAXMCASTCHILDREN?num:MAXMCASTCHILDREN;

  if (numchild) {
    arrayIndexPosList *slots = new arrayIndexPosList[numchild];
    num = 0;
    for (i=0; i<numpes; i++) {
      if (i==CmiMyPe()) continue;
      if (lists[i].length() == 0) continue;
      for (j=0; j<lists[i].length(); j++)
	slots[num].push_back(lists[i][j]);
      num = (num+1) % numchild;
    }

    // send messages
    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    for (i=0; i<numchild; i++) {
      int n = slots[i].length();
      multicastSetupMsg *m = new (n, n, 0) multicastSetupMsg;
      m->parent = CkSectionCookie(entry);
      m->nIdx = slots[i].length();
      m->rootSid = msg->rootSid;
      m->redNo = msg->redNo;
      for (j=0; j<slots[i].length(); j++) {
        m->arrIdx[j] = slots[i][j].idx;
        m->lastKnown[j] = slots[i][j].pe;
      }
      int childroot = slots[i][0].pe;
//CmiPrintf("[%d]:call set up %d numelem:%d\n", CmiMyPe(), childroot, n);
      cookieMsg *retmsg = mCastGrp[childroot].setup(m);
      entry->children.push_back(retmsg->cookie);
      delete retmsg;
    }
    delete [] slots;
  }
  delete [] lists;

  entry->setReady();

  cookieMsg *newmsg = new cookieMsg;
  newmsg->cookie.val = entry;
  return newmsg;
}

// when rebuilding, all multicast msgs will be buffered.
void CkMulticastMgr::rebuild(CkSectionCookie &sectId)
{
  // tear down old tree
  mCastEntry *curCookie = (mCastEntry*)sectId.val;
  // make sure I am the newest one
  while (curCookie->newc) curCookie = curCookie->newc;
  if (curCookie->isObsolete()) return;

  mCastEntry *newCookie = new mCastEntry(curCookie);  // allocate table for this section

  // build a chain
  newCookie->oldc = curCookie;
  curCookie->newc = newCookie;

  sectId.val = newCookie;

//CmiPrintf("rebuild: redNo:%d oldc:%p newc;%p\n", newCookie->red.redNo, curCookie, newCookie);

  curCookie->setObsolete();
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].reset(sectId);
}

// mark old cookie spanning tree as old and 
// build a new one
void CkMulticastMgr::reset(CkSectionCookie s)
{
  mCastEntry *newCookie = (mCastEntry*)s.val;
  mCastEntry *oldCookie = newCookie->oldc;

  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  // get rid of old one
//CmiPrintf("reset: oldc: %p\n", oldCookie);
  int mype = CmiMyPe();
  mCastGrp[mype].teardown(CkSectionCookie(mype, oldCookie, 0));

  // build a new one
  mCastGrp[mype].init(s);
}

void CkMulticastMgr::ArraySectionSend(int ep,void *m, CkArrayID a, CkSectionCookie &s)
{
//CmiPrintf("ArraySectionSend\n");

  if (s.pe == CmiMyPe()) {
    mCastEntry *entry = (mCastEntry *)s.val;   
    if (entry == NULL) {
      CmiAbort("Unknown array section, Did you forget to register the array section to CkMulticastMgr using setSection()?");
    }

    // update entry pointer in case there is newer one.
    if (entry->newc) {
      do { entry=entry->newc; } while (entry->newc);
      s.val = entry;
    }
    if (entry->needRebuild) rebuild(s);
  }

//CmiPrintf("ArraySend send to myself: %d\n", msgSize);

  register envelope *env = UsrToEnv(m);
  CkPackMessage(&env);
  m = EnvToUsr(env);
  multicastGrpMsg *msg = (multicastGrpMsg *)m;
  msg->aid = a;
  msg->_cookie = s;
  msg->ep = ep;

  if (s.pe == CmiMyPe())
    recvMsg(msg);
  else {
    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    mCastGrp[s.pe].recvMsg(msg);
  }
}

void CkMulticastMgr::recvMsg(multicastGrpMsg *msg)
{
  int i;
  mCastEntry *entry = (mCastEntry *)msg->_cookie.val;

  if (entry->notReady()) {
//CmiPrintf("enq buffer %p\n", msg);
    entry->msgBuf.enq(msg);
    return;
  }

  // send to spanning tree children
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  for (i=0; i<entry->children.length(); i++) {
    multicastGrpMsg *newmsg = (multicastGrpMsg *)CkCopyMsg((void **)&msg);
    newmsg->_cookie = entry->children[i];
    mCastGrp[entry->children[i].pe].recvMsg(newmsg);
  }

  // send to local
  int nLocal = entry->localElem.length();
//CmiPrintf("send to local %d\n", nLocal);
  for (i=0; i<nLocal-1; i++) {
    CProxyElement_ArrayBase ap(msg->aid, entry->localElem[i]);
    multicastGrpMsg *newm = (multicastGrpMsg *)CkCopyMsg((void **)&msg);
    ap.ckSend((CkArrayMessage *)newm, msg->ep);
  }
  if (nLocal) {
    CProxyElement_ArrayBase ap(msg->aid, entry->localElem[nLocal-1]);
    ap.ckSend((CkArrayMessage *)msg, msg->ep);
  }
  else {
    CkAssert (entry->rootSid.pe == CmiMyPe());
    delete msg;
  }

}

void CkGetSectionCookie(CkSectionCookie &id, void *msg)
{
  CkMcastBaseMsg *m = (CkMcastBaseMsg *)msg;
  if (CkMcastBaseMsg::checkMagic(m) == 0) 
    CmiAbort("Did you remember inherit multicast message from CkMcastBaseMsg?");
  id.pe = m->gpe();
  id.val = m->cookie();
  // note: retain old redNo
}

// Reduction

ReductionMsg* ReductionMsg::buildNew(int NdataSize,void *srcData,
                  CkReduction::reducerType reducer)
{
  ReductionMsg *newmsg = new (NdataSize, 0) ReductionMsg;
  newmsg->dataSize = NdataSize;
  memcpy(newmsg->data, srcData, NdataSize);
  newmsg->flag = 0;
  newmsg->redNo = 0;
  newmsg->gcounter = 0;
  return newmsg;
}

void CkMulticastMgr::setReductionClient(CProxySection_ArrayElement &proxy, redClientFn fn,void *param)
{
  CkSectionCookie &id = proxy.ckGetSectionCookie();
  mCastEntry *entry = (mCastEntry *)id.val;
  entry->red.storedClient = fn;
  entry->red.storedClientParam = param;
}

void CkMulticastMgr::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &id)
{
  if (id.val == NULL || id.redNo == -1) 
    CmiAbort("contribute: SectionID is not initialized\n");

  ReductionMsg *msg = ReductionMsg::buildNew(dataSize, data);
  msg->reducer = type;
  msg->sid = id;
  msg->flag = 1;   // from array element
  msg->redNo = id.redNo;
  msg->gcounter = 1;
  msg->rebuilt = (id.pe == CmiMyPe())?0:1;

  id.redNo++;
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
//CmiPrintf("[%d] val: %d %p\n", CmiMyPe(), id.pe, id.val);
  mCastGrp[id.pe].recvRedMsg(msg);
}

void CkMulticastMgr::recvRedMsg(ReductionMsg *msg)
{
  int i;
  CkSectionCookie id = msg->sid;
  mCastEntry *entry = (mCastEntry *)id.val;

  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  int updateReduceNo = 0;

  if (entry->isObsolete()) {
      // send up to root
//CmiPrintf("[%d] send to root %d\n", CmiMyPe(), entry->rootSid.pe);
      if (entry->rootSid.pe == CmiMyPe()) {
	// I am root, set to the new cookie if there is
	mCastEntry *newentry = entry->newc;
	while (newentry && newentry->newc) newentry=newentry->newc;
	entry = newentry;
	if (!entry || entry->isObsolete()) CmiAbort("Crazy!");
	msg->flag = 0;	     // indicate it is not on old spanning tree
	updateReduceNo = 1;  // reduce from old tree, new entry need update.
      }
      else {
	msg->sid = entry->rootSid;
	msg->flag = 0;
        mCastGrp[entry->rootSid.pe].recvRedMsg(msg);
        return;
      }
  }

//CmiPrintf("[%d] msg red:%d, %p, entry:%p redno:%d\n", CmiMyPe(), msg->redNo, msg, entry, entry->red.redNo);
  // old message come, ignore
  if (msg->redNo < entry->red.redNo) {
//CmiPrintf("[%d] msg redNo:%d, %p, entry:%p redno:%d\n", CmiMyPe(), msg->redNo, msg, entry, entry->red.redNo);
    CmiAbort("Could never happen! \n");
  }
  if (entry->notReady() || msg->redNo > entry->red.redNo) {
//CmiPrintf("[%d] Future redmsgs, buffered! msg:%p entry:%p %d\n", CmiMyPe(), msg, entry, entry->flag);
    entry->red.futureMsgs.push_back(msg);
    return;
  }

//CmiPrintf("[%d] recvRedMsg %d ref:%d\n", CmiMyPe(), msg->flag, entry->red.redNo);
  if (msg->flag == 1) entry->red.lcounter ++;
  if (msg->flag == 2) entry->red.ccounter ++;
  entry->red.gcounter += msg->gcounter;

  // buffer this msg
  // check first
  if (entry->red.msgs.length() && msg->dataSize != entry->red.msgs[0]->dataSize)
    CmiAbort("Reduction data are not of same length!");
  entry->red.msgs.push_back(msg);

//if (CmiMyPe() == 0)
//CmiPrintf("[%d] lcounter:%d-%d, ccounter:%d-%d, gcounter:%d-%d\n", CmiMyPe(),entry->red.lcounter,entry->localElem.length(), entry->red.ccounter, entry->children.length(), entry->red.gcounter, entry->allElem.length());
  int currentTreeUp = 0;
  if (entry->red.lcounter == entry->localElem.length() && 
      entry->red.ccounter == entry->children.length())
      currentTreeUp = 1;
  int mixTreeUp = 0;
  if (!entry->hasParent() && entry->red.gcounter == entry->allElem.length())
      mixTreeUp = 1;
  if (currentTreeUp || mixTreeUp)
  {
    int dataSize = msg->dataSize;
    CkReduction::reducerType reducer = msg->reducer;

    // reduce msgs
    reducerFn f= reducerTable[reducer];
    ReductionMsg *newmsg = (*f)(entry->red.msgs.length(), entry->red.msgs.getVec());
    // check if migration and free messages
    int rebuilt = 0;
    for (i=0; i<entry->red.msgs.length(); i++) {
      if (entry->red.msgs[i]->rebuilt) rebuilt = 1;
      delete entry->red.msgs[i];
    }
    entry->red.msgs.length() = 0;

    if (entry->hasParent()) {
      // send up
      newmsg->reducer = reducer;
      newmsg->sid = entry->parentGrp;
      newmsg->flag = 2;
      newmsg->redNo = entry->red.redNo;
      newmsg->gcounter = entry->red.gcounter;
      newmsg->rebuilt = rebuilt;
//CmiPrintf("send to parent: %d\n", entry->parentGrp.pe);
      mCastGrp[entry->parentGrp.pe].recvRedMsg(newmsg);
    }
    else {   // root
      if (entry->red.storedClient == NULL) 
	CmiAbort("Did you forget to register a reduction client?");
      entry->red.storedClient(id, entry->red.storedClientParam, dataSize,
	   newmsg->data);
      delete newmsg;

//CmiPrintf("currentTreeUp: %d entry:%p oldc: %p\n", currentTreeUp, entry, entry->oldc);
      if (currentTreeUp && entry->oldc) {
	// free old tree;
	mCastGrp[CmiMyPe()].freeup(CkSectionCookie(id.pe, entry->oldc, 0));
	entry->oldc = NULL;
      }
      if (rebuilt) entry->needRebuild = 1;
    }
    entry->incReduceNo();
//CmiPrintf("advanced entry:%p redNo: %d\n", entry, entry->red.redNo);
    if (updateReduceNo) mCastGrp[CmiMyPe()].updateRedNo(entry,entry->red.redNo);

    // reset counters
    entry->red.lcounter = entry->red.ccounter = entry->red.gcounter = 0;

    // release future msgs
    releaseFutureReduceMsgs(entry);
  }
}

void CkMulticastMgr::releaseFutureReduceMsgs(mCastEntryPtr entry)
{
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  for (int i=0; i<entry->red.futureMsgs.length(); i++) {
//CmiPrintf("releaseFutureReduceMsgs: %p\n", entry->red.futureMsgs[i]);
    mCastGrp[CmiMyPe()].recvRedMsg(entry->red.futureMsgs[i]);
  }
  entry->red.futureMsgs.length() = 0;
}

// these messages have to be sent to root
void CkMulticastMgr::releaseBufferedReduceMsgs(mCastEntryPtr entry)
{
  int i;
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  for (i=0; i<entry->red.msgs.length(); i++) {
//CmiPrintf("releaseBufferedReduceMsgs: %p\n", entry->red.msgs[i]);
    entry->red.msgs[i]->sid = entry->rootSid;
    entry->red.msgs[i]->flag = 0;
    mCastGrp[entry->rootSid.pe].recvRedMsg(entry->red.msgs[i]);
  }
  entry->red.msgs.length() = 0;

  for (i=0; i<entry->red.futureMsgs.length(); i++) {
//CmiPrintf("releaseBufferedFutureReduceMsgs: %p\n", entry->red.futureMsgs[i]);
    entry->red.futureMsgs[i]->sid = entry->rootSid;
    entry->red.futureMsgs[i]->flag = 0;
    mCastGrp[entry->rootSid.pe].recvRedMsg(entry->red.futureMsgs[i]);
  }
  entry->red.futureMsgs.length() = 0;
}

void CkMulticastMgr::updateRedNo(mCastEntryPtr entry, int red)
{
//CmiPrintf("[%d] updateRedNo entry:%p to %d\n", CmiMyPe(), entry, red);
  entry->red.redNo = red;

  CProxy_CkMulticastMgr mp(thisgroup);
  for (int i=0; i<entry->children.length(); i++) {
    mp[entry->children[i].pe].updateRedNo((mCastEntry *)entry->children[i].val, red);
  }

  releaseFutureReduceMsgs(entry);
}

////////////////////////////////////////////////////////////////////////////////
/////
///////////////// Builtin Reducer Functions //////////////
static ReductionMsg *invalid_reducer(int nMsg,ReductionMsg **msg)
{CkAbort("ERROR! Called the invalid reducer!\n");return NULL;}

/* A simple reducer, like sum_int, looks like this:
static ReductionMsg *sum_int(int nMsg,ReductionMsg **msg)
{
  int i,ret=0;
  for (i=0;i<nMsg;i++)
    ret+=*(int *)(msg[i]->data);
  return ReductionMsg::buildNew(sizeof(int),(void *)&ret);
}
*/

#define SIMPLE_REDUCTION(name,dataType,typeStr,loop) \
static ReductionMsg *name(int nMsg, ReductionMsg **msg)\
{\
  int m,i;\
  int nElem=msg[0]->dataSize/sizeof(dataType);\
  dataType *ret=(dataType *)(msg[0]->data);\
  for (m=1;m<nMsg;m++)\
  {\
    dataType *value=(dataType *)(msg[m]->data);\
    for (i=0;i<nElem;i++)\
    {\
      loop\
    }\
  }\
  return ReductionMsg::buildNew(nElem*sizeof(dataType),(void *)ret);\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_REDUCTION(nameBase,loop) \
  SIMPLE_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_REDUCTION(nameBase##_double,double,"%f",loop)


//Compute the sum the numbers passed by each element.
SIMPLE_POLYMORPH_REDUCTION(sum,ret[i]+=value[i];)

SIMPLE_POLYMORPH_REDUCTION(product,ret[i]*=value[i];)

SIMPLE_POLYMORPH_REDUCTION(max,if (ret[i]<value[i]) ret[i]=value[i];)

SIMPLE_POLYMORPH_REDUCTION(min,if (ret[i]>value[i]) ret[i]=value[i];)
CkMulticastMgr::reducerFn CkMulticastMgr::reducerTable[CkMulticastMgr::MAXREDUCERS]={
    ::invalid_reducer,
  //Compute the sum the numbers passed by each element.
    ::sum_int,::sum_float,::sum_double,
    ::product_int,::product_float,::product_double,
    ::max_int,::max_float,::max_double,
    ::min_int,::min_float,::min_double
};

#include "CkMulticast.def.h"
