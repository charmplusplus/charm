#include "charm++.h"
#include "envelope.h"

#include "ckmulticast.h"

#define COOKIE_NOTREADY 0
#define COOKIE_READY    1
#define COOKIE_OLD     2

class IndexPos;

typedef CkQ<multicastGrpMsg *> multicastGrpMsgBuf;
typedef CkVec<CkArrayIndexMax>  arrayIndexList;
typedef CkVec<CkSectionID>  sectionIdList;
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
  reductionInfo(): lcounter(0), ccounter(0), gcounter(0), storedClientParam(NULL), redNo(0) {}
};

// BOC entry for one array section
class mCastCookie {
public:
  CkSectionID parentGrp;
  sectionIdList children;
  arrayIndexList allElem;	// only on root
  arrayIndexList localElem;
  int pe;
  CkSectionID rootSid;
  multicastGrpMsgBuf msgBuf;
  int flag;
  mCastCookie *oldc, *newc;
  // for reduction
  reductionInfo red;
  int needRebuild;
public:
  // mCastCookie() { children = new CkSectionID[MAXMCASTCHILDREN];}
  mCastCookie(): flag(COOKIE_NOTREADY), oldc(NULL), newc(NULL), needRebuild(0){}
  mCastCookie(mCastCookie *);
  inline int hasParent() { return parentGrp.val?1:0; }
  inline int isObsolete() { return (flag == COOKIE_OLD); }
  inline int notReady() { return (flag == COOKIE_NOTREADY); }
  void incReduceNo();
};

class cookieMsg: public CMessage_cookieMsg {
public:
  CkSectionID cookie;
public:
  cookieMsg() {};
  cookieMsg(CkSectionID m): cookie(m) {};
};


// setup message
class multicastSetupMsg: public CMessage_multicastSetupMsg {
public:
  int  nIdx;
  CkArrayIndexMax *arrIdx;
  int      *lastKnown;
  CkSectionID parent;
  CkSectionID rootSid;
  int redNo;
};

class multicastGrpMsg: public CMessage_multicastGrpMsg {
public:
  CkSectionID cookie;
  CkArrayID aId;
  int ep;
  int msgsize;
  char *msg;
};

class ReductionMsg: public CMessage_ReductionMsg {
public:
  int dataSize;
  char *data;
  CkReduction::reducerType reducer;
  CkSectionID sid;
  int flag;  // 1: come from array elem 2: come from BOC
  int redNo;
  int gcounter;
  int rebuilt;
public:
  static ReductionMsg* buildNew(int NdataSize,void *srcData,
		  CkReduction::reducerType reducer=CkReduction::invalid);
};

#define HACK 0

extern void CkPackMessage(envelope **pEnv);


mCastCookie::mCastCookie (mCastCookie *old): flag(COOKIE_NOTREADY), oldc(NULL), newc(NULL)
{
//  aid = old->aid;
  parentGrp = old->parentGrp;
  for (int i=0; i<old->allElem.length(); i++)
    allElem.push_back(old->allElem[i]);
  pe = old->pe;
  red.storedClient = old->red.storedClient;
  red.storedClientParam = old->red.storedClientParam;
  red.redNo = old->red.redNo;
  needRebuild = 0;
}

void mCastCookie::incReduceNo()
{
  red.redNo ++;
  mCastCookie *next = newc;
  for (; next; next=next->newc) next->red.redNo++;
}

// call setup to return a sectionid.
void CkMulticastMgr::setSection(CkSectionID &_id, CkArrayID aid, CkArrayIndexMax *al, int n)
{
  mCastCookie *entry = new mCastCookie;
  for (int i=0; i<n; i++)
    entry->allElem.push_back(al[i]);
//  entry->aid = aid;
  _id.aid = aid;
  _id.val = entry;		// allocate table for this section
  // hack
#if HACK
  sid = _id;
#endif
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].init(_id);
}

void CkMulticastMgr::setSection(CkSectionID &id)
{
  // hack
#if HACK
  sid = id;
#endif
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].init(id);
}

void CkMulticastMgr::setSection(CProxySection_ArrayElement *proxy)
{
  CkSectionID &_id = proxy->ckGetSectionID();
  mCastCookie *entry = new mCastCookie;
  CkArrayIndexMax *al = proxy->ckGetArrayElements();
  for (int i=0; i<proxy->ckGetNumElements(); i++) {
    entry->allElem.push_back(al[i]);
  }
//  entry->aid = aid;
  _id.aid = proxy->ckGetArrayID();
  _id.val = entry;		// allocate table for this section
  // hack
#if HACK
  sid = _id;
#endif
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].init(_id);
}

void CkMulticastMgr::init(CkSectionID s)
{
  mCastCookie *entry = (mCastCookie *)s.val; 
  int n = entry->allElem.length();
//CmiPrintf("init: %d\n", n);
  multicastSetupMsg *msg = new (n, n, 0) multicastSetupMsg;
  msg->nIdx = n;
  msg->parent = CkSectionID();
  msg->rootSid = s;
  msg->redNo = entry->red.redNo;
  CkArray *array = CProxy_ArrayBase(s.aid).ckLocalBranch();
  for (int i=0; i<n; i++) {
    msg->arrIdx[i] = entry->allElem[i];
    int ape = array->lastKnown(entry->allElem[i]);
    msg->lastKnown[i] = ape;
  }
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  cookieMsg *cookiemsg = mCastGrp[CmiMyPe()].setup(msg);
  delete cookiemsg;

  // clear buffer
  while (!entry->msgBuf.isEmpty()) {
    multicastGrpMsg *newmsg = entry->msgBuf.deq();
//CmiPrintf("[%d] release buffer %p %d\n", CmiMyPe(), newmsg, newmsg->ep);
    newmsg->cookie.val = entry;
    mCastGrp[CmiMyPe()].recvMsg(newmsg);
  }
  // release reduction msgs
  releaseFutureReduceMsgs(entry);
}

void CkMulticastMgr::teardown(CkSectionID cookie)
{
  int i;
  mCastCookie *sect = (mCastCookie *)cookie.val;

  sect->flag = COOKIE_OLD;
  releaseBufferedReduceMsgs(sect);

  CProxy_CkMulticastMgr mp(thisgroup);
  for (i=0; i<sect->children.length(); i++) {
    mp[sect->children[i].pe].teardown(sect->children[i]);
  }
}

void CkMulticastMgr::freeup(CkSectionID cookie)
{
  mCastCookie *sect = (mCastCookie *)cookie.val;

  CProxy_CkMulticastMgr mp(thisgroup);
  for (int i=0; i<sect->children.length(); i++) {
    CkSectionID &s = sect->children[i];
    mp[s.pe].freeup(s);
  }
//CmiPrintf("[%d] Free up on %p\n", CmiMyPe(), sect);
  // free cookie
  delete sect;
}

cookieMsg * CkMulticastMgr::setup(multicastSetupMsg *msg)
{
  int i,j;
  mCastCookie *entry;
  if (msg->parent.pe == CmiMyPe()) entry = (mCastCookie *)msg->rootSid.val; //sid.val;
  else entry = new mCastCookie;
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
      lists[lastKnown].insertAtEnd(IndexPos(msg->arrIdx[i], lastKnown));
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

    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    for (i=0; i<numchild; i++) {
      int n = slots[i].length();
      multicastSetupMsg *m = new (n, n, 0) multicastSetupMsg;
      m->parent = CkSectionID(entry);
      m->nIdx = slots[i].length();
      m->rootSid = msg->rootSid;
      m->redNo = msg->redNo;
      for (int j=0; j<slots[i].length(); j++) {
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

  entry->flag = COOKIE_READY;

  cookieMsg *newmsg = new cookieMsg;
  newmsg->cookie.val = entry;
  return newmsg;
}

// when rebuilding, all multicast msgs will be buffered.
void CkMulticastMgr::rebuild(CkSectionID &sectId)
{
  // tear down old tree
  mCastCookie *curCookie = (mCastCookie*)sectId.val;
  if (curCookie->flag == COOKIE_OLD) return;

  mCastCookie *newCookie = new mCastCookie(curCookie);  // allocate table for this section

  // build a chain
  newCookie->oldc = curCookie;
  curCookie->newc = newCookie;

  sectId.val = newCookie;

  // hack
#if HACK
  sid.val = sectId.val;
#endif
//CmiPrintf("rebuild: redNo:%d oldc:%p newc;%p\n", newCookie->red.redNo, oldCookie, newCookie);

  curCookie->flag = COOKIE_OLD;
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CmiMyPe()].reset(sectId);
}

// mark old cookie spanning tree as old and 
// build a new one
void CkMulticastMgr::reset(CkSectionID s)
{
  mCastCookie *newCookie = (mCastCookie*)s.val;
  mCastCookie *oldCookie = newCookie->oldc;

  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  // get rid of old one
//CmiPrintf("reset: oldc: %p\n", oldCookie);
  int mype = CmiMyPe();
  mCastGrp[mype].teardown(CkSectionID(mype, oldCookie, 0));

  // build a new one
  mCastGrp[mype].init(s);
}

void CkMulticastMgr::ArraySend(int ep,void *m, const CkArrayIndexMax &idx, CkArrayID a)
{
  ArrayBroadcast(ep, m, a);
}

void CkMulticastMgr::ArrayBroadcast(int ep,void *m, CkArrayID a)
{
  ArraySectionSend(ep, m, a, sid);
}

void CkMulticastMgr::ArraySectionSend(int ep,void *m, CkArrayID a, CkSectionID &s)
{
  // hack
//  CkSectionID &thisSectId = sid;
//  mCastCookie *entry = (mCastCookie *)thisSectId.val;   
//CmiPrintf("ArraySectionSend: %p\n", s);
  CkSectionID *thisSectId = &s;
  mCastCookie *entry = (mCastCookie *)thisSectId->val;   
  //while (entry->newc) entry=entry->newc;

  if (entry->needRebuild) rebuild(s);

  register envelope *env = UsrToEnv(m);
  int msgSize = env->getTotalsize();

  // send to spanning tree children
//CmiPrintf("ArraySend send to myself: %d\n", msgSize);
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  multicastGrpMsg *newmsg = new (msgSize, 0) multicastGrpMsg;
  newmsg->cookie = *thisSectId;
  newmsg->aId = a;
  newmsg->ep = ep;
  newmsg->msgsize = msgSize;
  CkPackMessage(&env);
  memcpy(newmsg->msg, env, msgSize);
  CkFreeMsg(m);

  if (entry->flag == COOKIE_NOTREADY) {
//CmiPrintf("enq buffer %p\n", newmsg);
    entry->msgBuf.enq(newmsg);
  }
  else {
    mCastGrp[CmiMyPe()].recvMsg(newmsg);
  }

}


void CkMulticastMgr::recvMsg(multicastGrpMsg *msg)
{
  int i;
  envelope *env = (envelope *)msg->msg;
  void *m = EnvToUsr(env);

  mCastCookie *entry = (mCastCookie *)msg->cookie.val;

  // send to spanning tree children
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  for (i=0; i<entry->children.length(); i++) {
    multicastGrpMsg *newmsg = (multicastGrpMsg *)CkCopyMsg((void **)&msg);
    newmsg->cookie = entry->children[i];
    mCastGrp[entry->children[i].pe].recvMsg(newmsg);
  }

  // send to local
  int msgSize = env->getTotalsize();
//CmiPrintf("send to local %d\n", msgSize);
  for (i=0; i<entry->localElem.length(); i++) {
//CmiPrintf("local: %d %d\n", i, msg->ep);
//entry->localElem[i].print();
    CProxyElement_ArrayBase ap(msg->aId, entry->localElem[i]);
    envelope *newm = (envelope *)CmiAlloc(msgSize);
    memcpy(newm, env, msgSize);
    setSectionID(EnvToUsr(newm), msg->cookie);
    ap.ckSend((CkArrayMessage *)EnvToUsr(newm), msg->ep);
  }

  delete msg;
}

void setSectionID(void *msg, CkSectionID sid)
{
  envelope *env = UsrToEnv(msg);
  CkMcastBaseMsg *m = (CkMcastBaseMsg *)env;
  m->gpe() = sid.pe;
  m->cookie() = sid.val;
  m->redno() = sid.redNo;
}

void CkGetSectionID(CkSectionID &id, void *msg)
{
  envelope *env = UsrToEnv(msg);
  CkMcastBaseMsg *m = (CkMcastBaseMsg *)env;
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

void CkMulticastMgr::setReductionClient(CProxySection_ArrayElement *proxy, redClientFn fn,void *param)
{
  CkSectionID &id = proxy->ckGetSectionID();
  mCastCookie *entry = (mCastCookie *)id.val;
  entry->red.storedClient = fn;
  entry->red.storedClientParam = param;
}

void CkMulticastMgr::setReductionClient(CkSectionID id, redClientFn fn,void *param)
{
  mCastCookie *entry = (mCastCookie *)id.val;
  entry->red.storedClient = fn;
  entry->red.storedClientParam = param;
}

void CkMulticastMgr::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionID &id)
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
  mCastGrp[id.pe].recvRedMsg(msg);
}

void CkMulticastMgr::recvRedMsg(ReductionMsg *msg)
{
  int i;
  CkSectionID id = msg->sid;
  mCastCookie *entry = (mCastCookie *)id.val;

  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  if (entry->isObsolete()) {
      // send up to root
//CmiPrintf("[%d] send to root %d\n", CmiMyPe(), entry->rootSid.pe);
      if (entry->rootSid.pe == CmiMyPe()) {
	// I am root, set to the new cookie if there is
	mCastCookie *newentry = entry->newc;
	if (newentry) {
//CmiPrintf("send to new entry!\n");
	  msg->sid = CkSectionID(CmiMyPe(), newentry, id.redNo);
	}
	mCastGrp[CmiMyPe()].recvRedMsg(msg);
      }
      else {
	msg->sid = entry->rootSid;
        mCastGrp[entry->rootSid.pe].recvRedMsg(msg);
      }
      return;
  }

//CmiPrintf("[%d] msg %d, %p, entry:%p redno:%d\n", CmiMyPe(), msg->redNo, msg, entry, entry->red.redNo);
  if (msg->redNo < entry->red.redNo) CmiAbort("Could never happen! \n");
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
  entry->red.msgs.push_back(msg);

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
      mCastGrp[entry->parentGrp.pe].recvRedMsg(newmsg);
    }
    else {   // root
      entry->red.storedClient(id, entry->red.storedClientParam, dataSize,
	   newmsg->data);
      delete newmsg;

      if (currentTreeUp && entry->oldc) {
	// free old tree;
//CmiPrintf("free up : %p\n", entry->oldc);
	mCastGrp[CmiMyPe()].freeup(CkSectionID(id.pe, entry->oldc, 0));
	entry->oldc = NULL;
      }
      if (rebuilt) entry->needRebuild = 1;
    }
    entry->incReduceNo();
//CmiPrintf("advanced entry:%p redNo: %d\n", entry, entry->red.redNo);

    // reset counters
    entry->red.lcounter = entry->red.ccounter = entry->red.gcounter = 0;

    // release future msgs
    releaseFutureReduceMsgs(entry);
  }
}

void CkMulticastMgr::releaseFutureReduceMsgs(mCastCookie *entry)
{
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  for (int i=0; i<entry->red.futureMsgs.length(); i++) {
//CmiPrintf("releaseFutureReduceMsgs: %p\n", entry->red.futureMsgs[i]);
    mCastGrp[CmiMyPe()].recvRedMsg(entry->red.futureMsgs[i]);
  }
  entry->red.futureMsgs.length() = 0;
}

// these messages have to be sent to root
void CkMulticastMgr::releaseBufferedReduceMsgs(mCastCookie *entry)
{
  int i;
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  for (i=0; i<entry->red.msgs.length(); i++) {
//CmiPrintf("releaseBufferedReduceMsgs: %p\n", entry->red.msgs[i]);
    entry->red.msgs[i]->sid = entry->rootSid;
    mCastGrp[entry->rootSid.pe].recvRedMsg(entry->red.msgs[i]);
  }
  entry->red.msgs.length() = 0;

  for (i=0; i<entry->red.futureMsgs.length(); i++) {
//CmiPrintf("releaseBufferedFutureReduceMsgs: %p\n", entry->red.futureMsgs[i]);
    entry->red.futureMsgs[i]->sid = entry->rootSid;
    mCastGrp[entry->rootSid.pe].recvRedMsg(entry->red.futureMsgs[i]);
  }
  entry->red.futureMsgs.length() = 0;
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
