#include "charm++.h"
#include "envelope.h"

#include "ckmulticast.h"

#define DEBUGF(x)    // CkPrintf x;

#define MAXMCASTCHILDREN  2

typedef CkQ<multicastGrpMsg *> multicastGrpMsgBuf;
typedef CkVec<CkArrayIndexMax>  arrayIndexList;
typedef CkVec<CkSectionCookie>  sectionIdList;
typedef CkVec<CkReductionMsg *>  reductionMsgs;


class reductionInfo {
public:
  int lcount;   /**< local elem collected */
  int ccount;   /**< children node collected */
  int gcount;   /**< total elem collected */
  CkCallback *storedCallback;   /**< user callback */
  redClientFn storedClient;     /**< reduction client function */
  void *storedClientParam;      /**< user provided data */
  int redNo;                    /**< reduction sequence number */
  reductionMsgs  msgs;          /**< messages for this reduction */
  reductionMsgs  futureMsgs;    /**< messages of future reductions */
public:
  reductionInfo(): lcount(0), ccount(0), gcount(0), 
		   storedCallback(NULL), storedClientParam(NULL), redNo(0) {}
};

/// cookie status
#define COOKIE_NOTREADY 0
#define COOKIE_READY    1
#define COOKIE_OLD      2

/// cookie for an array section 
class mCastEntry {
public:
  CkSectionCookie parentGrp;	/**< spanning tree parent */
  sectionIdList children;       /**< children section list */
  int numChild;
  arrayIndexList allElem;	// only useful on root
  arrayIndexList localElem;
  int pe;			/**< should always be mype */
  CkSectionCookie rootSid;      /**< section ID of the root */
  multicastGrpMsgBuf msgBuf;
  mCastEntry *oldc, *newc;
  // for reduction
  reductionInfo red;
  char needRebuild;

private:
  char flag;
public:
  mCastEntry(): numChild(0), 
                oldc(NULL), newc(NULL), needRebuild(0),
		flag(COOKIE_NOTREADY) {}
  mCastEntry(mCastEntry *);
  inline int hasParent() { return parentGrp.val?1:0; }
  inline int isObsolete() { return (flag == COOKIE_OLD); }
  inline void setObsolete() { flag=COOKIE_OLD; }
  inline int notReady() { return (flag == COOKIE_NOTREADY); }
  inline void setReady() { flag=COOKIE_READY; }
  inline void incReduceNo() {
                red.redNo ++;
                for (mCastEntry *next = newc; next; next=next->newc) 
                   next->red.redNo++;
              }
};

class cookieMsg: public CMessage_cookieMsg {
public:
  CkSectionCookie cookie;
public:
  cookieMsg() {};
  cookieMsg(CkSectionCookie m): cookie(m) {};
};


/// multicast tree setup message
class multicastSetupMsg: public CMessage_multicastSetupMsg {
public:
  int  nIdx;
  CkArrayIndexMax *arrIdx;
  int      *lastKnown;
  CkSectionCookie parent;
  CkSectionCookie rootSid;
  int redNo;
};

/// message send in spanning tree
class multicastGrpMsg: public CkMcastBaseMsg, public CMessage_multicastGrpMsg {
};

extern void CkPackMessage(envelope **pEnv);

mCastEntry::mCastEntry (mCastEntry *old): 
  oldc(NULL), newc(NULL), flag(COOKIE_NOTREADY)
{
  parentGrp = old->parentGrp;
  for (int i=0; i<old->allElem.length(); i++)
    allElem.push_back(old->allElem[i]);
  pe = old->pe;
  red.storedCallback = old->red.storedCallback;
  red.storedClient = old->red.storedClient;
  red.storedClientParam = old->red.storedClientParam;
  red.redNo = old->red.redNo;
  needRebuild = 0;
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
  initCookie(_id);
}

void CkMulticastMgr::setSection(CkSectionCookie &id)
{
  initCookie(id);
}

// this is used
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
  initCookie(_id);
}

void CkMulticastMgr::initCookie(CkSectionCookie s)
{
  mCastEntry *entry = (mCastEntry *)s.val; 
  int n = entry->allElem.length();
  DEBUGF(("init: %d elems %p\n", n, s.val));
  multicastSetupMsg *msg = new (n, n, 0) multicastSetupMsg;
  msg->nIdx = n;
  msg->parent = CkSectionCookie(NULL);
  msg->rootSid = s;
  msg->redNo = entry->red.redNo;
  CkArray *array = CProxy_ArrayBase(s.aid).ckLocalBranch();
  for (int i=0; i<n; i++) {
    msg->arrIdx[i] = entry->allElem[i];
    int ape = array->lastKnown(entry->allElem[i]);
    msg->lastKnown[i] = ape;
  }
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[CkMyPe()].setup(msg);
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
DEBUGF(("[%d] Free up on %p\n", CkMyPe(), sect));
    mCastEntry *oldc= sect->oldc;
    delete sect;
    sect = oldc;
  }
}

void CkMulticastMgr::setup(multicastSetupMsg *msg)
{
  int i,j;
  mCastEntry *entry;
  if (msg->parent.pe == CkMyPe()) entry = (mCastEntry *)msg->rootSid.val; //sid.val;
  else entry = new mCastEntry;
  entry->pe = CkMyPe();
  entry->rootSid = msg->rootSid;
  entry->parentGrp = msg->parent;
  DEBUGF(("[%d] setup: %p redNo: %d => %d with %d elems\n", CkMyPe(), entry, entry->red.redNo, msg->redNo, msg->nIdx));
  entry->red.redNo = msg->redNo;

  int numpes = CkNumPes();
  arrayIndexPosList *lists = new arrayIndexPosList[numpes];
  for (i=0; i<msg->nIdx; i++) {
    // msg->arrIdx[i] is local ?
    int lastKnown = msg->lastKnown[i];
    if (lastKnown == CkMyPe()) {
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
    if (i==CkMyPe()) continue;
    if (lists[i].length()) num++;
  }
  if (MAXMCASTCHILDREN <= 0) numchild = num;
  else numchild = num<MAXMCASTCHILDREN?num:MAXMCASTCHILDREN;

  entry->numChild = numchild;

  if (numchild) {
    arrayIndexPosList *slots = new arrayIndexPosList[numchild];
    num = 0;
    for (i=0; i<numpes; i++) {
      if (i==CkMyPe()) continue;
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
      DEBUGF(("[%d] call set up %d numelem:%d\n", CkMyPe(), childroot, n));
      mCastGrp[childroot].setup(m);
    }
    delete [] slots;
  }
  else {
    childrenReady(entry);
  }
  delete [] lists;
}

void CkMulticastMgr::childrenReady(mCastEntry *entry)
{
  entry->setReady();
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  DEBUGF(("[%d] entry %p childrenReady with %d elems.\n", CkMyPe(), entry, entry->allElem.length()));
  if (entry->hasParent()) {
    mCastGrp[entry->parentGrp.pe].recvCookie(entry->parentGrp, CkSectionCookie(entry));
  }
  // clear msg buffer
  while (!entry->msgBuf.isEmpty()) {
    multicastGrpMsg *newmsg = entry->msgBuf.deq();
    DEBUGF(("[%d] release buffer %p %d\n", CkMyPe(), newmsg, newmsg->ep));
    newmsg->_cookie.val = entry;
    mCastGrp[CkMyPe()].recvMsg(newmsg);
  }
  // release reduction msgs
  releaseFutureReduceMsgs(entry);
}

void CkMulticastMgr::recvCookie(CkSectionCookie sid, CkSectionCookie child)
{
  mCastEntry *entry = (mCastEntry *)sid.val;
  entry->children.push_back(child);
  if (entry->children.length() == entry->numChild) {
    childrenReady(entry);
  }
}

// when rebuilding, all multicast msgs will be buffered.
void CkMulticastMgr::rebuild(CkSectionCookie &sectId)
{
  // tear down old tree
  mCastEntry *curCookie = (mCastEntry*)sectId.val;
  CkAssert(curCookie->pe == CkMyPe());
  // make sure I am the newest one
  while (curCookie->newc) curCookie = curCookie->newc;
  if (curCookie->isObsolete()) return;

  mCastEntry *newCookie = new mCastEntry(curCookie);  // allocate table for this section

  // build a chain
  newCookie->oldc = curCookie;
  curCookie->newc = newCookie;

  sectId.val = newCookie;

  DEBUGF(("rebuild: redNo:%d oldc:%p newc;%p\n", newCookie->red.redNo, curCookie, newCookie));

  curCookie->setObsolete();

  resetCookie(sectId);
}

// mark old cookie spanning tree as old and 
// build a new one
void CkMulticastMgr::resetCookie(CkSectionCookie s)
{
  mCastEntry *newCookie = (mCastEntry*)s.val;
  mCastEntry *oldCookie = newCookie->oldc;

  // get rid of old one
  DEBUGF(("reset: oldc: %p\n", oldCookie));
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  int mype = CkMyPe();
  mCastGrp[mype].teardown(CkSectionCookie(mype, oldCookie, 0));

  // build a new one
  initCookie(s);
}

void CkMulticastMgr::ArraySectionSend(int ep,void *m, CkArrayID a, CkSectionCookie &s)
{
  DEBUGF(("ArraySectionSend\n"));

  if (s.pe == CkMyPe()) {
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

  register envelope *env = UsrToEnv(m);
  CkPackMessage(&env);
  m = EnvToUsr(env);
  multicastGrpMsg *msg = (multicastGrpMsg *)m;
  msg->aid = a;
  msg->_cookie = s;
  msg->ep = ep;

  if (s.pe == CkMyPe())
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
    DEBUGF(("entry not ready, enq buffer %p\n", msg));
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
  DEBUGF(("send to local %d\n", nLocal));
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
    CkAssert (entry->rootSid.pe == CkMyPe());
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

#if 0
CkReductionMsg* CkMcastReductionMsg::buildNew(int NdataSize,void *srcData,
                  CkReduction::reducerType reducer)
{
  CkMcastReductionMsg *newmsg = new (NdataSize, 0) CkMcastReductionMsg;
  newmsg->dataSize = NdataSize;
  memcpy(newmsg->data, srcData, NdataSize);
  newmsg->flag = 0;
  newmsg->redNo = 0;
  newmsg->gcounter = 0;
  return newmsg;
}
#endif

void CkMulticastMgr::setReductionClient(CProxySection_ArrayElement &proxy, CkCallback *cb)
{
  CkSectionCookie &id = proxy.ckGetSectionCookie();
  mCastEntry *entry = (mCastEntry *)id.val;
  entry->red.storedCallback = cb;
}

void CkMulticastMgr::setReductionClient(CProxySection_ArrayElement &proxy, redClientFn fn,void *param)
{
  CkSectionCookie &id = proxy.ckGetSectionCookie();
  mCastEntry *entry = (mCastEntry *)id.val;
  entry->red.storedClient = fn;
  entry->red.storedClientParam = param;
}

inline CkReductionMsg *CkMulticastMgr::buildContributeMsg(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &id, CkCallback &cb)
{
  CkReductionMsg *msg = CkReductionMsg::buildNew(dataSize, data);
  msg->reducer = type;
  msg->sid = id;
  msg->sourceFlag = 1;   // from array element
  msg->redNo = id.redNo;
  msg->gcount = 1;
  msg->rebuilt = (id.pe == CkMyPe())?0:1;
  msg->callback = cb;
  return msg;
}

void CkMulticastMgr::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &id)
{
  CkCallback cb;
  contribute(dataSize, data, type, id, cb);
}

void CkMulticastMgr::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionCookie &id, CkCallback &cb)
{
  if (id.val == NULL || id.redNo == -1) 
    CmiAbort("contribute: SectionID is not initialized\n");

  CkReductionMsg *msg = CkReductionMsg::buildNew(dataSize, data);
  msg->reducer = type;
  msg->sid = id;
  msg->sourceFlag = 1;   // from array element
  msg->redNo = id.redNo;
  msg->gcount = 1;
  msg->rebuilt = (id.pe == CkMyPe())?0:1;
  msg->callback = cb;

  id.redNo++;
  DEBUGF(("[%d] val: %d %p\n", CkMyPe(), id.pe, id.val));
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[id.pe].recvRedMsg(msg);
}

void CkMulticastMgr::recvRedMsg(CkReductionMsg *msg)
{
  int i;
  CkSectionCookie id = msg->sid;
  mCastEntry *entry = (mCastEntry *)id.val;

  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  int updateReduceNo = 0;

  // update entry if obsolete
  if (entry->isObsolete()) {
      // send up to root
    DEBUGF(("[%d] entry obsolete-send to root %d\n", CkMyPe(), entry->rootSid.pe));
    if (!entry->hasParent()) { //rootSid.pe == CkMyPe()) {
      // I am root, set to the new cookie if there is
      mCastEntry *newentry = entry->newc;
      while (newentry && newentry->newc) newentry=newentry->newc;
      entry = newentry;
      if (!entry || entry->isObsolete()) CmiAbort("Crazy!");
      msg->sourceFlag = 0;	     // indicate it is not on old spanning tree
      updateReduceNo = 1;  // reduce from old tree, new entry need update.
    }
    else {
      msg->sid = entry->rootSid;
      msg->sourceFlag = 0;
      mCastGrp[entry->rootSid.pe].recvRedMsg(msg);
      return;
    }
  }

  reductionInfo &redInfo = entry->red;

  DEBUGF(("[%d] msg %p red:%d, entry:%p redno:%d\n", CkMyPe(), msg, msg->redNo, entry, entry->red.redNo));
  // old message come, ignore
  if (msg->redNo < redInfo.redNo) {
  DEBUGF(("[%d] msg redNo:%d, %p, entry:%p redno:%d\n", CkMyPe(), msg->redNo, msg, entry, redInfo.redNo));
    CmiAbort("Could never happen! \n");
  }
  if (entry->notReady() || msg->redNo > redInfo.redNo) {
    DEBUGF(("[%d] Future redmsgs, buffered! msg:%p entry:%p \n", CkMyPe(), msg, entry));
    redInfo.futureMsgs.push_back(msg);
    return;
  }

  DEBUGF(("[%d] recvRedMsg flag:%d red:%d\n", CkMyPe(), msg->flag, redInfo.redNo));
  if (msg->sourceFlag == 1) redInfo.lcount ++;
  if (msg->sourceFlag == 2) redInfo.ccount ++;
  redInfo.gcount += msg->gcount;

  // buffer this msg
  // check first
  if (redInfo.msgs.length() && msg->dataSize != redInfo.msgs[0]->dataSize)
    CmiAbort("Reduction data are not of same length!");
  redInfo.msgs.push_back(msg);

  if (CkMyPe() == 0)
  DEBUGF(("[%d] lcount:%d-%d, ccount:%d-%d, gcount:%d-%d root:%d\n", CkMyPe(),entry->red.lcount,entry->localElem.length(), entry->red.ccount, entry->children.length(), entry->red.gcount, entry->allElem.length(), !entry->hasParent()));

  int currentTreeUp = 0;
  if (redInfo.lcount == entry->localElem.length() && 
      redInfo.ccount == entry->children.length())
      currentTreeUp = 1;
  int mixTreeUp = 0;
  if (!entry->hasParent() && redInfo.gcount == entry->allElem.length())
      mixTreeUp = 1;
  if (currentTreeUp || mixTreeUp)
  {
    int dataSize = msg->dataSize;
    CkReduction::reducerType reducer = msg->reducer;

    // reduce msgs
    CkReduction::reducerFn f= CkReduction::reducerTable[reducer];
    CkAssert(f != NULL);
    // check valid callback in msg and check if migration happened
    CkCallback msg_cb;
    int rebuilt = 0;
    for (i=0; i<redInfo.msgs.length(); i++) {
      if (redInfo.msgs[i]->rebuilt) rebuilt = 1;
      if (!redInfo.msgs[i]->callback.isInvalid()) msg_cb = redInfo.msgs[i]->callback;
    }
    CkReductionMsg *newmsg = (*f)(redInfo.msgs.length(), redInfo.msgs.getVec());
    // check if migration and free messages
    for (i=0; i<redInfo.msgs.length(); i++) {
      delete redInfo.msgs[i];
    }
    redInfo.msgs.length() = 0;

    int oldRedNo = redInfo.redNo;
    entry->incReduceNo();
    DEBUGF(("advanced entry:%p redNo: %d\n", entry, entry->red.redNo));
    if (updateReduceNo) mCastGrp[CkMyPe()].updateRedNo(entry, redInfo.redNo);

    if (entry->hasParent()) {
      // send up to parent
      newmsg->reducer = reducer;
      newmsg->sid = entry->parentGrp;
      newmsg->sourceFlag = 2;
      newmsg->redNo = oldRedNo;
      newmsg->gcount = redInfo.gcount;
      newmsg->rebuilt = rebuilt;
      newmsg->callback = msg_cb;
      DEBUGF(("send to parent %p: %d\n", entry->parentGrp.val, entry->parentGrp.pe));
      mCastGrp[entry->parentGrp.pe].recvRedMsg(newmsg);
    }
    else {   // root
      newmsg->sid = id;
      if (!msg_cb.isInvalid()) {
        msg_cb.send(newmsg);
      }
      else if (redInfo.storedCallback != NULL) {
        redInfo.storedCallback->send(newmsg);
      }
      else if (redInfo.storedClient != NULL) {
        redInfo.storedClient(id, redInfo.storedClientParam, dataSize,
	   newmsg->data);
        delete newmsg;
      } 
      else
	CmiAbort("Did you forget to register a reduction client?");

      DEBUGF(("currentTreeUp: %d entry:%p oldc: %p\n", currentTreeUp, entry, entry->oldc));
      if (currentTreeUp && entry->oldc) {
	// free old tree;
	mCastGrp[CkMyPe()].freeup(CkSectionCookie(id.pe, entry->oldc, 0));
	entry->oldc = NULL;
      }
      if (rebuilt) entry->needRebuild = 1;
    }

    // reset counters
    redInfo.lcount = redInfo.ccount = redInfo.gcount = 0;

    // release future msgs
    releaseFutureReduceMsgs(entry);
  }
}

void CkMulticastMgr::releaseFutureReduceMsgs(mCastEntryPtr entry)
{
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  for (int i=0; i<entry->red.futureMsgs.length(); i++) {
    DEBUGF(("releaseFutureReduceMsgs: %p\n", entry->red.futureMsgs[i]));
    mCastGrp[CkMyPe()].recvRedMsg(entry->red.futureMsgs[i]);
  }
  entry->red.futureMsgs.length() = 0;
}

// these messages have to be sent to root
void CkMulticastMgr::releaseBufferedReduceMsgs(mCastEntryPtr entry)
{
  int i;
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  for (i=0; i<entry->red.msgs.length(); i++) {
    DEBUGF(("releaseBufferedReduceMsgs: %p\n", entry->red.msgs[i]));
    entry->red.msgs[i]->sid = entry->rootSid;
    entry->red.msgs[i]->sourceFlag = 0;
    mCastGrp[entry->rootSid.pe].recvRedMsg(entry->red.msgs[i]);
  }
  entry->red.msgs.length() = 0;

  for (i=0; i<entry->red.futureMsgs.length(); i++) {
    DEBUGF(("releaseBufferedFutureReduceMsgs: %p\n", entry->red.futureMsgs[i]));
    entry->red.futureMsgs[i]->sid = entry->rootSid;
    entry->red.futureMsgs[i]->sourceFlag = 0;
    mCastGrp[entry->rootSid.pe].recvRedMsg(entry->red.futureMsgs[i]);
  }
  entry->red.futureMsgs.length() = 0;
}

void CkMulticastMgr::updateRedNo(mCastEntryPtr entry, int red)
{
  DEBUGF(("[%d] updateRedNo entry:%p to %d\n", CkMyPe(), entry, red));
  if (entry->red.redNo < red)
    entry->red.redNo = red;

  CProxy_CkMulticastMgr mp(thisgroup);
  for (int i=0; i<entry->children.length(); i++) {
    mp[entry->children[i].pe].updateRedNo((mCastEntry *)entry->children[i].val, red);
  }

  releaseFutureReduceMsgs(entry);
}

#if 0
////////////////////////////////////////////////////////////////////////////////
/////
///////////////// Builtin Reducer Functions //////////////
static CkReductionMsg *invalid_reducer(int nMsg,CkReductionMsg **msg)
{CkAbort("ERROR! Called the invalid reducer!\n");return NULL;}

/* A simple reducer, like sum_int, looks like this:
static CkReductionMsg *sum_int(int nMsg,CkReductionMsg **msg)
{
  int i,ret=0;
  for (i=0;i<nMsg;i++)
    ret+=*(int *)(msg[i]->data);
  return CkReductionMsg::buildNew(sizeof(int),(void *)&ret);
}
*/

#define SIMPLE_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg, CkReductionMsg **msg)\
{\
  int m,i;\
  int nElem=msg[0]->getSize()/sizeof(dataType);\
  dataType *ret=(dataType *)(msg[0]->getData());\
  for (m=1;m<nMsg;m++)\
  {\
    dataType *value=(dataType *)(msg[m]->getData());\
    for (i=0;i<nElem;i++)\
    {\
      loop\
    }\
  }\
  return CkReductionMsg::buildNew(nElem*sizeof(dataType),(void *)ret);\
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

CkReduction::reducerFn CkMulticastMgr::reducerTable[CkMulticastMgr::MAXREDUCERS]={
    ::invalid_reducer,
  //Compute the sum the numbers passed by each element.
    ::sum_int,::sum_float,::sum_double,
    ::product_int,::product_float,::product_double,
    ::max_int,::max_float,::max_double,
    ::min_int,::min_float,::min_double
};
#endif

#include "CkMulticast.def.h"
