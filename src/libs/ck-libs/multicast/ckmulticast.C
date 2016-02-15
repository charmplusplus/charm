/*
 *  Charm++ support for array section multicast and reduction
 *
 *  written by Gengbin Zheng,   gzheng@uiuc.edu
 *  on 12/2001
 *
 *  features:
 *     using a spanning tree (factor defined in ckmulticast.h)
 *     support pipelining via fragmentation  (SPLIT_MULTICAST)
 *     support *any-time* migration, spanning tree will be rebuilt automatically
 * */

#include "charm++.h"
#include "envelope.h"
#include "register.h"

#include "ckmulticast.h"
#include "spanningTreeStrategy.h"
#include "XArraySectionReducer.h"

#include <map>
#include <vector>

#define DEBUGF(x)  // CkPrintf x;

// turn on or off fragmentation in multicast
#if CMK_MESSAGE_LOGGING
#define SPLIT_MULTICAST  0
#else
#define SPLIT_MULTICAST  1
#endif

// maximum number of fragments into which a message can be broken
#define MAXFRAGS 100

typedef CkQ<multicastGrpMsg *>   multicastGrpMsgBuf;
typedef CkVec<CkArrayIndex>   arrayIndexList;
typedef CkVec<CkSectionInfo>     sectionIdList;
typedef CkVec<CkReductionMsg *>  reductionMsgs;
typedef CkQ<int>                 PieceSize;
typedef CkVec<LDObjid>          ObjKeyList;
typedef unsigned char            byte;

/** Information about the status of reductions proceeding along a given section
 *
 * An instance of this class is stored in every mCastEntry object making it possible
 * to track redn operations on a per section basis all along the spanning tree.
 */
class reductionInfo {
    public:
        /// Number of local array elements which have contributed a given fragment
        int lcount [MAXFRAGS];
        /// Number of child vertices (NOT array elements) that have contributed a given fragment
        int ccount [MAXFRAGS];
        /// The total number of array elements that have contributed so far to a given fragment
        int gcount [MAXFRAGS];
        /// The number of fragments processed so far
        int npProcessed;
        /// User callback
        CkCallback *storedCallback;
        /// User reduction client function
        redClientFn storedClient;
        /// User provided data for the redn client function
        void *storedClientParam;
        /// Reduction sequence number
        int redNo;
        /// Messages for this reduction
        reductionMsgs  msgs [MAXFRAGS];
        /// Messages of future reductions
        reductionMsgs futureMsgs;

    public:
        reductionInfo(): storedCallback(NULL),
                         storedClientParam(NULL),
                         redNo(0),
                         npProcessed(0) {
            for (int i=0; i<MAXFRAGS; i++)
                lcount [i] = ccount [i] = gcount [i] = 0;
        }
};

/// cookie status
#define COOKIE_NOTREADY 0
#define COOKIE_READY    1
#define COOKIE_OLD      2

class mCastPacket {
public:
  CkSectionInfo cookie;
  int offset;
  int n;
  char *data;
  int seqno;
  int count;
  int totalsize;

  mCastPacket(CkSectionInfo &_cookie, int _offset, int _n, char *_d, int _s, int _c, int _t):
		cookie(_cookie), offset(_offset), n(_n), data(_d), seqno(_s), count(_c), totalsize(_t) {}
};

typedef CkQ<mCastPacket *> multicastGrpPacketBuf;

class SectionLocation {
public:
  mCastEntry *entry;
  int         pe;
public:
  SectionLocation(): entry(NULL), pe(-1) {}
  SectionLocation( mCastEntry *e, int p) { set(e, p); }
  inline void set(mCastEntry *e, int p) { entry = e; pe = p; }
  inline void clear() { entry = NULL; pe = -1; }
};




/// Cookie for an array section 
class mCastEntry 
{
    public:
        /// Array ID 
        CkArrayID     aid;
        /// Spanning tree parent
        CkSectionInfo parentGrp;
        /// List of direct children
        sectionIdList children;
        /// Number of direct children
        int numChild;
        /// List of all tree member array indices (Only useful on the tree root)
        arrayIndexList allElem;
        /// Only useful on root for LB
        ObjKeyList     allObjKeys;
        /// List of array elements on local PE
        arrayIndexList localElem;
        /// Should always be myPE
        int pe;
        /// Section ID of the root
        CkSectionInfo rootSid;
        multicastGrpMsgBuf msgBuf;
        /// Buffer storing the pending packets
        multicastGrpPacketBuf packetBuf;
        /// For multicast packetization
        char *asm_msg;
        int   asm_fill;
        /// Linked list of entries on the same processor
        mCastEntry *oldc, *newc;
        /// Old spanning tree
        SectionLocation   oldtree;
        // for reduction
        reductionInfo red;
        //
        char needRebuild;
    private:
        char flag;
    public:
        mCastEntry(CkArrayID _aid): aid(_aid), numChild(0), asm_msg(NULL), asm_fill(0),
                    oldc(NULL), newc(NULL), needRebuild(0), flag(COOKIE_NOTREADY) {}
        mCastEntry(mCastEntry *);
        /// Check if this tree is only a branch and has a parent
        inline int hasParent() { return parentGrp.get_val()?1:0; }
        /// Is this tree obsolete?
        inline int isObsolete() { return (flag == COOKIE_OLD); }
        /// Make the current tree obsolete
        inline void setObsolete() { flag=COOKIE_OLD; }
        /// Check if this (branch of the) tree is ready for use
        inline int notReady() { return (flag == COOKIE_NOTREADY); }
        /// Mark this (branch of the) tree as ready for use
        inline void setReady() { flag=COOKIE_READY; }
        /// Increment the reduction number across the whole linked list of cookies
        inline void incReduceNo() {
            red.redNo ++;
            for (mCastEntry *next = newc; next; next=next->newc) 
                next->red.redNo++;
        }
        /// Get a handle on the array ID this tree is a member of
        inline CkArrayID getAid() { return aid; }
        inline int hasOldtree() { return oldtree.entry != NULL; }
        inline void print() {
            CmiPrintf("[%d] mCastEntry: %p, numChild: %d pe: %d flag: %d asm_msg:%p asm_fill:%d\n", CkMyPe(), this, numChild, pe, flag, asm_msg, asm_fill);
        }
};




class cookieMsg: public CMessage_cookieMsg {
public:
  CkSectionInfo cookie;
public:
  cookieMsg() {};
  cookieMsg(CkSectionInfo m): cookie(m) {};
};




/// multicast tree setup message
class multicastSetupMsg: public CMessage_multicastSetupMsg {
public:
  int  nIdx;
  CkArrayIndex *arrIdx;
  int      *lastKnown;
  CkSectionInfo parent;
  CkSectionInfo rootSid;
  int redNo;
};




/// message send in spanning tree
class multicastGrpMsg: public CkMcastBaseMsg, public CMessage_multicastGrpMsg {
};


extern void CkPackMessage(envelope **pEnv);
extern void CkUnpackMessage(envelope **pEnv);



void _ckMulticastInit(void)
{
/*
  CkDisableTracing(CkIndex_CkMulticastMgr::recvMsg(0));
  CkDisableTracing(CkIndex_CkMulticastMgr::recvRedMsg(0));
*/
}


mCastEntry::mCastEntry (mCastEntry *old): 
  numChild(0), oldc(NULL), newc(NULL), flag(COOKIE_NOTREADY)
{
  int i;
  aid = old->aid;
  parentGrp = old->parentGrp;
  for (i=0; i<old->allElem.length(); i++)
    allElem.push_back(old->allElem[i]);
#if CMK_LBDB_ON
  CmiAssert(old->allElem.length() == old->allObjKeys.length());
  for (i=0; i<old->allObjKeys.length(); i++)
    allObjKeys.push_back(old->allObjKeys[i]);
#endif
  pe = old->pe;
  red.storedCallback = old->red.storedCallback;
  red.storedClient = old->red.storedClient;
  red.storedClientParam = old->red.storedClientParam;
  red.redNo = old->red.redNo;
  needRebuild = 0;
  asm_msg = NULL;
  asm_fill = 0;
}

extern LDObjid idx2LDObjid(const CkArrayIndex &idx);    // cklocation.C




void CkMulticastMgr::setSection(CkSectionInfo &_id, CkArrayID aid, CkArrayIndex *al, int n)
{
    // Create a multicast entry
    mCastEntry *entry = new mCastEntry(aid);
    // Push all the section member indices into the entry
    for (int i=0; i<n; i++) {
        entry->allElem.push_back(al[i]);
#if CMK_LBDB_ON
        const LDObjid key = idx2LDObjid(al[i]);
        entry->allObjKeys.push_back(key);
#endif
    }
    //  entry->aid = aid;
    _id.get_aid() = aid;
    _id.get_val() = entry;		// allocate table for this section
    // 
    initCookie(_id);
}




void CkMulticastMgr::setSection(CkSectionInfo &id)
{
  initCookie(id);
}




/// @warning: This is deprecated
void CkMulticastMgr::setSection(CProxySection_ArrayElement &proxy)
{
  CkArrayID aid = proxy.ckGetArrayID();
  CkSectionInfo &_id = proxy.ckGetSectionInfo();

  mCastEntry *entry = new mCastEntry(aid);

  const CkArrayIndex *al = proxy.ckGetArrayElements();
  for (int i=0; i<proxy.ckGetNumElements(); i++) {
    entry->allElem.push_back(al[i]);
#if CMK_LBDB_ON
    const LDObjid key = idx2LDObjid(al[i]);
    entry->allObjKeys.push_back(key);
#endif
  }
  _id.get_type() = MulticastMsg;
  _id.get_aid() = aid;
  _id.get_val() = entry;		// allocate table for this section
  initCookie(_id);
}




void CkMulticastMgr::resetSection(CProxySection_ArrayElement &proxy)
{
  CkSectionInfo &info = proxy.ckGetSectionInfo();

  int oldpe = info.get_pe();
  if (oldpe == CkMyPe()) return;	// we don't have to recreate one

  CkArrayID aid = proxy.ckGetArrayID();
  CkSectionID *sid = proxy.ckGetSectionIDs();
  mCastEntry *entry = new mCastEntry(aid);

  mCastEntry *oldentry = (mCastEntry *)info.get_val();
  DEBUGF(("[%d] resetSection: old entry:%p new entry:%p\n", CkMyPe(), oldentry, entry));

  const CkArrayIndex *al = sid->_elems;
  CmiAssert(info.get_aid() == aid);
  prepareCookie(entry, *sid, al, sid->_nElems, aid);

  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

    // store old tree info
  entry->oldtree.set(oldentry, oldpe);

    // obsolete old tree
  mCastGrp[oldpe].retire(CkSectionInfo(oldpe, oldentry, 0, entry->getAid()), info);

  // find reduction number
  mCastGrp[oldpe].retrieveCookie(CkSectionInfo(oldpe, oldentry, 0, aid), info);
}




/// Build a mCastEntry object with relevant section info and set the section cookie to point to this object
void CkMulticastMgr::prepareCookie(mCastEntry *entry, CkSectionID &sid, const CkArrayIndex *al, int count, CkArrayID aid)
{
  for (int i=0; i<count; i++) {
    entry->allElem.push_back(al[i]);
#if CMK_LBDB_ON
    const LDObjid key = idx2LDObjid(al[i]);
    entry->allObjKeys.push_back(key);
#endif
  }
  sid._cookie.get_type() = MulticastMsg;
  sid._cookie.get_aid() = aid;
  sid._cookie.get_val() = entry;	// allocate table for this section
  sid._cookie.get_pe() = CkMyPe();
}




// this is used
void CkMulticastMgr::initDelegateMgr(CProxy *cproxy)
{
  CProxySection_ArrayBase *proxy = (CProxySection_ArrayBase *)cproxy;
  int numSubSections = proxy->ckGetNumSubSections();
  for (int i=0; i<numSubSections; i++)
  {
      CkArrayID aid = proxy->ckGetArrayIDn(i);
      mCastEntry *entry = new mCastEntry(aid);
      CkSectionID *sid = &( proxy->ckGetSectionID(i) );
      const CkArrayIndex *al = proxy->ckGetArrayElements(i);
      prepareCookie(entry, *sid, al, proxy->ckGetNumElements(i), aid);
      initCookie(sid->_cookie);
  }
}




void CkMulticastMgr::retrieveCookie(CkSectionInfo s, CkSectionInfo srcInfo)
{
  mCastEntry *entry = (mCastEntry *)s.get_val();
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  mCastGrp[srcInfo.get_pe()].recvCookieInfo(srcInfo, entry->red.redNo);
}

// now that we get reduction number from the old cookie,
// we continue to build the spanning tree
void CkMulticastMgr::recvCookieInfo(CkSectionInfo s, int red)
{
  mCastEntry *entry = (mCastEntry *)s.get_val();
  entry->red.redNo = red;

  initCookie(s);

  // TODO delete old tree
}




void CkMulticastMgr::initCookie(CkSectionInfo s)
{
    mCastEntry *entry = (mCastEntry *)s.get_val();
    int n = entry->allElem.length();
    DEBUGF(("init: %d elems %p\n", n, s.get_val()));
    // Create and initialize a setup message
    multicastSetupMsg *msg = new (n, n, 0) multicastSetupMsg;
    msg->nIdx = n;
    msg->parent = CkSectionInfo(entry->getAid());
    msg->rootSid = s;
    msg->redNo = entry->red.redNo;
    // Fill the message with the section member indices and their last known locations
    CkArray *array = CProxy_ArrayBase(s.get_aid()).ckLocalBranch();
    for (int i=0; i<n; i++) {
      msg->arrIdx[i] = entry->allElem[i];
      int ape = array->lastKnown(entry->allElem[i]);
      CmiAssert(ape >=0 && ape < CkNumPes());
      msg->lastKnown[i] = ape;
    }
    // Trigger the spanning tree build
    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    mCastGrp[CkMyPe()].setup(msg);
}




void CkMulticastMgr::teardown(CkSectionInfo cookie)
{
    int i;
    mCastEntry *sect = (mCastEntry *)cookie.get_val();
    // Mark this section as obsolete
    sect->setObsolete();
    // Release the buffered messages 
    releaseBufferedReduceMsgs(sect);
    // Propagate the teardown to each of your children
    CProxy_CkMulticastMgr mp(thisgroup);
    for (i=0; i<sect->children.length(); i++)
        mp[sect->children[i].get_pe()].teardown(sect->children[i]);
}




void CkMulticastMgr::retire(CkSectionInfo cookie, CkSectionInfo newroot)
{
    int i;
    mCastEntry *sect = (mCastEntry *)cookie.get_val();
    // Reset the root section info
    sect->rootSid = newroot;
    // Mark this section as obsolete
    sect->setObsolete();
    // Release the buffered messages 
    releaseBufferedReduceMsgs(sect);
    // Propagate the teardown to each of your children
    CProxy_CkMulticastMgr mp(thisgroup);
    for (i=0; i<sect->children.length(); i++)
        mp[sect->children[i].get_pe()].teardown(sect->children[i]);
}




void CkMulticastMgr::freeup(CkSectionInfo cookie)
{
  mCastEntry *sect = (mCastEntry *)cookie.get_val();
  CProxy_CkMulticastMgr mp(thisgroup);
  // Parse through all the section members on this PE and...
  while (sect) 
  {
      // Free their children
      for (int i=0; i<sect->children.length(); i++)
          mp[ sect->children[i].get_pe() ].freeup(sect->children[i]);
      // Free the cookie itself
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
    CkArrayID aid = msg->rootSid.get_aid();
    if (msg->parent.get_pe() == CkMyPe()) 
      entry = (mCastEntry *)msg->rootSid.get_val(); //sid.val;
    else 
      entry = new mCastEntry(aid);
    entry->aid = aid;
    entry->pe = CkMyPe();
    entry->rootSid = msg->rootSid;
    entry->parentGrp = msg->parent;

    DEBUGF(("[%d] setup: %p redNo: %d => %d with %d elems\n", CkMyPe(), entry, entry->red.redNo, msg->redNo, msg->nIdx));
    entry->red.redNo = msg->redNo;

    // Create a numPE sized array of vectors to hold the array elements in each PE
    int numpes = CkNumPes();
    std::map<int, std::vector<CkArrayIndex> > elemBins;
    // Sort each array index in the setup message based on last known location
    for (i=0; i<msg->nIdx; i++) 
    {
      int lastKnown = msg->lastKnown[i];
      // If msg->arrIdx[i] local, add it to a special local element list
      if (lastKnown == CkMyPe())
          entry->localElem.insertAtEnd(msg->arrIdx[i]);
      // else, add it to the list corresponding to its PE
      else
          elemBins[lastKnown].push_back(msg->arrIdx[i]);
    }

    CkVec<int> mySubTreePEs;
    mySubTreePEs.reserve(numpes);
    // The first PE in my subtree should be me, the tree root (as required by the spanning tree builder)
    mySubTreePEs.push_back(CkMyPe());
    // Identify the child PEs in the tree, ie the PEs with section members on them
    for (std::map<int, std::vector<CkArrayIndex> >::iterator itr = elemBins.begin();
         itr != elemBins.end(); ++itr)
        mySubTreePEs.push_back(itr->first);
    // The number of multicast children can be limited by the spanning tree factor 
    int num = mySubTreePEs.size() - 1, numchild = 0;
    if (factor <= 0) numchild = num;
    else numchild = num<factor?num:factor;
  
    entry->numChild = numchild;

    // If there are any children, go about building a spanning tree
    if (numchild) 
    {
        // Build the next generation of the spanning tree rooted at my PE
        int *peListPtr = mySubTreePEs.getVec();
        topo::SpanningTreeVertex *nextGenInfo;
        nextGenInfo = topo::buildSpanningTreeGeneration(peListPtr,peListPtr + mySubTreePEs.size(),numchild);
        numchild = nextGenInfo->childIndex.size();
        entry->numChild = numchild;

        CProxy_CkMulticastMgr  mCastGrp(thisgroup);

        // Ask each direct child to setup its subtree
        for (i=0; i < numchild; i++)
        {
            // Determine the indices of the first and last PEs in this branch of my sub-tree
            int childStartIndex = nextGenInfo->childIndex[i], childEndIndex;
            if (i < numchild-1)
                childEndIndex = nextGenInfo->childIndex[i+1];
            else
                childEndIndex = mySubTreePEs.size();

            // Find the total number of section member elements on this subtree
            int numSubTreeElems = 0;
            for (j = childStartIndex; j < childEndIndex; j++)
                numSubTreeElems += elemBins[ mySubTreePEs[j] ].size();

            // Prepare the setup msg intended for the child
            multicastSetupMsg *m = new (numSubTreeElems, numSubTreeElems, 0) multicastSetupMsg;
            m->parent = CkSectionInfo(aid, entry);
            m->nIdx = numSubTreeElems;
            m->rootSid = msg->rootSid;
            m->redNo = msg->redNo;

            // Give each child the number, indices and location of its children
            for (int j = childStartIndex, cnt = 0; j < childEndIndex; j++)
            {
                int childPE = mySubTreePEs[j];
                for (int k = 0; k < elemBins[childPE].size(); k++, cnt++)
                {
                    m->arrIdx[cnt]  = elemBins[childPE][k];
                    m->lastKnown[cnt] = childPE;
                }
            }

            int childroot = mySubTreePEs[childStartIndex];
            DEBUGF(("[%d] call set up %d numelem:%d\n", CkMyPe(), childroot, numSubTreeElems));
            // Send the message to the child
            mCastGrp[childroot].setup(m);
        }
        delete nextGenInfo;
    }
    // else, tell yourself that your children are ready
    else 
        childrenReady(entry);
    delete msg;
}




void CkMulticastMgr::childrenReady(mCastEntry *entry)
{
    // Mark this entry as ready
    entry->setReady();
    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    DEBUGF(("[%d] entry %p childrenReady with %d elems.\n", CkMyPe(), entry, entry->allElem.length()));
    if (entry->hasParent()) 
        mCastGrp[entry->parentGrp.get_pe()].recvCookie(entry->parentGrp, CkSectionInfo(entry->getAid(), entry));
#if SPLIT_MULTICAST
    // clear packet buffer
    while (!entry->packetBuf.isEmpty()) 
    {
        mCastPacket *packet = entry->packetBuf.deq();
        packet->cookie.get_val() = entry;
        mCastGrp[CkMyPe()].recvPacket(packet->cookie, packet->offset, packet->n, packet->data, packet->seqno, packet->count, packet->totalsize, 1);
        delete [] packet->data;
        delete packet;
    }
#endif
    // clear msg buffer
    while (!entry->msgBuf.isEmpty()) 
    {
        multicastGrpMsg *newmsg = entry->msgBuf.deq();
        DEBUGF(("[%d] release buffer %p ep:%d\n", CkMyPe(), newmsg, newmsg->ep));
        newmsg->_cookie.get_val() = entry;
        mCastGrp[CkMyPe()].recvMsg(newmsg);
    }
    // release reduction msgs
    releaseFutureReduceMsgs(entry);
}




void CkMulticastMgr::recvCookie(CkSectionInfo sid, CkSectionInfo child)
{
  mCastEntry *entry = (mCastEntry *)sid.get_val();
  entry->children.push_back(child);
  if (entry->children.length() == entry->numChild) {
    childrenReady(entry);
  }
}




// rebuild is called when root not migrated
// when rebuilding, all multicast msgs will be buffered.
void CkMulticastMgr::rebuild(CkSectionInfo &sectId)
{
  // tear down old tree
  mCastEntry *curCookie = (mCastEntry*)sectId.get_val();
  CkAssert(curCookie->pe == CkMyPe());
  // make sure I am the newest one
  while (curCookie->newc) curCookie = curCookie->newc;
  if (curCookie->isObsolete()) return;

  //CmiPrintf("tree rebuild\n");
  mCastEntry *newCookie = new mCastEntry(curCookie);  // allocate table for this section

  // build a chain
  newCookie->oldc = curCookie;
  curCookie->newc = newCookie;

  sectId.get_val() = newCookie;

  DEBUGF(("rebuild: redNo:%d oldc:%p newc;%p\n", newCookie->red.redNo, curCookie, newCookie));

  curCookie->setObsolete();

  resetCookie(sectId);
}

void CkMulticastMgr::resetCookie(CkSectionInfo s)
{
  mCastEntry *newCookie = (mCastEntry*)s.get_val();
  mCastEntry *oldCookie = newCookie->oldc;

  // get rid of old one
  DEBUGF(("reset: oldc: %p\n", oldCookie));
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  int mype = CkMyPe();
  mCastGrp[mype].teardown(CkSectionInfo(mype, oldCookie, 0, oldCookie->getAid()));

  // build a new one
  initCookie(s);
}

void CkMulticastMgr::SimpleSend(int ep,void *m, CkArrayID a, CkSectionID &sid, int opts)
{
  DEBUGF(("[%d] SimpleSend: nElems:%d\n", CkMyPe(), sid._nElems));
    // set an invalid cookie since we don't have it
  ((multicastGrpMsg *)m)->_cookie = CkSectionInfo(-1, NULL, 0, a);
  for (int i=0; i< sid._nElems-1; i++) {
     CProxyElement_ArrayBase ap(a, sid._elems[i]);
     void *newMsg=CkCopyMsg((void **)&m);
#if CMK_MESSAGE_LOGGING
	envelope *env = UsrToEnv(newMsg);
	env->flags = env->flags | CK_MULTICAST_MSG_MLOG;
#endif
     ap.ckSend((CkArrayMessage *)newMsg,ep,opts|CK_MSG_LB_NOTRACE);
  }
  if (sid._nElems > 0) {
     CProxyElement_ArrayBase ap(a, sid._elems[sid._nElems-1]);
     ap.ckSend((CkArrayMessage *)m,ep,opts|CK_MSG_LB_NOTRACE);
  }
}

void CkMulticastMgr::ArraySectionSend(CkDelegateData *pd,int ep,void *m, int nsid, CkSectionID *sid, int opts)
{
#if CMK_MESSAGE_LOGGING
	envelope *env = UsrToEnv(m);
	env->flags = env->flags | CK_MULTICAST_MSG_MLOG;
#endif

    for (int snum = 0; snum < nsid; snum++) {
        void *msgCopy = m;
        if (nsid - snum > 1)
            msgCopy = CkCopyMsg(&m);
        sendToSection(pd, ep, msgCopy, &(sid[snum]), opts);
    }
}



void CkMulticastMgr::sendToSection(CkDelegateData *pd,int ep,void *m, CkSectionID *sid, int opts)
{
  DEBUGF(("ArraySectionSend\n"));
  multicastGrpMsg *msg = (multicastGrpMsg *)m;
  msg->ep = ep;
  CkSectionInfo &s = sid->_cookie;
  mCastEntry *entry;

  // If this section is rooted at this PE
  if (s.get_pe() == CkMyPe()) {
    entry = (mCastEntry *)s.get_val();   
    if (NULL == entry)
      CmiAbort("Unknown array section, Did you forget to register the array section to CkMulticastMgr using setSection()?");

    // update entry pointer in case there is a newer one.
    if (entry->newc) {
      do { entry=entry->newc; } while (entry->newc);
      s.get_val() = entry;
    }

#if CMK_LBDB_ON
    // fixme: running obj?
    envelope *env = UsrToEnv(msg);
    const LDOMHandle &om = CProxy_ArrayBase(s.get_aid()).ckLocMgr()->getOMHandle();
    LBDatabaseObj()->MulticastSend(om,entry->allObjKeys.getVec(),entry->allObjKeys.size(),env->getTotalsize());
#endif

    // The first time we need to rebuild the spanning tree, we do p2p sends to refresh lastKnown
    if (entry->needRebuild == 1) {
      msg->_cookie = s;
      SimpleSend(ep, msg, s.get_aid(), *sid, opts);
      entry->needRebuild = 2;
      return;
    }
    // else the second time, we just rebuild cos now we'll have all the lastKnown PEs
    else if (entry->needRebuild == 2) rebuild(s);
  }
  // else, if the root has migrated, we have a sub-optimal mcast
  else {
    // fixme - in this case, not recorded in LB
    CmiPrintf("Warning: Multicast not optimized after multicast root migrated. \n");
  }

  // update cookie
  msg->_cookie = s;

#if SPLIT_MULTICAST
  // split multicast msg into SPLIT_NUM copies
  register envelope *env = UsrToEnv(m);
  CkPackMessage(&env);
  int totalsize = env->getTotalsize();
  int packetSize = 0;
  int totalcount = 0;
  if(totalsize < split_threshold){
    packetSize = totalsize;
    totalcount = 1;
  }else{
    packetSize = split_size;
    totalcount = totalsize/split_size;
    if(totalsize%split_size) totalcount++; 
    //packetSize = totalsize/SPLIT_NUM;
    //if (totalsize%SPLIT_NUM) packetSize ++;
    //totalcount = SPLIT_NUM;
  }
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  int sizesofar = 0;
  char *data = (char*) env;
  if (totalcount == 1) {
    // If the root of this section's tree is on this PE, then just propagate msg
    if (s.get_pe() == CkMyPe()) {
      CkUnpackMessage(&env);
      msg = (multicastGrpMsg *)EnvToUsr(env);
      recvMsg(msg);
    }
    // else send msg to root of section's spanning tree
    else {
      CProxy_CkMulticastMgr  mCastGrp(thisgroup);
      msg = (multicastGrpMsg *)EnvToUsr(env);
      mCastGrp[s.get_pe()].recvMsg(msg);
    }
    return;
  }
  for (int i=0; i<totalcount; i++) {
    int mysize = packetSize;
    if (mysize + sizesofar > totalsize) {
      mysize = totalsize-sizesofar;
    }
    //CmiPrintf("[%d] send to %d : mysize: %d total: %d \n", CkMyPe(), s.get_pe(), mysize, totalsize);
    mCastGrp[s.get_pe()].recvPacket(s, sizesofar, mysize, data, i, totalcount, totalsize, 0);
    sizesofar += mysize;
    data += mysize;
  }
  CmiFree(env);
#else
  if (s.get_pe() == CkMyPe()) {
    recvMsg(msg);
  }
  else {
    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    mCastGrp[s.get_pe()].recvMsg(msg);
  }
#endif
}

void CkMulticastMgr::recvPacket(CkSectionInfo &_cookie, int offset, int n, char *data, int seqno, int count, int totalsize, int fromBuffer)
{
  int i;
  mCastEntry *entry = (mCastEntry *)_cookie.get_val();


  if (!fromBuffer && (entry->notReady() || !entry->packetBuf.isEmpty())) {
    char *newdata = new char[n];
    memcpy(newdata, data, n);
    entry->packetBuf.enq(new mCastPacket(_cookie, offset, n, newdata, seqno, count, totalsize));
//CmiPrintf("[%d] Buffered recvPacket: seqno: %d %d frombuf:%d empty:%d entry:%p\n", CkMyPe(), seqno, count, fromBuffer, entry->packetBuf.isEmpty(),entry);
    return;
  }

//CmiPrintf("[%d] recvPacket ready: seqno: %d %d buffer: %d entry:%p\n", CkMyPe(), seqno, count, fromBuffer, entry);

  // send to spanning tree children
  // can not optimize using list send because the difference in cookie
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  for (i=0; i<entry->children.length(); i++) {
    mCastGrp[entry->children[i].get_pe()].recvPacket(entry->children[i], offset, n, data, seqno, count, totalsize, 0);
  }

  if (entry->asm_msg == NULL) {
    CmiAssert(entry->asm_fill == 0);
    entry->asm_msg = (char *)CmiAlloc(totalsize);
  }
  memcpy(entry->asm_msg+offset, data, n);
  entry->asm_fill += n;
  if (entry->asm_fill == totalsize) {
    CkUnpackMessage((envelope **)&entry->asm_msg);
    multicastGrpMsg *msg = (multicastGrpMsg *)EnvToUsr((envelope*)entry->asm_msg);
    msg->_cookie = _cookie;
//    mCastGrp[CkMyPe()].recvMsg(msg);
    sendToLocal(msg);
    entry->asm_msg = NULL;
    entry->asm_fill = 0;
  }
//  if (fromBuffer) delete [] data;
}

void CkMulticastMgr::recvMsg(multicastGrpMsg *msg)
{
  int i;
  CkSectionInfo &sectionInfo = msg->_cookie;
  mCastEntry *entry = (mCastEntry *)msg->_cookie.get_val();
  CmiAssert(entry->getAid() == sectionInfo.get_aid());

  if (entry->notReady()) {
    DEBUGF(("entry not ready, enq buffer %p\n", msg));
    entry->msgBuf.enq(msg);
    return;
  }

  // send to spanning tree children
  // can not optimize using list send because the difference in cookie
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);
  for (i=0; i<entry->children.length(); i++) {
    multicastGrpMsg *newmsg = (multicastGrpMsg *)CkCopyMsg((void **)&msg);
#if CMK_MESSAGE_LOGGING
	envelope *env = UsrToEnv(newmsg);
	env->flags = env->flags | CK_MULTICAST_MSG_MLOG;
#endif
    newmsg->_cookie = entry->children[i];
    mCastGrp[entry->children[i].get_pe()].recvMsg(newmsg);
  }

  sendToLocal(msg);
}

void CkMulticastMgr::sendToLocal(multicastGrpMsg *msg)
{
  int i;
  CkSectionInfo &sectionInfo = msg->_cookie;
  mCastEntry *entry = (mCastEntry *)msg->_cookie.get_val();
  CmiAssert(entry->getAid() == sectionInfo.get_aid());
  CkGroupID aid = sectionInfo.get_aid();

  // send to local
  int nLocal = entry->localElem.length();
  DEBUGF(("[%d] send to local %d elems\n", CkMyPe(), nLocal));
  for (i=0; i<nLocal-1; i++) {
    CProxyElement_ArrayBase ap(aid, entry->localElem[i]);
    if (_entryTable[msg->ep]->noKeep) {
      CkSendMsgArrayInline(msg->ep, msg, sectionInfo.get_aid(), entry->localElem[i], CK_MSG_KEEP);
    }
    else {
      // send through scheduler queue
      multicastGrpMsg *newm = (multicastGrpMsg *)CkCopyMsg((void **)&msg);
      ap.ckSend((CkArrayMessage *)newm, msg->ep, CK_MSG_LB_NOTRACE);
    }
    // use CK_MSG_DONTFREE so that the message can be reused
    // the drawback of this scheme bypassing queue is that 
    // if # of local element is huge, this leads to a long time occupying CPU
    // also load balancer seems not be able to correctly instrument load
//    CkSendMsgArrayInline(msg->ep, msg, msg->aid, entry->localElem[i], CK_MSG_KEEP);
    //CmiNetworkProgressAfter(3);
  }
  if (nLocal) {
    CProxyElement_ArrayBase ap(aid, entry->localElem[nLocal-1]);
    ap.ckSend((CkArrayMessage *)msg, msg->ep, CK_MSG_LB_NOTRACE);
//    CkSendMsgArrayInline(msg->ep, msg, msg->aid, entry->localElem[nLocal-1]);
  }
  else {
    CkAssert (entry->rootSid.get_pe() == CkMyPe());
    delete msg;
  }
}



void CkGetSectionInfo(CkSectionInfo &id, void *msg)
{
  CkMcastBaseMsg *m = (CkMcastBaseMsg *)msg;
  if (CkMcastBaseMsg::checkMagic(m) == 0) {
    CmiPrintf("ERROR: This is not a CkMulticast message!\n");
    CmiAbort("Did you remember to do CkMulticast delegation, and inherit multicast message from CkMcastBaseMsg in correct order?");
  }
  // ignore invalid cookie sent by SimpleSend
  if (m->gpe() != -1) {
    id.get_type() = MulticastMsg;
    id.get_pe() = m->gpe();
    id.get_val() = m->cookie();
  }
  // note: retain old redNo
}

// Reduction

void CkMulticastMgr::setReductionClient(CProxySection_ArrayElement &proxy, CkCallback *cb)
{
  CkCallback *sectionCB;
  int numSubSections = proxy.ckGetNumSubSections();
  // If its a cross-array section,
  if (numSubSections > 1)
  {
      /** @warning: Each instantiation is a mem leak! :o
       * The class is trivially small, but there's one instantiation for each
       * section delegated to CkMulticast. The objects need to live as long as
       * their section exists and is used. The idea of 'destroying' an array
       * section is still academic, and hence no effort has been made to charge
       * some 'owner' entity with the task of deleting this object.
       *
       * Reimplementing delegated x-array reductions will make this consideration moot
       */
      // Configure the final cross-section reducer
      ck::impl::XArraySectionReducer *red =
          new ck::impl::XArraySectionReducer(numSubSections, cb);
      // Configure the subsection callback to deposit with the final reducer
      sectionCB = new CkCallback(ck::impl::processSectionContribution, red);
  }
  // else, just direct the reduction to the actual client cb
  else
      sectionCB = cb;
  // Wire the sections together by storing the subsection cb in each sectionID
  for (int i=0; i<numSubSections; i++)
  {
      CkSectionInfo &sInfo = proxy.ckGetSectionID(i)._cookie;
      mCastEntry *entry = (mCastEntry *)sInfo.get_val();
      entry->red.storedCallback = sectionCB;
  }
}

void CkMulticastMgr::setReductionClient(CProxySection_ArrayElement &proxy, redClientFn fn,void *param)
{
  CkSectionInfo &id = proxy.ckGetSectionInfo();
  mCastEntry *entry = (mCastEntry *)id.get_val();
  entry->red.storedClient = fn;
  entry->red.storedClientParam = param;
}

inline CkReductionMsg *CkMulticastMgr::buildContributeMsg(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &id, CkCallback &cb, int userFlag)
{
  CkReductionMsg *msg = CkReductionMsg::buildNew(dataSize, data);
  msg->reducer = type;
  msg->sid = id;
  msg->sourceFlag = -1;   // from array element
  msg->redNo = id.get_redNo();
  msg->gcount = 1;
  msg->rebuilt = (id.get_pe() == CkMyPe())?0:1;
  msg->callback = cb;
  msg->userFlag=userFlag;
#if CMK_MESSAGE_LOGGING
  envelope *env = UsrToEnv(msg);
  env->flags = env->flags | CK_REDUCTION_MSG_MLOG;
#endif
  return msg;
}



void CkMulticastMgr::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &id, int userFlag, int fragSize)
{
  CkCallback cb;
  contribute(dataSize, data, type, id, cb, userFlag, fragSize);
}


void CkMulticastMgr::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &id, CkCallback &cb, int userFlag, int fragSize)
{
  if (id.get_val() == NULL || id.get_redNo() == -1) 
    CmiAbort("contribute: SectionID is not initialized\n");

  int nFrags;
  if (-1 == fragSize) {		// no frag
    nFrags = 1;
    fragSize = dataSize;
  }
  else {
    CmiAssert (dataSize >= fragSize);
    nFrags = dataSize/fragSize;
    if (dataSize%fragSize) nFrags++;
  }

  if (MAXFRAGS < nFrags) {
    CmiPrintf ("Recompile CkMulticast library for fragmenting msgs into more than %d fragments\n", MAXFRAGS);
    CmiAbort ("frag size too small\n");
  }

  int mpe = id.get_pe();
  CProxy_CkMulticastMgr  mCastGrp(thisgroup);

  // break the message into k-piece fragments
  int fSize = fragSize;
  for (int i=0; i<nFrags; i++) {
    if ((0 != i) && ((nFrags-1) == i) && (0 != dataSize%fragSize)) {
      fSize = dataSize%fragSize;
    }

    CkReductionMsg *msg = CkReductionMsg::buildNew(fSize, data);

    // initialize the new msg
    msg->reducer            = type;
    msg->sid                = id;
    msg->nFrags             = nFrags;
    msg->fragNo             = i;
    msg->sourceFlag         = -1;
    msg->redNo              = id.get_redNo();
    msg->gcount             = 1;
    msg->rebuilt            = (mpe == CkMyPe())?0:1;
    msg->callback           = cb;
    msg->userFlag           = userFlag;

#if CMK_MESSAGE_LOGGING
	envelope *env = UsrToEnv(msg);
	env->flags = env->flags | CK_REDUCTION_MSG_MLOG;
#endif

    mCastGrp[mpe].recvRedMsg(msg);

    data = (void*)(((char*)data) + fSize);
  }

  id.get_redNo()++;
  DEBUGF(("[%d] val: %d %p\n", CkMyPe(), id.get_pe(), id.get_val()));
}

CkReductionMsg* CkMulticastMgr::combineFrags (CkSectionInfo& id, 
                                              mCastEntry* entry,
                                              reductionInfo& redInfo) {
  int i;
  int dataSize = 0;
  int nFrags   = redInfo.msgs[0][0]->nFrags;

  // to avoid memcpy and allocation cost for non-pipelined reductions
  if (1 == nFrags) {
    CkReductionMsg* msg = redInfo.msgs[0][0];

    // free up the msg slot
    redInfo.msgs[0].length() = 0;

    return msg;
  }

  for (i=0; i<nFrags; i++) {
    dataSize += redInfo.msgs[i][0]->dataSize;
  }

  CkReductionMsg *msg = CkReductionMsg::buildNew(dataSize, NULL);
#if CMK_MESSAGE_LOGGING
  envelope *env = UsrToEnv(msg);
  env->flags = env->flags | CK_REDUCTION_MSG_MLOG;
#endif

  // initialize msg header
  msg->redNo      = redInfo.msgs[0][0]->redNo;
  msg->reducer    = redInfo.msgs[0][0]->reducer;
  msg->sid        = id;
  msg->nFrags     = nFrags;

  // I guess following fields need not be initialized
  msg->sourceFlag = 2;
  msg->rebuilt    = redInfo.msgs[0][0]->rebuilt;
  msg->callback   = redInfo.msgs[0][0]->callback;
  msg->userFlag   = redInfo.msgs[0][0]->userFlag;

  byte* data = (byte*)msg->getData ();
  for (i=0; i<nFrags; i++) {
    // copy data from fragments to msg
    memcpy(data, redInfo.msgs[i][0]->getData(), redInfo.msgs[i][0]->dataSize);
    data += redInfo.msgs[i][0]->dataSize;

    // free fragments
    delete redInfo.msgs[i][0];
    redInfo.msgs[i].length() = 0;    
  }

  return msg;
}



void CkMulticastMgr::reduceFragment (int index, CkSectionInfo& id,
                                     mCastEntry* entry, reductionInfo& redInfo,
                                     int currentTreeUp) {

    CProxy_CkMulticastMgr  mCastGrp(thisgroup);
    reductionMsgs& rmsgs = redInfo.msgs[index];
    int dataSize         = rmsgs[0]->dataSize;
    int i;
    int oldRedNo = redInfo.redNo;
    int nFrags   = rmsgs[0]->nFrags;
    int fragNo   = rmsgs[0]->fragNo;
    int userFlag = rmsgs[0]->userFlag;

    // Figure out (from one of the msg fragments) which reducer function to use
    CkReduction::reducerType reducer = rmsgs[0]->reducer;
    CkReduction::reducerFn f= CkReduction::reducerTable[reducer].fn;
    CkAssert(NULL != f);

    // Check if migration occurred in any of the subtrees, and pick one valid callback
    CkCallback msg_cb;
    int rebuilt = 0;
    for (i=0; i<rmsgs.length(); i++) {
        if (rmsgs[i]->rebuilt) rebuilt = 1;
        if (!rmsgs[i]->callback.isInvalid()) msg_cb = rmsgs[i]->callback;
    }

    // Perform the actual reduction
    CkReductionMsg *newmsg = (*f)(rmsgs.length(), rmsgs.getVec());
#if CMK_MESSAGE_LOGGING
	envelope *env = UsrToEnv(newmsg);
	env->flags = env->flags | CK_REDUCTION_MSG_MLOG;
#endif
    newmsg->redNo  = redInfo.redNo;
    newmsg->nFrags = nFrags;
    newmsg->fragNo = fragNo;
    newmsg->userFlag = userFlag;
    newmsg->reducer = reducer;

    // Increment the number of fragments processed
    redInfo.npProcessed ++;

    // Delete all the fragments which are no longer needed
    for (i=0; i<rmsgs.length(); i++)
        if (rmsgs[i]!=newmsg) delete rmsgs[i];
    rmsgs.length() = 0;

    // If I am not the tree root
    if (entry->hasParent()) {
        // send up to parent
        newmsg->sid        = entry->parentGrp;
        newmsg->sourceFlag = 2;
        newmsg->redNo      = oldRedNo; ///< @todo: redundant, duplicate assignment?
        newmsg->gcount     = redInfo.gcount [index];
        newmsg->rebuilt    = rebuilt;
        newmsg->callback   = msg_cb;
        DEBUGF(("[%d] ckmulticast: send %p to parent %d\n", CkMyPe(), entry->parentGrp.get_val(), entry->parentGrp.get_pe()));
        mCastGrp[entry->parentGrp.get_pe()].recvRedMsg(newmsg);
    } else {
        newmsg->sid = id;
        // Buffer the reduced fragment
        rmsgs.push_back (newmsg);
        // If all the fragments have been reduced
        if (redInfo.npProcessed == nFrags) {
            // Combine the fragments
            newmsg = combineFrags (id, entry, redInfo);
            // Set the reference number based on the user flag at the contribute call
            CkSetRefNum(newmsg, userFlag);
            // Trigger the appropriate reduction client
            if ( !msg_cb.isInvalid() )
                msg_cb.send(newmsg);
            else if (redInfo.storedCallback != NULL)
                redInfo.storedCallback->send(newmsg);
            else if (redInfo.storedClient != NULL) {
                redInfo.storedClient(id, redInfo.storedClientParam, dataSize, newmsg->data);
                delete newmsg;
            }
            else
                CmiAbort("Did you forget to register a reduction client?");

            DEBUGF(("ckmulticast: redn client called - currentTreeUp: %d entry:%p oldc: %p\n", currentTreeUp, entry, entry->oldc));
            //
            if (currentTreeUp) {
                if (entry->oldc) {
                    // free old tree on same processor;
                    mCastGrp[CkMyPe()].freeup(CkSectionInfo(id.get_pe(), entry->oldc, 0, entry->getAid()));
                    entry->oldc = NULL;
                }
                if (entry->hasOldtree()) {
                    // free old tree on old processor
                    int oldpe = entry->oldtree.pe;
                    mCastGrp[oldpe].freeup(CkSectionInfo(oldpe, entry->oldtree.entry, 0, entry->getAid()));
                    entry->oldtree.clear();
                }
            }
            // Indicate if a tree rebuild is required
            if (rebuilt && !entry->needRebuild) entry->needRebuild = 1;
        }
    }
}



/**
 * Called from:
 *   - contribute(): calls PE specified in the cookie
 *   - reduceFragment(): calls parent PE
 *   - recvRedMsg(): calls root PE (if tree is obsolete)
 *   - releaseFutureRedMsgs: calls this PE
 *   - releaseBufferedRedMsgs: calls root PE
 */
void CkMulticastMgr::recvRedMsg(CkReductionMsg *msg)
{
    int i;
    /// Grab the section info embedded in the redn msg
    CkSectionInfo id = msg->sid;
    /// ... and get at the ptr which shows me which cookie to use
    mCastEntry *entry = (mCastEntry *)id.get_val();
    CmiAssert(entry!=NULL);

    CProxy_CkMulticastMgr  mCastGrp(thisgroup);

    int updateReduceNo = 0;

    //-------------------------------------------------------------------------
    /// If this cookie is obsolete
    if (entry->isObsolete()) {
        // Send up to root
        DEBUGF(("[%d] ckmulticast: section cookie obsolete. Will send to root %d\n", CkMyPe(), entry->rootSid.get_pe()));

        /// If I am the root, traverse the linked list of cookies to get the latest
        if (!entry->hasParent()) {
            mCastEntry *newentry = entry->newc;
            while (newentry && newentry->newc) newentry=newentry->newc;
            if (newentry) entry = newentry;
            CmiAssert(entry!=NULL);
        }

        ///
        if (!entry->hasParent() && !entry->isObsolete()) {
            /// Indicate it is not on old spanning tree
            msg->sourceFlag = 0;
            /// Flag the redn as coming from an old tree and that the new entry cookie needs to know the new redn num.
            updateReduceNo  = 1;
        }
        /// If I am not the root or this latest cookie is also obsolete
        else {
            // Ensure that you're here with reason
            CmiAssert(entry->rootSid.get_pe() != CkMyPe() || entry->rootSid.get_val() != entry);
            // Edit the msg so that the recipient knows where to find its cookie
            msg->sid = entry->rootSid;
            msg->sourceFlag = 0;
            // Send the msg directly to the root of the redn tree
            mCastGrp[entry->rootSid.get_pe()].recvRedMsg(msg);
            return;
        }
    }

    /// Grab the locally stored redn info
    reductionInfo &redInfo = entry->red;

    //-------------------------------------------------------------------------
    /// If you've received a msg from a previous redn, something has gone horribly wrong somewhere!
    if (msg->redNo < redInfo.redNo) {
        CmiPrintf("[%d] msg redNo:%d, msg:%p, entry:%p redno:%d\n", CkMyPe(), msg->redNo, msg, entry, redInfo.redNo);
        CmiAbort("CkMulticast received a reduction msg with redNo less than the current redn number. Should never happen! \n");
    }

    //-------------------------------------------------------------------------
    /// If the current tree is not yet ready or if you've received a msg for a future redn, buffer the msg
    if (entry->notReady() || msg->redNo > redInfo.redNo) {
        DEBUGF(("[%d] Future redmsgs, buffered! msg:%p entry:%p ready:%d msg red:%d sys redno:%d\n", CkMyPe(), msg, entry, entry->notReady(), msg->redNo, redInfo.redNo));
        redInfo.futureMsgs.push_back(msg);
        return;
    }

    //-------------------------------------------------------------------------
    const int index = msg->fragNo;
    // New contribution from an ArrayElement
    if (msg->isFromUser()) {
        redInfo.lcount [index] ++;
    }
    // Redn from a child
    if (msg->sourceFlag == 2) {
        redInfo.ccount [index] ++;
    }
    // Total elems that have contributed the indexth fragment
    redInfo.gcount [index] += msg->gcount;

    //-------------------------------------------------------------------------
    // Buffer the msg
    redInfo.msgs [index].push_back(msg);

    //-------------------------------------------------------------------------
    /// Flag if this fragment can be reduced (if all local elements and children have contributed this fragment)
    int currentTreeUp = 0;
    if (redInfo.lcount [index] == entry->localElem.length() && redInfo.ccount [index] == entry->children.length())
        currentTreeUp = 1;

    /// Flag (only at the redn root) if all array elements contributed all their fragments
    int mixTreeUp = 0;
    if (!entry->hasParent()) {
        mixTreeUp = 1;
        for (int i=0; i<msg->nFrags; i++)
            if (entry->allElem.length() != redInfo.gcount [i])
                mixTreeUp = 0;
    }

    //-------------------------------------------------------------------------
    /// If this fragment can be reduced, or if I am the root and have received all fragments from all elements
    if (currentTreeUp || mixTreeUp)
    {
        const int nFrags = msg->nFrags;
        /// Reduce this fragment
        reduceFragment (index, id, entry, redInfo, currentTreeUp);

        // If migration happened, and my sub-tree reconstructed itself,
        // share the current reduction number with myself and all my children
        if (updateReduceNo)
            mCastGrp[CkMyPe()].updateRedNo(entry, redInfo.redNo);

        /// If all the fragments for the current reduction have been processed
        if (redInfo.npProcessed == nFrags) {

            /// Increment the reduction number in all of this section's cookies
            entry->incReduceNo();

            /// Reset bookkeeping counters
            for (i=0; i<nFrags; i++) {
                redInfo.lcount [i] = 0;
                redInfo.ccount [i] = 0;
                redInfo.gcount [i] = 0;
            }
            redInfo.npProcessed = 0;
            /// Now that, the current redn is done, release any pending msgs from future redns
            releaseFutureReduceMsgs(entry);
        }
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

  for (int j=0; j<MAXFRAGS; j++) {
    for (i=0; i<entry->red.msgs[j].length(); i++) {
      CkReductionMsg *msg = entry->red.msgs[j][i];
      DEBUGF(("releaseBufferedReduceMsgs:%p red:%d in entry:%p\n", msg, msg->redNo, entry));
      msg->sid = entry->rootSid;
      msg->sourceFlag = 0;
      mCastGrp[entry->rootSid.get_pe()].recvRedMsg(msg);
    }
    entry->red.msgs[j].length() = 0;
  }


  for (i=0; i<entry->red.futureMsgs.length(); i++) {
    CkReductionMsg *msg = entry->red.futureMsgs[i];
    DEBUGF(("releaseBufferedFutureReduceMsgs: %p red:%d in entry: %p\n", msg,msg->redNo, entry));
    msg->sid = entry->rootSid;
    msg->sourceFlag = 0;
    mCastGrp[entry->rootSid.get_pe()].recvRedMsg(msg);
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
    mp[entry->children[i].get_pe()].updateRedNo((mCastEntry *)entry->children[i].get_val(), red);
  }

  releaseFutureReduceMsgs(entry);
}

#include "CkMulticast.def.h"

