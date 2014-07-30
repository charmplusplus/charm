#ifndef CK_SMP_COORDINATOR_H
#define CK_SMP_COORDINATOR_H

#include "CkSmpCoordination.decl.h"
#include "CkSmpCoordinationMsg.h"
#include "ckmulticast.h"

extern CkReduction::reducerType CkSmpCoordinationReducerId;

// data structure that is used in reduction over all
// PEs in SMP node 
class CkSmpCoordinatorLeaderInfo {
  int pe_;
  int nObjs_;

  public:
  CkSmpCoordinatorLeaderInfo() : 
    pe_(-1),
    nObjs_(-1)
  {}

  CkSmpCoordinatorLeaderInfo(int pe, int nObjs) :
    pe_(pe),
    nObjs_(nObjs)
  {}

  const int &pe() const { return pe_; }
  const int &nObjs() const { return nObjs_; }

  int &pe() { return pe_; }
  int &nObjs() { return nObjs_; }

  CkSmpCoordinatorLeaderInfo &operator+=(const CkSmpCoordinatorLeaderInfo &other){
    if(other.nObjs() > nObjs()){
      *this = other; 
    }

    return *this;
  }
};

template<typename ClientType>
class CkSmpCoordinator : public CBase_CkSmpCoordinator<ClientType> {
  int nPesPerNode_;
  int firstPeInNode_;
  // leader pe for this round of execution 
  int leaderPe_;
  // am i the leader for my smp node?
  bool isLeader_;

  // saved callback; can be either registration or 
  // sync callback, depending on coordinator state
  CkCallback callback_;

  // section involving all pes on node
  // this is required for book-keeping operations such
  // as count
  CProxySection_CkSmpCoordinator<ClientType> allPesSection_;
  // each pe creates a different 'all-pes' section,
  // which has that pe as its root
  CkVec<CkSectionInfo> allPesCookies_;

  // during the setup phase, we should receive one
  // 'all-pes' section cookie from each pe in node
  int nCookiesRecvd_;

  int nSync_;
  int nExpectedSync_;

  // section involving only the non-empty pes
  // this is used when syncing cores on the node
  //CProxySection_CkSmpCoordinator nonEmptyPesSection_;
  //CkSectionInfo nonEmptyPesCookie_;

  CkGroupID mcastGid_;
  CkMulticastMgr *mcastMgr_;

  // how many objects did the client tell me are on 
  // this pe?
  int nObjects_;

  // registered client
  ClientType *client_;

  public:
  CkSmpCoordinator(CkGroupID mcastGid);
  CkSmpCoordinator(CkMigrateMessage *) {}
  void pup(PUP::er &p);

  // called before beginning computation, so that
  // cores can exchange cookies, and we don't
  // have to create a new 'all-pes' section each time
  // registration() is called
  void setup(const CkCallback &cb);
  // this is the message that conveys cookies
  void setup(CkSmpCoordinationLeaderMsg *msg);

  // non-entry method; 
  // called by user code at initialization time, and
  // every time load balancing causes objects to move
  // around; the registry then finds one non-empty core
  // on each smp node to act as the leader. 
  void registration(ClientType *client, int nObjects, const CkCallback &cb);

  // multicast sent by leader to ask all pes on node
  // to submit their object counts and identities
  //void count(CkSmpCoordinationMsg *msg);

  // result of 'count' reduction is passed to this 
  // entry method
  void findLeader(CkReductionMsg *msg);

  // used to tell new leader to take charge
  //void leader(CkVec<CkArrayIndex1D> &newSectionMembers);
  
  void announce(CkSmpCoordinationLeaderMsg *msg);

  public:
  // non-entry method; called by client when he wants to
  // synchronize all the pes on each smp node
  void sync(const CkCallback &cb);

  // entry method: called by leader so that even if
  // a pe's client doesn't invoke sync(), it can still
  // contribute to the synchronizing reduction
  void leaderSync(CkSmpCoordinationMsg *);

  // target of synchronizing reduction used above
  void syncDone();

  // entry method; when each pe on a node has either 
  // called sync(), or is empty (i.e. hosts no objects)
  // this method is invoked
  void invoke(CkSmpCoordinationMsg *dummy);

  // called by user code to check whether the core on
  // which this method is called is the leader for the
  // smp node
  bool isLeader() const;
  int getLeader() const;

  // called by user code to deliver msg from one 
  // member of set of pes on node, to all (including
  // self)
  template<typename PayloadType> 
  void mcast(const PayloadType &payload);

  // this is the entry method corresponding to the
  // above call, and is invoked on the recipients of
  // the above message
  template<typename PayloadType>
  void mcast(CkSmpCoordinationPayloadMsg<PayloadType> *msg);

  private:
  void reset();
  void contributeCount();
  void checkSyncDone();
  int toRank(int pe);
};


#define COORDINATOR_VERBOSE /*CkPrintf*/


#define CK_TEMPLATES_ONLY
#include "CkSmpCoordination.def.h"
#undef CK_TEMPLATES_ONLY

template<typename ClientType>
CkSmpCoordinator<ClientType>::CkSmpCoordinator(CkGroupID mcastGid)
{
  mcastGid_ = mcastGid;
  mcastMgr_ = CProxy_CkMulticastMgr(mcastGid_).ckLocalBranch();
  reset();
  //CkPrintf("[%d] CkSmpCoordinator::CkSmpCoordinator()\n", CkMyPe());
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::pup(PUP::er &p){
  CBase_CkSmpCoordinator<ClientType>::pup(p);
  p | allPesSection_;
  p | allPesCookies_;
  p | mcastGid_;
  if(p.isUnpacking()){
    reset();
    mcastMgr_ = CProxy_CkMulticastMgr(mcastGid_).ckLocalBranch();
    // should be set after registration call
    client_ = NULL;
  }
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::reset(){
  nPesPerNode_ = CkNumPes()/CkNumNodes();
  firstPeInNode_ = CkMyNode() * nPesPerNode_;
  leaderPe_ = firstPeInNode_;
  isLeader_ = (CkMyPe() == leaderPe_);
  nObjects_ = -1;
  this->setMigratable(false);
  allPesCookies_.resize(nPesPerNode_);
  nCookiesRecvd_ = 0;

  client_ = NULL;

  nSync_ = 0;
  nExpectedSync_ = -1;
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::setup(const CkCallback &cb){
  callback_ = cb;
  // create section containing all pes on node
  int startPe = CkMyNode() * nPesPerNode_; 
  int endPe = startPe + nPesPerNode_ - 1;
  allPesSection_ = CProxySection_CkSmpCoordinator<ClientType>::ckNew(this->thisProxy, startPe, endPe, 1);
  allPesSection_.ckSectionDelegate(mcastMgr_);
  allPesSection_.setup(new CkSmpCoordinationLeaderMsg(CkMyPe()));
}

template<typename ClientType>
int CkSmpCoordinator<ClientType>::toRank(int pe){
  return (pe - firstPeInNode_);
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::setup(CkSmpCoordinationLeaderMsg *msg){
  CkGetSectionInfo(allPesCookies_[toRank(msg->leaderPe)], msg);
  nCookiesRecvd_++;
  if(nCookiesRecvd_ == nPesPerNode_){
    nCookiesRecvd_ = 0;
    this->contribute(callback_);
  }
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::registration(ClientType *client, int nObjects, const CkCallback &cb){
  client_ = client;
  nObjects_ = nObjects;
  callback_ = cb;

  // must wait for leader to invoke sync()
  nExpectedSync_ = 1;
  // wait for client to call sync() only if there are objects on it
  if(nObjects_ > 0) nExpectedSync_++;
  
  COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::registration() leader %d nSync %d nExpectedSync %d\n", CkMyPe(), leaderPe_, nSync_, nExpectedSync_);
  contributeCount();
}

/*
void CkSmpCoordinator::count(CkSmpCoordinationMsg *msg){
  // currently, all pes section is reconstructed on each
  // call to registration(); therefore, we must rset
  // the reduction number of the cookie before saving
  // the metadata for the (new) section
  // XXX in principle, we could reduce this overhead by 
  // remembering the previous leader; if the previous and
  // current leader are the same, and the current leader
  // has already created the all-pes section, there should
  // be no reconstruction of the spanning tree, and no
  // resetting of the reduction number
  allPesCookie_ = CkSectionInfo();
  CkGetSectionInfo(allPesCookie_, msg);

  delete msg;

  if(okToContributeCount()){
    contributeCount();
  }
}

bool CkSmpCoordinator::okToContributeCount (){
}
*/

template<typename ClientType>
void CkSmpCoordinator<ClientType>::contributeCount(){
  CkSmpCoordinatorLeaderInfo info(CkMyPe(), nObjects_);

  CkCallback cb(CkIndex_CkSmpCoordinator<ClientType>::findLeader(NULL), leaderPe_, this->thisProxy);
  
  COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::contributeCount() nObjects %d\n", CkMyPe(), nObjects_);
  
  //CkPrintf("[%d] CkSmpCoordinator::contributeCount nbytes %d\n", CkMyPe(), sizeof(CkSmpCoordinatorLeaderInfo));
  mcastMgr_->contribute(sizeof(info), &info, CkSmpCoordinationReducerId, allPesCookies_[toRank(leaderPe_)], cb);

  /*
  if(nObjects_ == 0){
    // this pe will excluded from nonEmptyPesSection, 
    // and will not receive 'announce' from leader; therefore,

    // don't know leader's identity
    leaderPe_ = -1;
    CkPrintf("[%d] CkSmpCoordinator::contributeCount() newLeader UNKNOWN\n", CkMyPe());
    // but am sure that it can't be me
    isLeader_ = false;
    // return control to the client
    callback_.send();
  }
  */
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::findLeader(CkReductionMsg *msg){
  CkSmpCoordinatorLeaderInfo *result = (CkSmpCoordinatorLeaderInfo *) msg->getData();
  CkAssert(msg->getSize() == sizeof(CkSmpCoordinatorLeaderInfo));

  CkAssert(result->pe() >= 0);
  CkAssert(result->nObjs() >= 0);
  allPesSection_.announce(new CkSmpCoordinationLeaderMsg(result->pe()));
  delete msg;
}

/*
void CkSmpCoordinator::leader(CkVec<CkArrayIndex1D> &newSectionMembers){
  // create a new section and announce your leadership
  // to its members
  // XXX this needn't always hold
  CkAssert(newSectionMembers.size() > 0);
  nonEmptyPesSection_ = CProxySection_CkSmpCoordinator::ckNew(thisProxy, &newSectionMembers[0], newSectionMembers.size());
  nonEmptyPesSection_.ckSectionDelegate(mcastMgr_);
  CkPrintf("[%d] CkSmpCoordinator::leader()\n", CkMyPe());
  nonEmptyPesSection_.announce(new CkSmpCoordinationLeaderMsg(CkMyPe()));
}
*/

template<typename ClientType>
void CkSmpCoordinator<ClientType>::announce(CkSmpCoordinationLeaderMsg *msg){
  leaderPe_ = msg->leaderPe;
  isLeader_ = (leaderPe_ == CkMyPe());
  // since the members of this array do not migrate, 
  // we don't need to update the section info
  //CkGetSectionInfo(nonEmptyPesCookie_, msg);

  delete msg;

  COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::announce() newLeader %d\n", CkMyPe(), leaderPe_);
  callback_.send();
}

template<typename ClientType>
bool CkSmpCoordinator<ClientType>::isLeader() const {
  return isLeader_;
}

template<typename ClientType>
int CkSmpCoordinator<ClientType>::getLeader() const {
  return leaderPe_;
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::sync(const CkCallback &cb){
  callback_ = cb;
  if (nObjects_ > 0) {
    nSync_++;
  }
  COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::sync() nSync %d nExpected %d\n", CkMyPe(), nSync_, nExpectedSync_);
  /*
  CkCallback syncDoneCb(CkIndex_CkSmpCoordinator::syncDone(), leaderPe_, thisProxy);
  mcastMgr_->contribute(0, NULL, CkReduction::sum_int, nonEmptyPesCookie_, syncDoneCb);
  */
  if(isLeader_){
    COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::sync() sending leaderSync\n", CkMyPe());
    allPesSection_.leaderSync(new CkSmpCoordinationMsg);
  }

  checkSyncDone();
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::leaderSync(CkSmpCoordinationMsg *msg){
  delete msg;
  nSync_++;
  COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::leaderSync() nSync %d nExpected %d\n", CkMyPe(), nSync_, nExpectedSync_);
  checkSyncDone();
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::checkSyncDone(){
  if(nSync_ == nExpectedSync_){
    nSync_ = 0;
    CkCallback syncDoneCb(CkIndex_CkSmpCoordinator<ClientType>::syncDone(), leaderPe_, this->thisProxy);
    mcastMgr_->contribute(0, NULL, CkReduction::sum_int, allPesCookies_[toRank(leaderPe_)], syncDoneCb);
  }
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::syncDone(){
  allPesSection_.invoke(new CkSmpCoordinationMsg);
}

template<typename ClientType>
void CkSmpCoordinator<ClientType>::invoke(CkSmpCoordinationMsg *dummy){
  if(nObjects_ > 0){
    COORDINATOR_VERBOSE("[%d] CkSmpCoordinator::invokeCallback()\n", CkMyPe());
    callback_.send(); 
  }
  delete dummy;
}

template<typename ClientType>
template<typename PayloadType>
void CkSmpCoordinator<ClientType>::mcast(const PayloadType &payload){
  allPesSection_.mcast(new CkSmpCoordinationPayloadMsg<PayloadType>(payload));
}

template<typename ClientType>
template<typename PayloadType>
void CkSmpCoordinator<ClientType>::mcast(CkSmpCoordinationPayloadMsg<PayloadType> *msg){
  client_->mcast(msg->payload);
  delete msg;
}


#endif // CK_SMP_COORDINATOR_H
