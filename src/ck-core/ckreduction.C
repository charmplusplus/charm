/*
Parallel Programming Lab, University of Illinois at Urbana-Champaign
Orion Sky Lawlor, 3/29/2000, olawlor@acm.org

A reduction takes some sort of inputs (contributions)
from some set of objects (contributors) scattered across
all PE's and combines (reduces) all the contributions
onto one PE.  This library provides several different
kinds of combination routines (reducers), and all the
support framework for calling them.

Included here are the classes:
  -CkReduction, which gives the user-visible names as
an enumeration for the reducer functions, and maintains the
reducer table. (don't instantiate these)
  -CkReductionMgr, a Chare Group which actually runs a
reduction over a dynamic set (allowing insertions, deletions, and
migrations) of contributors scattered across all PE's.  It can
handle several overlapping reductions, but they will complete
serially. It carries out the reduction among all elements on
a processor.
  -CkReductionMsg, the message carrying reduction data
used by the reduction manager.
   -CkNodeReductionMgr, a Chare Node Group runs reductions
on node groups. It is used by the CkReductionMgr to carry out
the reduction among different nodes.   

In the reduction manager, there are several counters used:
  -reductionMgr::redNo is a sequential reduction count, starting
at zero for the first reduction.  When a reduction completes, it increments
redNo.
  -contributorInfo::redNo is the direct analog for contributors--
it starts at zero and is incremented at each contribution.  Hence
contributorInfo::redNo leads the local reductionMgr::redNo.
  -lcount is the number of contributors on this PE.  When
an element migrates away, lcount decreases.  lcount is also the number
of contributions to wait for before reducing and sending up.
  -gcount is the net birth-death contributor count on this PE.
When a contributor migrates away, gcount stays the same.  Unlike lcount,
gcount can go negative (if, e.g., a contributor migrates in and then dies).

We need a separate gcount because for a short time, a migrant
is local to no PE.  To make sure we get its contribution, node zero
compares its number of received contributions to gcount summed over all PE's
(this count is piggybacked with the reduction data in CkReductionMsg).
If we haven't gotten a contribution from all living contributors, node zero
waits for the migrant contributions to straggle in.

*/
#include "charm++.h"
#include "ck.h"

#include "pathHistory.h"

#if CMK_DEBUG_REDUCTIONS
//Debugging messages:
// Reduction mananger internal information:
#define DEBR(x) CkPrintf x
#define AA "Red PE%d Node%d #%d (%d,%d) Group %d> "
#define AB ,CkMyPe(),CkMyNode(),redNo,nRemote,nContrib,thisgroup.idx

#define DEBN(x) CkPrintf x
#define AAN "Red Node%d "
#define ABN ,CkMyNode()

// For status and data messages from the builtin reducer functions.
#define RED_DEB(x) //CkPrintf x
#define DEBREVAC(x) CkPrintf x
#define DEB_TUPLE(x) CkPrintf x
#else
//No debugging info-- empty defines
#define DEBR(x) // CkPrintf x
#define DEBRMLOG(x) CkPrintf x
#define AA
#define AB
#define DEBN(x) //CkPrintf x
#define RED_DEB(x) //CkPrintf x
#define DEBREVAC(x) //CkPrintf x
#define DEB_TUPLE(x) //CkPrintf x
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

extern int _inrestart;

Group::Group()
  : CkReductionMgr(CkpvAccess(_currentGroupRednMgr))
{
	if (_inrestart) CmiAbort("A Group object did not call the migratable constructor of its base class!");

	creatingContributors();
	contributorStamped(&reductionInfo);
	contributorCreated(&reductionInfo);
	doneCreatingContributors();
#if !GROUP_LEVEL_REDUCTION
	DEBR(("[%d,%d]Creating nodeProxy with gid %d\n",CkMyNode(),CkMyPe(),CkpvAccess(_currentGroupRednMgr)));
#endif
}

Group::Group(CkMigrateMessage *msg):CkReductionMgr(msg)
{
	creatingContributors();
	contributorStamped(&reductionInfo);
	contributorCreated(&reductionInfo);
	doneCreatingContributors();
}

CK_REDUCTION_CONTRIBUTE_METHODS_DEF(Group,
				    ((CkReductionMgr *)this),
				    reductionInfo,false)
CK_REDUCTION_CLIENT_DEF(CProxy_Group,(CkReductionMgr *)CkLocalBranch(_ck_gid))

CK_BARRIER_CONTRIBUTE_METHODS_DEF(Group,
                                   ((CkReductionMgr *)this),
                                   reductionInfo,false)



CkGroupInitCallback::CkGroupInitCallback(void) {}
/*
The callback is just used to tell the caller that this group
has been constructed.  (Now they can safely call CkLocalBranch)
*/
void CkGroupInitCallback::callMeBack(CkGroupCallbackMsg *m)
{
  m->call();
  delete m;
}

/*
The callback is just used to tell the caller that this group
is constructed and ready to process other calls.
*/
CkGroupReadyCallback::CkGroupReadyCallback(void)
{
  _isReady = 0;
}
void
CkGroupReadyCallback::callBuffered(void)
{
  int n = _msgs.length();
  for(int i=0;i<n;i++)
  {
    CkGroupCallbackMsg *msg = _msgs.deq();
    msg->call();
    delete msg;
  }
}
void
CkGroupReadyCallback::callMeBack(CkGroupCallbackMsg *msg)
{
  if(_isReady) {
    msg->call();
    delete msg;
  } else {
    _msgs.enq(msg);
  }
}

CkReductionClientBundle::CkReductionClientBundle(CkReductionClientFn fn_,void *param_)
	:CkCallback(callbackCfn,(void *)this),fn(fn_),param(param_) {}
void CkReductionClientBundle::callbackCfn(void *thisPtr,void *reductionMsg)
{
	CkReductionClientBundle *b=(CkReductionClientBundle *)thisPtr;
	CkReductionMsg *m=(CkReductionMsg *)reductionMsg;
	b->fn(b->param,m->getSize(),m->getData());
	delete m;
}

///////////////// Reduction Manager //////////////////
/*
One CkReductionMgr runs a non-overlapping set of reductions.
It collects messages from all local contributors, then sends
the reduced message up the reduction tree to node zero, where
they're passed to the user's client function.
*/

CkReductionMgr::CkReductionMgr(CProxy_CkArrayReductionMgr groupRednMgr)
  :
#if !GROUP_LEVEL_REDUCTION
  nodeProxy(groupRednMgr),
#endif
  thisProxy(thisgroup),
  isDestroying(false)
{ 
#ifdef BINOMIAL_TREE
  init_BinomialTree();
#else
  init_BinaryTree();
#endif
  redNo=0;
  completedRedNo = -1;
  inProgress=false;
  creating=false;
  startRequested=false;
  gcount=lcount=0;
  nContrib=nRemote=0;
  is_inactive = false;
  maxStartRequest=0;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	numImmigrantRecObjs = 0;
	numEmigrantRecObjs = 0;
#endif
  disableNotifyChildrenStart = false;

  barrier_gCount=0;
  barrier_nSource=0;
  barrier_nContrib=barrier_nRemote=0;

  DEBR((AA "In reductionMgr constructor at %d \n" AB,this));
}

CkReductionMgr::CkReductionMgr(CkMigrateMessage *m) :CkGroupInitCallback(m)
                                                    , isDestroying(false)
{
  numKids = -1;
  redNo=0;
  completedRedNo = -1;
  inProgress=false;
  creating=false;
  startRequested=false;
  gcount=lcount=0;
  nContrib=nRemote=0;
  is_inactive = false;
  maxStartRequest=0;
  DEBR((AA "In reductionMgr migratable constructor at %d \n" AB,this));

  barrier_gCount=0;
  barrier_nSource=0;
  barrier_nContrib=barrier_nRemote=0;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  numImmigrantRecObjs = 0;
  numEmigrantRecObjs = 0;
#endif

}

CkReductionMgr::~CkReductionMgr()
{
#if !GROUP_LEVEL_REDUCTION
  if (CkMyRank() == 0) {
    delete nodeProxy.ckLocalBranch();
  }
#endif
}

void CkReductionMgr::flushStates()
{
  // CmiPrintf("[%d] CkReductionMgr::flushState\n", CkMyPe());
  redNo=0;
  completedRedNo = -1;
  inProgress=false;
  creating=false;
  startRequested=false;
  nContrib=nRemote=0;
  maxStartRequest=0;

  while (!msgs.isEmpty()) { delete msgs.deq(); }
  while (!futureMsgs.isEmpty()) delete futureMsgs.deq();
  while (!futureRemoteMsgs.isEmpty()) delete futureRemoteMsgs.deq();
  while (!finalMsgs.isEmpty()) delete finalMsgs.deq();

  adjVec.length()=0;

#if ! GROUP_LEVEL_REDUCTION
  nodeProxy[CkMyNode()].ckLocalBranch()->flushStates();
#endif
}

//////////// Reduction Manager Client API /////////////

//Add the given client function.  Overwrites any previous client.
void CkReductionMgr::ckSetReductionClient(CkCallback *cb)
{
  DEBR((AA "Setting reductionClient in ReductionMgr groupid %d\n" AB,thisgroup.idx));

  if (CkMyPe()!=0)
	  CkError("WARNING: ckSetReductionClient should only be called from processor zero!\n");  
  storedCallback=*cb;
#if ! GROUP_LEVEL_REDUCTION
  CkCallback *callback =new CkCallback(CkIndex_CkReductionMgr::ArrayReductionHandler(0),thishandle);
  nodeProxy.ckSetReductionClient(callback);
#endif
}

///////////////////////////// Contributor ////////////////////////
//Contributors keep a copy of this structure:

/*Contributor migration support:
*/
void contributorInfo::pup(PUP::er &p)
{
  p(redNo);
}

////////////////////// Contributor list maintainance: /////////////////
//These just set and clear the "creating" flag to prevent
// reductions from finishing early because not all elements
// have been created.
void CkReductionMgr::creatingContributors(void)
{
  DEBR((AA "Creating contributors...\n" AB));
  creating=true;
}
void CkReductionMgr::doneCreatingContributors(void)
{
  DEBR((AA "Done creating contributors...\n" AB));
  creating=false;
  checkIsActive();
  if (startRequested) startReduction(redNo,CkMyPe());
  finishReduction();
}

//A new contributor will be created
void CkReductionMgr::contributorStamped(contributorInfo *ci)
{
  DEBR((AA "Contributor %p stamped\n" AB,ci));
  //There is another contributor
  gcount++;
  if (inProgress)
  {
    ci->redNo=redNo+1;//Created *during* reduction => contribute to *next* reduction
    adj(redNo).gcount--;//He'll wrongly be counted in the global count at end
  } else
    ci->redNo=redNo;//Created *before* reduction => contribute to *that* reduction
}

//A new contributor was actually created
void CkReductionMgr::contributorCreated(contributorInfo *ci)
{
  DEBR((AA "Contributor %p created in grp %d\n" AB,ci,thisgroup.idx));
  //We've got another contributor
  lcount++;
  //He may not need to contribute to some of our reductions:
  for (int r=redNo;r<ci->redNo;r++)
    adj(r).lcount--;//He won't be contributing to r here
  checkIsActive();
}

/*Don't expect any more contributions from this one.
This is rather horrifying because we now have to make
sure the global element count accurately reflects all the
contributions the element made before it died-- these may stretch
far into the future.  The adj() vector is what saves us here.
*/
void CkReductionMgr::contributorDied(contributorInfo *ci)
{
#if CMK_MEM_CHECKPOINT
  // ignore from listener if it is during restart from crash
  if (CkInRestarting()) return;
#endif

  if (isDestroying) return;

  DEBR((AA "Contributor %p(%d) died\n" AB,ci,ci->redNo));
  //We lost a contributor
  gcount--;

  if (ci->redNo<redNo)
  {//Must have been migrating during reductions-- root is waiting for his
  // contribution, which will never come.
    DEBR((AA "Dying guy %p must have been migrating-- he's at #%d!\n" AB,ci,ci->redNo));
    for (int r=ci->redNo;r<redNo;r++)
      thisProxy[0].MigrantDied(new CkReductionNumberMsg(r));
  }

  //Add to the global count for all his future messages (wherever they are)
  int r;
  for (r=redNo;r<ci->redNo;r++)
  {//He already contributed to this reduction, but won't show up in global count.
    DEBR((AA "Dead guy %p left contribution for #%d\n" AB,ci,r));
    adj(r).gcount++;
  }

  lcount--;
  //He's already contributed to several reductions here
  for (r=redNo;r<ci->redNo;r++)
    adj(r).lcount++;//He'll be contributing to r here

  // Check whether the death of this contributor made this pe go barren at this
  // redNo
  if (ci->redNo <= redNo) {
    checkIsActive();
  }
  finishReduction();
}

//Migrating away (note that global count doesn't change)
void CkReductionMgr::contributorLeaving(contributorInfo *ci)
{
  DEBR((AA "Contributor %p(%d) migrating away\n" AB,ci,ci->redNo));
  lcount--;//We lost a local
  //He's already contributed to several reductions here
  for (int r=redNo;r<ci->redNo;r++)
    adj(r).lcount++;//He'll be contributing to r here

  // Check whether this made this pe go barren at redNo
  if (ci->redNo <= redNo) {
    checkIsActive();
  }
  finishReduction();
}

//Migrating in (note that global count doesn't change)
void CkReductionMgr::contributorArriving(contributorInfo *ci)
{
  DEBR((AA "Contributor %p(%d) migrating in\n" AB,ci,ci->redNo));
  lcount++;//We gained a local
#if CMK_MEM_CHECKPOINT
  // ignore from listener if it is during restart from crash
  // because the ci may be old.
  if (CkInRestarting()) return;
#endif
  //He has already contributed (elsewhere) to several reductions:
  for (int r=redNo;r<ci->redNo;r++)
    adj(r).lcount--;//He won't be contributing to r here

  // Check if the arrival of a new contributor makes this PE become active again
  if (ci->redNo == redNo) {
    checkIsActive();
  }
}

//Contribute-- the given msg can contain any data.  The reducerType
// field of the message must be valid.
// Each contributor must contribute exactly once to the each reduction.
void CkReductionMgr::contribute(contributorInfo *ci,CkReductionMsg *m)
{
#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&(m->log));
#endif
  DEBR((AA "Contributor %p contributed for %d in grp %d ismigratable %d \n" AB,ci,ci->redNo,thisgroup.idx,m->isMigratableContributor()));
  //m->ci=ci;
  m->redNo=ci->redNo++;
  m->sourceFlag=-1;//A single contribution
  m->gcount=0;

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))

	// if object is an immigrant recovery object, we send the contribution to the source PE
	if(CpvAccess(_currentObj)->mlogData->immigrantRecFlag){
		
		// turning on the message-logging bypass flag
		envelope *env = UsrToEnv(m);
		env->flags = env->flags | CK_BYPASS_DET_MLOG;
    	thisProxy[CpvAccess(_currentObj)->mlogData->immigrantSourcePE].contributeViaMessage(m);
		return;
	}

    Chare *oldObj = CpvAccess(_currentObj);
    CpvAccess(_currentObj) = this;

	// adding contribution
	addContribution(m);

    CpvAccess(_currentObj) = oldObj;
#else
  addContribution(m);
#endif
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkReductionMgr::contributeViaMessage(CkReductionMsg *m){
	//if(CkMyPe() == 2) CkPrintf("[%d] ---> Contributing Via Message\n",CkMyPe());
	
	// turning off bypassing flag
	envelope *env = UsrToEnv(m);
	env->flags = env->flags & ~CK_BYPASS_DET_MLOG;

	// adding contribution
    addContribution(m);
}
#else
void CkReductionMgr::contributeViaMessage(CkReductionMsg *m){}
#endif

void CkReductionMgr::checkIsActive() {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) || CMK_MEM_CHECKPOINT
  return;
#endif

  // Check the number of kids in the inactivelist before or at this redNo
  std::map<int, int>::iterator it;
  int c_inactive = 0;
  for (it = inactiveList.begin(); it != inactiveList.end(); it++) {
    if (it->second <= redNo) {
      DEBR((AA "Kid %d is inactive from redNo %d\n" AB, it->first, it->second));
      c_inactive++;
    }
  }
  DEBR((AA "CheckIsActive redNo %d, kids %d(inactive %d), lcount %d\n" AB, redNo,
    numKids, c_inactive, lcount));

  if(numKids == c_inactive && lcount == 0) {
    if(!is_inactive) {
      informParentInactive();
    }
    is_inactive = true;
  } else if(is_inactive) {
    is_inactive = false;
  }
}

/*
* Add to the child to the inactiveList
*/
void CkReductionMgr::checkAndAddToInactiveList(int id, int red_no) {
  // If there is already a reduction in progress corresponding to red_no, then
  // the time to call ReductionStarting is past so explicitly invoke
  // ReductionStarting on the kid
  if (inProgress && redNo == red_no) {
    thisProxy[id].ReductionStarting(new CkReductionNumberMsg(red_no));
  }

  std::map<int, int>::iterator it;
  it = inactiveList.find(id);
  if (it == inactiveList.end()) {
    inactiveList.insert(std::pair<int, int>(id, red_no));
  } else {
    it->second = red_no;
  }
  // If the red_no is redNo, then check whether this makes this PE inactive
  if (redNo == red_no) {
    checkIsActive();
  }
}

/*
* This is invoked when a real contribution is received from the kid for a
* particular red_no
*/
void CkReductionMgr::checkAndRemoveFromInactiveList(int id, int red_no) {
  std::map<int, int>::iterator it;
  it = inactiveList.find(id);
  if (it == inactiveList.end()) {
    return;
  }
  if (it->second <= red_no) {
    inactiveList.erase(it);
    DEBR((AA "Parent removing kid %d from inactivelist red_no %d\n" AB,
      id, red_no));
  }
}

// Inform parent that I am inactive
void CkReductionMgr::informParentInactive() {
  if (hasParent()) {
    DEBR((AA "Inform parent to add to inactivelist red_no %d\n" AB, redNo));
    thisProxy[treeParent()].AddToInactiveList(
      new CkReductionInactiveMsg(CkMyPe(), redNo));
  }
}

/*
*  Send ReductionStarting message to all the inactive kids which are inactive
*  for the specified red_no
*/
void CkReductionMgr::sendReductionStartingToKids(int red_no) {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) || CMK_MEM_CHECKPOINT
  for (int k=0;k<treeKids();k++)
  {
    DEBR((AA "Asking child PE %d to start #%d\n" AB,firstKid()+k,redNo));
    thisProxy[kids[k]].ReductionStarting(new CkReductionNumberMsg(redNo));
  }
#else
  std::map<int, int>::iterator it;
  for (it = inactiveList.begin(); it != inactiveList.end(); it++) {
    if (it->second <= red_no) {
      DEBR((AA "Parent sending reductionstarting to inactive kid %d\n" AB,
        it->first));
      thisProxy[it->first].ReductionStarting(new CkReductionNumberMsg(red_no));
    }
  }
#endif
}


//////////// Reduction Manager Remote Entry Points /////////////
//Sent down the reduction tree (used by barren PEs)
void CkReductionMgr::ReductionStarting(CkReductionNumberMsg *m)
{
 if(CkMyPe()==0){
	//CkPrintf("!!!!!!!!!!!!!!!!!!!!!!1Who called me ???? %d \n",m->num);
	//delete m;
	//return;
 }
 DEBR((AA " Group ReductionStarting called for redNo %d\n" AB,m->num));
 int srcPE = (UsrToEnv(m))->getSrcPe();
  if (isPresent(m->num) && !inProgress)
  {
    DEBR((AA "Starting reduction #%d at parent's request\n" AB,m->num));
    startReduction(m->num,srcPE);
    finishReduction();
  } else if (isFuture(m->num)){
//   CkPrintf("[%d] arrays Mesg No %d redNo %d \n",CkMyPe(),m->num,redNo);
	  DEBR((AA "Asked to startfuture Reduction %d \n" AB,m->num));
	  if(maxStartRequest < m->num){
		  maxStartRequest = m->num;
	  }
 //   CkAbort("My reduction tree parent somehow got ahead of me! in arrays\n");
	  
    }
  else //is Past
    DEBR((AA "Ignoring parent's late request to start #%d\n" AB,m->num));
  delete m;
}

//Sent to root of reduction tree with reduction contribution
// of migrants that missed the main reduction.
void CkReductionMgr::LateMigrantMsg(CkReductionMsg *m)
{
#if GROUP_LEVEL_REDUCTION
#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&(m->log));
#endif
  addContribution(m);
#else
  m->secondaryCallback = m->callback;
  m->callback = CkCallback(CkIndex_CkReductionMgr::ArrayReductionHandler(NULL),0,thisProxy);
  CkArrayReductionMgr *nodeMgr=nodeProxy[CkMyNode()].ckLocalBranch();
  nodeMgr->LateMigrantMsg(m);
/*	int len = finalMsgs.length();
	finalMsgs.enq(m);
//	CkPrintf("[%d]Late Migrant Detected for %d ,  (%d %d )\n",CkMyPe(),m->redNo,len,finalMsgs.length());
	endArrayReduction();*/
#endif
}

//A late migrating contributor will never contribute to this reduction
void CkReductionMgr::MigrantDied(CkReductionNumberMsg *m)
{
  if (CkMyPe() != 0 || m->num < completedRedNo) CkAbort("Late MigrantDied message recv'd!\n");
  DEBR((AA "Migrant died before contributing to #%d\n" AB,m->num));
 // CkPrintf("[%d,%d]Migrant Died called\n",CkMyNode(),CkMyPe());	 		  
  adj(m->num).gcount--;//He won't be contributing to this one.
  finishReduction();
  delete m;
}

//////////// Reduction Manager State /////////////
void CkReductionMgr::startReduction(int number,int srcPE)
{
  if (isFuture(number)){ /*CkAbort("Can't start reductions out of order!\n");*/ return;}
  if (isPast(number)) {/*CkAbort("Can't restart reduction that's already finished!\n");*/return;}
  if (inProgress){
  	DEBR((AA "This reduction is already in progress\n" AB));
  	return;//This reduction already started
  }
  if (creating) //Don't start yet-- we're creating elements
  {
    DEBR((AA "Postponing start request #%d until we're done creating\n" AB,redNo));
    startRequested=true;
    return;
  }

//If none of these cases, we need to start the reduction--
  DEBR((AA "Starting reduction #%d  %d %d \n" AB,redNo,completedRedNo,number));
  inProgress=true;
 

	/*
		FAULT_EVAC
	*/
  if(!CmiNodeAlive(CkMyPe())){
	return;
  }

  if(disableNotifyChildrenStart) return;
 
  //Sent start requests to our kids (in case they don't already know)
#if GROUP_LEVEL_REDUCTION
  sendReductionStartingToKids(redNo);
  //for (int k=0;k<treeKids();k++)
  //{
  //  DEBR((AA "Asking child PE %d to start #%d\n" AB,firstKid()+k,redNo));
  //  thisProxy[kids[k]].ReductionStarting(new CkReductionNumberMsg(redNo));
  //}
#else
  nodeProxy[CkMyNode()].ckLocalBranch()->startNodeGroupReduction(number,thisgroup);
#endif
	
	/*
  int temp;
  //making it a broadcast done only by PE 0
  if(!hasParent()){
		temp = completedRedNo+1;
		for(int i=temp;i<=number;i++){
			for(int j=0;j<CkNumPes();j++){
				if(j != CkMyPe() && j != srcPE){
					if((CmiNodeAlive(j)||allowMessagesOnly !=-1){
						thisProxy[j].ReductionStarting(new CkReductionNumberMsg(i));
					}
				}
			}
		}	
	}	else{
		temp = number;
	}*/
/*  if(!hasParent()){
		temp = completedRedNo+1;
	}	else{
		temp = number;
	}
	for(int i=temp;i<=number;i++){
	//	DEBR((AA "Asking all child PEs to start #%d \n" AB,i));
		if(hasParent()){
	  // kick-start your parent too ...
			if(treeParent() != srcPE){
				if(CmiNodeAlive(treeParent())||allowMessagesOnly !=-1){
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = oldObj;
#endif
		  		thisProxy[treeParent()].ReductionStarting(new CkReductionNumberMsg(i));
				}	
			}	
		}
	  for (int k=0;k<treeKids();k++)
	  {
			if(firstKid()+k != srcPE){
				if(CmiNodeAlive(kids[k])||allowMessagesOnly !=-1){
			    DEBR((AA "Asking child PE %d to start #%d\n" AB,kids[k],redNo));
			    thisProxy[kids[k]].ReductionStarting(new CkReductionNumberMsg(i));
				}	
			}	
  	}
	}
	*/
}	

/*Handle a message from one element for the reduction*/
void CkReductionMgr::addContribution(CkReductionMsg *m)
{
  if (isPast(m->redNo))
  {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CmiAbort("this version should not have late migrations");
#else
	//We've moved on-- forward late contribution straight to root
    DEBR((AA "Migrant gives late contribution for #%d!\n" AB,m->redNo));
   	// if (!hasParent()) //Root moved on too soon-- should never happen
   	//   CkAbort("Late reduction contribution received at root!\n");
    thisProxy[0].LateMigrantMsg(m);
#endif
  }
  else if (isFuture(m->redNo)) {//An early contribution-- add to future Q
    DEBR((AA "Contributor gives early contribution-- for #%d\n" AB,m->redNo));
    futureMsgs.enq(m);
  } else {// An ordinary contribution
    DEBR((AA "Recv'd local contribution %d for #%d at %d\n" AB,nContrib,m->redNo,this));
   // CkPrintf("[%d] Local Contribution for %d in Mesg %d at %.6f\n",CkMyPe(),redNo,m->redNo,CmiWallTimer());
    startReduction(m->redNo,CkMyPe());
    msgs.enq(m);
    nContrib++;
    finishReduction();
  }
}

/**function checks if it has got all contributions that it is supposed to
get at this processor. If it is done it sends the reduced result to the local
nodegroup */
void CkReductionMgr::finishReduction(void)
{
  /*CkPrintf("[%d]finishReduction called for redNo %d with nContrib %d (!inProgress) | creating) %d at %.6f\n",CkMyPe(),redNo, nContrib,(!inProgress) | creating,CmiWallTimer());*/
  DEBR((AA "in finishReduction (inProgress=%d) in grp %d\n" AB,inProgress,thisgroup.idx));
  if ((!inProgress) || creating){
  	DEBR((AA "Either not in Progress or creating\n" AB));
  	return;
  }

  bool partialReduction = false;

  //CkPrintf("[%d]finishReduction called for redNo %d with nContrib %d at %.6f\n",CkMyPe(),redNo, nContrib,CmiWallTimer());
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	if (nContrib<(lcount+adj(redNo).lcount) - numImmigrantRecObjs + numEmigrantRecObjs){
          if (msgs.length() > 1 && CkReduction::reducerTable[msgs.peek()->reducer].streamable) {
            partialReduction = true;
          }
          else {
            DEBR((AA "Need more local messages %d %d\n" AB,nContrib,(lcount+adj(redNo).lcount)));
            return; //Need more local messages
          }
	}
#else
  if (nContrib<(lcount+adj(redNo).lcount)){
         if (msgs.length() > 1 && CkReduction::reducerTable[msgs.peek()->reducer].streamable) {
           partialReduction = true;
         }
         else {
           DEBR((AA "Need more local messages %d %d\n" AB,nContrib,(lcount+adj(redNo).lcount)));
           return; //Need more local messages
         }
  }
#endif

#if GROUP_LEVEL_REDUCTION
  if (nRemote<treeKids()) {
    if (msgs.length() > 1 && CkReduction::reducerTable[msgs.peek()->reducer].streamable) {
      partialReduction = true;
    }
    else {
      DEBR((AA "Need more remote messages %d %d\n" AB,nRemote,treeKids()));
      return; //Need more remote messages
    }
  }
	
#endif
 
  DEBR((AA "Reducing data... %d %d\n" AB,nContrib,(lcount+adj(redNo).lcount)));
  CkReductionMsg *result=reduceMessages();
  result->redNo=redNo;

  if (partialReduction) {
    msgs.enq(result);
    return;
  }

#if GROUP_LEVEL_REDUCTION
  if (hasParent())
  {//Pass data up tree to parent
    DEBR((AA "Passing reduced data up to parent node %d.\n" AB,treeParent()));
    DEBR((AA "Message gcount is %d+%d+%d.\n" AB,result->gcount,gcount,adj(redNo).gcount));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    result->gcount+=gcount+adj(redNo).gcount;
#else
    result->gcount+=gcount+adj(redNo).gcount;
#endif
    thisProxy[treeParent()].RecvMsg(result);
  }
  else 
  {//We are root-- pass data to client
    DEBR((AA "Final gcount is %d+%d+%d.\n" AB,result->gcount,gcount,adj(redNo).gcount));
    int totalElements=result->gcount+gcount+adj(redNo).gcount;
    if (totalElements>result->nSources()) 
    {
      DEBR((AA "Only got %d of %d contributions (c'mon, migrators!)\n" AB,result->nSources(),totalElements));
      msgs.enq(result);
      return; // Wait for migrants to contribute
    } else if (totalElements<result->nSources()) {
      DEBR((AA "Got %d of %d contributions\n" AB,result->nSources(),totalElements));
#if !defined(_FAULT_CAUSAL_)
      CkAbort("ERROR! Too many contributions at root!\n");
#endif
    }
    DEBR((AA "Passing result to client function\n" AB));
    CkSetRefNum(result, result->getUserFlag());
    if (!result->callback.isInvalid())
	    result->callback.send(result);
    else if (!storedCallback.isInvalid())
	    storedCallback.send(result);
    else
	    CkAbort("No reduction client!\n"
		    "You must register a client with either SetReductionClient or during contribute.\n");
  }

#else
  result->gcount+=gcount+adj(redNo).gcount;

  result->secondaryCallback = result->callback;
  result->callback = CkCallback(CkIndex_CkReductionMgr::ArrayReductionHandler(NULL),0,thisProxy);
	DEBR((AA "Reduced mesg gcount %d localgcount %d\n" AB,result->gcount,gcount));

  //CkPrintf("[%d] Got all local Messages in finishReduction %d in redNo %d\n",CkMyPe(),nContrib,redNo);

 // DEBR(("[%d,%d]Callback for redNo %d in group %d  mesggcount=%d localgcount=%d\n",CkMyNode(),CkMyPe(),redNo,thisgroup.idx,ret->gcount,gcount));
  
  // Find our node reduction manager, and pass reduction to him:
  CkArrayReductionMgr *nodeMgr=nodeProxy[CkMyNode()].ckLocalBranch();
  nodeMgr->contributeArrayReduction(result);
#endif

  //House Keeping Operations will have to check later what needs to be changed
  redNo++;
  // Check after every reduction contribution whether this makes the PE inactive
  // starting this redNo
  checkIsActive();
  //Shift the count adjustment vector down one slot (to match new redNo)
  int i;
#if !GROUP_LEVEL_REDUCTION
    /* nodegroup reduction will adjust adjVec in endArrayReduction on PE 0 */
  if(CkMyPe()!=0)
#endif
  {
	completedRedNo++;
  	for (i=1;i<(int)(adjVec.length());i++){
	   adjVec[i-1]=adjVec[i];
	}
	adjVec.length()--;  
  }

  inProgress=false;
  startRequested=false;
  nRemote=nContrib=0;

  //Look through the future queue for messages we can now handle
  int n=futureMsgs.length();
  for (i=0;i<n;i++)
  {
    CkReductionMsg *m=futureMsgs.deq();
    if (m!=NULL) //One of these addContributions may have finished us.
      addContribution(m);//<- if *still* early, puts it back in the queue
  }
#if GROUP_LEVEL_REDUCTION
  n=futureRemoteMsgs.length();
  for (i=0;i<n;i++)
  {
    CkReductionMsg *m=futureRemoteMsgs.deq();
    if (m!=NULL) {
      RecvMsg(m);//<- if *still* early, puts it back in the queue
    }
  }
#endif

  if(maxStartRequest >= redNo){
	  startReduction(redNo,CkMyPe());
	  finishReduction();
  }
 

}

//Sent up the reduction tree with reduced data
  void CkReductionMgr::RecvMsg(CkReductionMsg *m)
{
#if GROUP_LEVEL_REDUCTION
#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&m->log);
#endif
  if (isPresent(m->redNo)) { //Is a regular, in-order reduction message
    DEBR((AA "Recv'd remote contribution %d for #%d\n" AB,nRemote,m->redNo));
    // If the remote contribution is real, then check whether we can remove the
    // child from the inactiveList if it is in the list
    if (m->nSources() > 0) {
      checkAndRemoveFromInactiveList(m->fromPE, m->redNo);
    }
    startReduction(m->redNo, CkMyPe());
    msgs.enq(m);
    nRemote++;
    finishReduction();
  }
  else if (isFuture(m->redNo)) {
    DEBR((AA "Recv'd early remote contribution %d for #%d\n" AB,nRemote,m->redNo));
    futureRemoteMsgs.enq(m);
  } 
  else CkAbort("Recv'd late remote contribution!\n");
#endif
}

void CkReductionMgr::AddToInactiveList(CkReductionInactiveMsg *m) {
  int id = m->id;
  int last_redno = m->redno;
  delete m;

  DEBR((AA "Parent add kid %d to inactive list from redno %d\n" AB,
    id, last_redno));
  checkAndAddToInactiveList(id, last_redno);

  finishReduction();
  if (last_redno <= redNo) {
    checkIsActive();
  }
}

//////////// Reduction Manager Utilities /////////////

//Return the countAdjustment struct for the given redNo:
countAdjustment &CkReductionMgr::adj(int number)
{
  number-=completedRedNo;
  number--;
  if (number<0) CkAbort("Requested adjustment to prior reduction!\n");
  //Pad the adjustment vector with zeros until it's at least number long
  while ((int)(adjVec.length())<=number)
    adjVec.push_back(countAdjustment());
  return adjVec[number];
}

//Combine (& free) the current message vector msgs.
CkReductionMsg *CkReductionMgr::reduceMessages(void)
{
#if CMK_BIGSIM_CHARM
  _TRACE_BG_END_EXECUTE(1);
  void* _bgParentLog = NULL;
  _TRACE_BG_BEGIN_EXECUTE_NOMSG("GroupReduce", &_bgParentLog, 0);
#endif
  CkReductionMsg *ret=NULL;

  //Look through the vector for a valid reducer, swapping out placeholder messages
  CkReduction::reducerType r=CkReduction::invalid;
  int msgs_gcount=0;//Reduced gcount
  int msgs_nSources=0;//Reduced nSources
  CMK_REFNUM_TYPE msgs_userFlag=(CMK_REFNUM_TYPE)-1;
  CkCallback msgs_callback;
  int i;
  int nMsgs=0;
  CkReductionMsg **msgArr=new CkReductionMsg*[msgs.length()];
  CkReductionMsg *m;
  bool isMigratableContributor;

  // Copy message queue into msgArr, skipping placeholders:
  while (NULL!=(m=msgs.deq()))
  {
    msgs_gcount+=m->gcount;
    if (m->sourceFlag!=0)
    { //This is a real message from an element, not just a placeholder
      msgs_nSources+=m->nSources();
#if CMK_BIGSIM_CHARM
      _TRACE_BG_ADD_BACKWARD_DEP(m->log);
#endif

      // for "nop" reducer type, only need to accept one message
      if (nMsgs == 0 || m->reducer != CkReduction::nop) {
        msgArr[nMsgs++]=m;
        r=m->reducer;
        if (!m->callback.isInvalid()){
#if CMK_ERROR_CHECKING
          if(nMsgs > 1 && !(msgs_callback == m->callback))
            CkAbort("mis-matched client callbacks in reduction messages\n");
#endif
          msgs_callback=m->callback;
        }
        if (m->userFlag!=(CMK_REFNUM_TYPE)-1)
          msgs_userFlag=m->userFlag;
	isMigratableContributor=m->isMigratableContributor();
      }
      else {
#if CMK_ERROR_CHECKING
        if(!(msgs_callback == m->callback))
          CkAbort("mis-matched client callbacks in reduction messages\n");
#endif  
        delete m;
      }
    }
    else
    { //This is just a placeholder message-- forget it
      delete m;
    }
  }

  if (nMsgs==0||r==CkReduction::invalid)
  //No valid reducer in the whole vector
    ret=CkReductionMsg::buildNew(0,NULL);
  else
  {//Use the reducer to reduce the messages
		//if there is only one msg to be reduced just return that message
    if(nMsgs == 1 &&
       msgArr[0]->reducer != CkReduction::set &&
       msgArr[0]->reducer != CkReduction::tuple) {
      ret = msgArr[0];
    }else{
      if (msgArr[0]->reducer == CkReduction::nop) {
        // nMsgs > 1 indicates that reduction type is not nop
        // this means any data with reducer type nop was submitted
        // only so that counts would agree, and can be removed
        delete msgArr[0];
        msgArr[0] = msgArr[nMsgs - 1];
        nMsgs--;
      }
      CkReduction::reducerFn f=CkReduction::reducerTable[r].fn;
      ret=(*f)(nMsgs,msgArr);
    }
    ret->reducer=r;
  }



#if USE_CRITICAL_PATH_HEADER_ARRAY

#if CRITICAL_PATH_DEBUG > 3
  CkPrintf("combining critical path information from messages in CkReductionMgr::reduceMessages\n");
#endif

  MergeablePathHistory path(CkpvAccess(currentlyExecutingPath));
  path.updateMax(UsrToEnv(ret));
  // Combine the critical paths from all the reduction messages
  for (i=0;i<nMsgs;i++){
    if (msgArr[i]!=ret){
      //      CkPrintf("[%d] other path = %lf\n", CkMyPe(), UsrToEnv(msgArr[i])->pathHistory.getTime() );
      path.updateMax(UsrToEnv(msgArr[i]));
    }
  }
  

#if CRITICAL_PATH_DEBUG > 3
  CkPrintf("[%d] result path = %lf\n", CkMyPe(), path.getTime() );
#endif
  
  PathHistoryTableEntry tableEntry(path);
  tableEntry.addToTableAndEnvelope(UsrToEnv(ret));
  
#endif

	//Go back through the vector, deleting old messages
  for (i=0;i<nMsgs;i++) if (msgArr[i]!=ret) delete msgArr[i];
  delete [] msgArr;

  //Set the message counts
  ret->redNo=redNo;
  ret->gcount=msgs_gcount;
  ret->userFlag=msgs_userFlag;
  ret->callback=msgs_callback;
  ret->sourceFlag=msgs_nSources;
	ret->setMigratableContributor(isMigratableContributor);
  ret->fromPE = CkMyPe();
  DEBR((AA "Reduced gcount=%d; sourceFlag=%d\n" AB,ret->gcount,ret->sourceFlag));

  return ret;
}


//Checkpointing utilities
//pack-unpack method for CkReductionMsg
//if packing pack the message and then unpack and return it
//if unpacking allocate memory for it read it off disk and then unapck
//and return it
void CkReductionMgr::pup(PUP::er &p)
{
//We do not store the client function pointer or the client function parameter,
//it is the responsibility of the programmer to correctly restore these
  CkGroupInitCallback::pup(p);
  p(redNo);
  p(completedRedNo);
  p(inProgress); p(creating); p(startRequested);
  p(nContrib); p(nRemote); p(disableNotifyChildrenStart);
  p|msgs;
  p|futureMsgs;
  p|futureRemoteMsgs;
  p|finalMsgs;
  p|adjVec;
#if !GROUP_LEVEL_REDUCTION
  p|nodeProxy;
#endif
  p|storedCallback;
    // handle CkReductionClientBundle
  if (storedCallback.type == CkCallback::callCFn && storedCallback.d.cfn.fn == CkReductionClientBundle::callbackCfn) 
  {
    CkReductionClientBundle *bd;
    if (p.isUnpacking()) 
      bd = new CkReductionClientBundle;
    else
      bd = (CkReductionClientBundle *)storedCallback.d.cfn.param;
    p|*bd;
    if (p.isUnpacking()) storedCallback.d.cfn.param = bd;
  }

  // subtle --- Gengbin
  // Group : CkReductionMgr
  // CkArray: CkReductionMgr
  // lcount/gcount in Group is set in Group constructor
  // lcount/gcount in CkArray is not, it is set when array elements are created
  // we can not pup because inserting array elems will add the counters again
//  p|lcount;
//  p|gcount;
//  p|lcount;
//  //  p|gcount;
//  //  printf("[%d] nodeProxy nodeGroup %d pupped in group %d \n",CkMyPe(),(nodeProxy.ckGetGroupID()).idx,thisgroup.idx);
  if(p.isUnpacking()){
    thisProxy = thisgroup;
    maxStartRequest=0;
#ifdef BINOMIAL_TREE
    init_BinomialTree();
#else
    init_BinaryTree();
#endif
    is_inactive = false;
    checkIsActive();
  }

  DEBR(("[%d,%d] pupping _____________  gcount = %d \n",CkMyNode(),CkMyPe(),gcount));
}


//Callback for doing Reduction through NodeGroups added by Sayantan

void CkReductionMgr::ArrayReductionHandler(CkReductionMsg *m){
	finalMsgs.enq(m);
	//CkPrintf("ArrayReduction Handler Invoked for %d \n",m->redNo);
	adj(m->redNo).mainRecvd = 1;
	DEBR(("~~~~~~~~~~~~~ ArrayReductionHandler Callback called for redNo %d with mesgredNo %d at %.6f %d\n",completedRedNo,m->redNo,CmiWallTimer()));
	endArrayReduction();
}

void CkReductionMgr :: endArrayReduction(){
	CkReductionMsg *ret=NULL;
  	int nMsgs=finalMsgs.length();
	//CkPrintf("endArrayReduction Invoked for %d \n",completedRedNo+1);
  	//Look through the vector for a valid reducer, swapping out placeholder messages
	//CkPrintf("Length of Final Message %d \n",nMsgs);
  	CkReduction::reducerType r=CkReduction::invalid;
  	int msgs_gcount=0;//Reduced gcount
  	int msgs_nSources=0;//Reduced nSources
  	CMK_REFNUM_TYPE msgs_userFlag=(CMK_REFNUM_TYPE)-1;
  	CkCallback msgs_callback;
	CkCallback msgs_secondaryCallback;
	CkVec<CkReductionMsg *> tempMsgs;
  	int i;
	int numMsgs = 0;
  	for (i=0;i<nMsgs;i++)
  	{
          CkReductionMsg *m=finalMsgs.deq();
          if(m->redNo == completedRedNo +1){
            msgs_gcount+=m->gcount;
            if (m->sourceFlag!=0)
            { //This is a real message from an element, not just a placeholder
              msgs_nSources+=m->nSources();

              // for "nop" reducer type, only need to accept one message
              if (tempMsgs.length() == 0 || m->reducer != CkReduction::nop) {
                r=m->reducer;
                if (!m->callback.isInvalid())
                  msgs_callback=m->callback;
                if(!m->secondaryCallback.isInvalid())
                  msgs_secondaryCallback = m->secondaryCallback;
                if (m->userFlag!=(CMK_REFNUM_TYPE)-1)
                  msgs_userFlag=m->userFlag;
                tempMsgs.push_back(m);
              }
              else {
                delete m;
              }
            }
            else {
              delete m;
            }
          }else{
            finalMsgs.enq(m);
          }
	}
	numMsgs = tempMsgs.length();

	DEBR(("[%d]Total = %d %d Sources = %d Number of Messages %d Adj(Completed redno).mainRecvd %d\n",CkMyPe(),msgs_gcount,  adj(completedRedNo+1).gcount,msgs_nSources,numMsgs,adj(completedRedNo+1).mainRecvd));

	if(numMsgs == 0){
		return;
	}
	if(adj(completedRedNo+1).mainRecvd == 0){
		for(i=0;i<numMsgs;i++){
			finalMsgs.enq(tempMsgs[i]);
		}
		return;
	}

/*
	NOT NEEDED ANYMORE DONE at nodegroup level
	if(msgs_gcount  > msgs_nSources){
		for(i=0;i<numMsgs;i++){
			finalMsgs.enq(tempMsgs[i]);
		}
		return;
	}*/

	if (numMsgs==0||r==CkReduction::invalid)
  		//No valid reducer in the whole vector
    		ret=CkReductionMsg::buildNew(0,NULL);
  	else{//Use the reducer to reduce the messages
               CkReduction::reducerFn f=CkReduction::reducerTable[r].fn;
		// has to be corrected elements from above need to be put into a temporary vector
    		CkReductionMsg **msgArr=&tempMsgs[0];//<-- HACK!

                if (numMsgs > 1 && msgArr[0]->reducer == CkReduction::nop) {
                  // nMsgs > 1 indicates that reduction type is not "nop"
                  // this means any data with reducer type nop was submitted
                  // only so that counts would agree, and can be removed
                  delete msgArr[0];
                  msgArr[0] = msgArr[numMsgs - 1];
                  numMsgs--;
                }

    		ret=(*f)(numMsgs,msgArr);
    		ret->reducer=r;

  	}

	
#if USE_CRITICAL_PATH_HEADER_ARRAY

#if CRITICAL_PATH_DEBUG > 3
	CkPrintf("[%d] combining critical path information from messages in CkReductionMgr::endArrayReduction(). numMsgs=%d\n", CkMyPe(), numMsgs);
#endif

	MergeablePathHistory path(CkpvAccess(currentlyExecutingPath));
	path.updateMax(UsrToEnv(ret));
	// Combine the critical paths from all the reduction messages into the header for the new result
	for (i=0;i<numMsgs;i++){
	  if (tempMsgs[i]!=ret){
	    //	    CkPrintf("[%d] other path = %lf\n", CkMyPe(), UsrToEnv(tempMsgs[i])->pathHistory.getTime() );
	    path.updateMax(UsrToEnv(tempMsgs[i]));
	  } else {
	    //  CkPrintf("[%d] other path is ret = %lf\n", CkMyPe(), UsrToEnv(tempMsgs[i])->pathHistory.getTime() );
	  }
	}
	// Also consider the path which got us into this entry method

#if CRITICAL_PATH_DEBUG > 3
	CkPrintf("[%d] result path = %lf\n", CkMyPe(), path.getTime() );
#endif

#endif
  




	for(i = 0;i<numMsgs;i++){
		if (tempMsgs[i] != ret) delete tempMsgs[i];
	}

	//CkPrintf("Length of finalMsgs after endReduction %d \n",finalMsgs.length());
	//CkPrintf("Data size of result = %d Length of finalMsg %d \n",ret->getLength(),finalMsgs.length());

	ret->redNo=completedRedNo+1;
  	ret->gcount=msgs_gcount;
  	ret->userFlag=msgs_userFlag;
  	ret->callback=msgs_callback;
	ret->secondaryCallback = msgs_secondaryCallback;
  	ret->sourceFlag=msgs_nSources;

	DEBR(("~~~~~~~~~~~~~~~~~ About to call callback from end of GROUP REDUCTION %d at %.6f\n",completedRedNo,CmiWallTimer()));

	CkSetRefNum(ret, ret->getUserFlag());
	if (!ret->secondaryCallback.isInvalid())
	    ret->secondaryCallback.send(ret);
    else if (!storedCallback.isInvalid())
	    storedCallback.send(ret);
    else{
      DEBR(("No reduction client for group %d \n",thisgroup.idx));
	    CkAbort("No reduction client!\n"
		    "You must register a client with either SetReductionClient or during contribute.\n");
    }
	completedRedNo++;

	DEBR(("[%d,%d]------------END OF GROUP REDUCTION %d for group %d at %.6f\n",CkMyNode(),CkMyPe(),completedRedNo,thisgroup.idx,CkWallTimer()));

	for (i=1;i<(int)(adjVec.length());i++)
    		adjVec[i-1]=adjVec[i];
	adjVec.length()--;
	endArrayReduction();
}

void CkReductionMgr::init_BinaryTree(){
	parent = (CkMyPe()-1)/TREE_WID;
	int firstkid = CkMyPe()*TREE_WID+1;
	numKids=CkNumPes()-firstkid;
        if (numKids>TREE_WID) numKids=TREE_WID;
        if (numKids<0) numKids=0;

	for(int i=0;i<numKids;i++){
		kids.push_back(firstkid+i);
		newKids.push_back(firstkid+i);
	}
}

void CkReductionMgr::init_BinomialTree(){
	int depth = (int )ceil((log((double )CkNumPes())/log((double)2)));
	/*upperSize = (unsigned )pow((double)2,depth);*/
	upperSize = (unsigned) 1 << depth;
	label = upperSize-CkMyPe()-1;
	int p=label;
	int count=0;
	while( p > 0){
		if(p % 2 == 0)
			break;
		else{
			p = p/2;
			count++;
		}
	}
	/*parent = label + rint(pow((double)2,count));*/
	parent = label + (1<<count);
	parent = upperSize -1 -parent;
	int temp;
	if(count != 0){
		numKids = 0;
		for(int i=0;i<count;i++){
			/*temp = label - rint(pow((double)2,i));*/
			temp = label - (1<<i);
			temp = upperSize-1-temp;
			if(temp <= CkNumPes()-1){
				kids.push_back(temp);
				numKids++;
			}
		}
	}else{
		numKids = 0;
	//	kids = NULL;
	}
}


int CkReductionMgr::treeRoot(void)
{
  return 0;
}
bool CkReductionMgr::hasParent(void) //Root Node
{
  return (bool)(CkMyPe()!=treeRoot());
}
int CkReductionMgr::treeParent(void) //My parent Node
{
  return parent;
}

int CkReductionMgr::firstKid(void) //My first child Node
{
  return CkMyPe()*TREE_WID+1;
}
int CkReductionMgr::treeKids(void)//Number of children in tree
{
  return numKids;
}


//                simple "stateless" barrier
//                no state checkpointed, for FT purpose
//                require no overlapping barriers
void CkReductionMgr::barrier(CkReductionMsg *m)
{
  barrier_nContrib++;
  barrier_nSource++;
  if(!m->callback.isInvalid())
      barrier_storedCallback=m->callback;
  finishBarrier();
  delete m;
}

void CkReductionMgr::finishBarrier(void)
{
       if(barrier_nContrib<lcount){//need more local message
               DEBR(("[%d] current contrib:%d,lcount:%d\n",CkMyPe(),barrier_nContrib,lcount));
               return;
       }
       if(barrier_nRemote<treeKids()){//need more remote messages
               DEBR(("[%d] current remote:%d,kids:%d\n",CkMyPe(),barrier_nRemote,treeKids()));
               return;
       }
       CkReductionMsg * result = CkReductionMsg::buildNew(0,NULL);
       result->callback=barrier_storedCallback;
       result->sourceFlag=barrier_nSource;
       result->gcount=barrier_gCount;
       if(hasParent())
       {
               DEBR(("[%d]send to parent:%d\n",CkMyPe(),treeParent()));
               result->gcount+=gcount;
               thisProxy[treeParent()].Barrier_RecvMsg(result);
       }
       else{
               int totalElements=result->gcount+gcount;
               DEBR(("[%d]root,totalElements:%d,source:%d\n",CkMyPe(),totalElements,result->nSources()));
               if(totalElements<result->nSources()){
                       CkAbort("ERROR! Too many contributions at barrier root\n");
               }
               CkSetRefNum(result,result->getUserFlag());
               if(!result->callback.isInvalid())
                       result->callback.send(result);
               else if(!barrier_storedCallback.isInvalid())
                               barrier_storedCallback.send(result);
               else 
                       CkAbort("No reduction client!\n");
       }
       barrier_nRemote=barrier_nContrib=0;
       barrier_gCount=0;
       barrier_nSource=0;
}

void CkReductionMgr::Barrier_RecvMsg(CkReductionMsg *m)
{
       barrier_nRemote++;
       barrier_gCount+=m->gcount;
       barrier_nSource+=m->nSources();
       if(!m->callback.isInvalid())
               barrier_storedCallback=m->callback;
       finishBarrier();
}



/////////////////////////////////////////////////////////////////////////

////////////////////////////////
//CkReductionMsg support

//ReductionMessage default private constructor-- does nothing
CkReductionMsg::CkReductionMsg(){}
CkReductionMsg::~CkReductionMsg(){}

//This define gives the distance from the start of the CkReductionMsg
// object to the start of the user data area (just below last object field)
#define ARM_DATASTART (sizeof(CkReductionMsg)-sizeof(double))

//"Constructor"-- builds and returns a new CkReductionMsg.
//  the "data" array you specify will be copied into this object.
CkReductionMsg *CkReductionMsg::buildNew(int NdataSize,const void *srcData,
    CkReduction::reducerType reducer, CkReductionMsg *buf)
{
  int len[1];
  len[0]=NdataSize;
  CkReductionMsg *ret = buf ? buf : new(len,0) CkReductionMsg();

  ret->dataSize=NdataSize;
  if (srcData!=NULL && !buf)
    memcpy(ret->data,srcData,NdataSize);
  ret->userFlag=(CMK_REFNUM_TYPE)-1;
  ret->reducer=reducer;
  //ret->ci=NULL;
  ret->sourceFlag=-1000;
  ret->gcount=0;
  ret->migratableContributor = true;
#if CMK_BIGSIM_CHARM
  ret->log = NULL;
#endif
  return ret;
}

// Charm kernel message runtime support:
void *
CkReductionMsg::alloc(int msgnum,size_t size,int *sz,int priobits)
{
  int totalsize=ARM_DATASTART+(*sz);
  DEBR(("CkReductionMsg::Allocating %d store; %d bytes total\n",*sz,totalsize));
  CkReductionMsg *ret = (CkReductionMsg *)
    CkAllocMsg(msgnum,totalsize,priobits);
  ret->data=(void *)(&ret->dataStorage);
  return (void *) ret;
}

void *
CkReductionMsg::pack(CkReductionMsg* in)
{
  DEBR(("CkReductionMsg::pack %d %d %d %d\n",in->sourceFlag,in->redNo,in->gcount,in->dataSize));
  //CkPrintf("CkReductionMsg::pack %d %d %d %d\n",in->sourceFlag,in->redNo,in->gcount,in->dataSize);
  in->data = NULL;
  return (void*) in;
}

CkReductionMsg* CkReductionMsg::unpack(void *in)
{
  CkReductionMsg *ret = (CkReductionMsg *)in;
  DEBR(("CkReductionMsg::unpack %d %d %d %d\n",ret->sourceFlag,ret->redNo,ret->gcount,ret->dataSize));
  //CkPrintf("CkReductionMsg::unpack %d %d %d %d\n",ret->sourceFlag,ret->redNo,ret->gcount,ret->dataSize);
  ret->data=(void *)(&ret->dataStorage);
  return ret;
}


/////////////////////////////////////////////////////////////////////////////////////
///////////////// Builtin Reducer Functions //////////////
/* A simple reducer, like sum_int, looks like this:
CkReductionMsg *sum_int(int nMsg,CkReductionMessage **msg)
{
  int i,ret=0;
  for (i=0;i<nMsg;i++)
    ret+=*(int *)(msg[i]->getData());
  return CkReductionMsg::buildNew(sizeof(int),(void *)&ret);
}

To keep the code small and easy to change, the implementations below
are built with preprocessor macros.
*/

//////////////// simple reducers ///////////////////
/*A define used to quickly and tersely construct simple reductions.
The basic idea is to use the first message's data array as
(pre-initialized!) scratch space for folding in the other messages.
 */

static CkReductionMsg *invalid_reducer(int nMsg,CkReductionMsg **msg)
{
	CkAbort("Called the invalid reducer type 0.  This probably\n"
		"means you forgot to initialize your custom reducer index.\n");
	return NULL;
}

static CkReductionMsg *nop(int nMsg,CkReductionMsg **msg)
{
  return CkReductionMsg::buildNew(0,NULL, CkReduction::invalid, msg[0]);
}

#define SIMPLE_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg,CkReductionMsg **msg)\
{\
  RED_DEB(("/ PE_%d: " #name " invoked on %d messages\n",CkMyPe(),nMsg));\
  int m,i;\
  int nElem=msg[0]->getLength()/sizeof(dataType);\
  dataType *ret=(dataType *)(msg[0]->getData());\
  for (m=1;m<nMsg;m++)\
  {\
    dataType *value=(dataType *)(msg[m]->getData());\
    for (i=0;i<nElem;i++)\
    {\
      RED_DEB(("|\tmsg%d (from %d) [%d]=" typeStr "\n",m,msg[m]->sourceFlag,i,value[i]));\
      loop\
    }\
  }\
  RED_DEB(("\\ PE_%d: " #name " finished\n",CkMyPe()));\
  return CkReductionMsg::buildNew(nElem*sizeof(dataType),(void *)ret, CkReduction::invalid, msg[0]);\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_REDUCTION(nameBase,loop) \
  SIMPLE_REDUCTION(nameBase##_char,char,"%c",loop) \
  SIMPLE_REDUCTION(nameBase##_short,short,"%h",loop) \
  SIMPLE_REDUCTION(nameBase##_int,int,"%d",loop) \
  SIMPLE_REDUCTION(nameBase##_long,long,"%ld",loop) \
  SIMPLE_REDUCTION(nameBase##_long_long,long long,"%lld",loop) \
  SIMPLE_REDUCTION(nameBase##_uchar,unsigned char,"%c",loop) \
  SIMPLE_REDUCTION(nameBase##_ushort,unsigned short,"%hu",loop) \
  SIMPLE_REDUCTION(nameBase##_uint,unsigned int,"%u",loop) \
  SIMPLE_REDUCTION(nameBase##_ulong,unsigned long,"%lu",loop) \
  SIMPLE_REDUCTION(nameBase##_ulong_long,unsigned long long,"%llu",loop) \
  SIMPLE_REDUCTION(nameBase##_float,float,"%f",loop) \
  SIMPLE_REDUCTION(nameBase##_double,double,"%f",loop)

//Compute the sum the numbers passed by each element.
SIMPLE_POLYMORPH_REDUCTION(sum,ret[i]+=value[i];)

//Compute the product of the numbers passed by each element.
SIMPLE_POLYMORPH_REDUCTION(product,ret[i]*=value[i];)

//Compute the largest number passed by any element.
SIMPLE_POLYMORPH_REDUCTION(max,if (ret[i]<value[i]) ret[i]=value[i];)

//Compute the smallest integer passed by any element.
SIMPLE_POLYMORPH_REDUCTION(min,if (ret[i]>value[i]) ret[i]=value[i];)


//Compute the logical AND of the integers passed by each element.
// The resulting integer will be zero if any source integer is zero; else 1.
SIMPLE_REDUCTION(logical_and,int,"%d",
        if (value[i]==0)
     ret[i]=0;
  ret[i]=!!ret[i];//Make sure ret[i] is 0 or 1
)

//Compute the logical AND of the integers passed by each element.
// The resulting integer will be zero if any source integer is zero; else 1.
SIMPLE_REDUCTION(logical_and_int,int,"%d",
        if (value[i]==0)
     ret[i]=0;
  ret[i]=!!ret[i];//Make sure ret[i] is 0 or 1
)

//Compute the logical AND of the bools passed by each element.
// The resulting bool will be false if any source bool is false; else true.
SIMPLE_REDUCTION(logical_and_bool,bool,"%d",
  if (!value[i]) ret[i]=false;
)

//Compute the logical OR of the integers passed by each element.
// The resulting integer will be 1 if any source integer is nonzero; else 0.
SIMPLE_REDUCTION(logical_or,int,"%d",
  if (value[i]!=0)
           ret[i]=1;
  ret[i]=!!ret[i];//Make sure ret[i] is 0 or 1
)

//Compute the logical OR of the integers passed by each element.
// The resulting integer will be 1 if any source integer is nonzero; else 0.
SIMPLE_REDUCTION(logical_or_int,int,"%d",
  if (value[i]!=0)
           ret[i]=1;
  ret[i]=!!ret[i];//Make sure ret[i] is 0 or 1
)

//Compute the logical OR of the bools passed by each element.
// The resulting bool will be true if any source bool is true; else false.
SIMPLE_REDUCTION(logical_or_bool,bool,"%d",
  if (value[i]) ret[i]=true;
)

//Compute the logical XOR of the integers passed by each element.
// The resulting integer will be 1 if an odd number of source integers is nonzero; else 0.
SIMPLE_REDUCTION(logical_xor_int,int,"%d",
  ret[i] = (!ret[i] != !value[i]);
)

//Compute the logical XOR of the bools passed by each element.
// The resulting bool will be true if an odd number of source bools is true; else false.
SIMPLE_REDUCTION(logical_xor_bool,bool,"%d",
  ret[i] = (ret[i] != value[i]);
)

SIMPLE_REDUCTION(bitvec_and,int,"%d",ret[i]&=value[i];)
SIMPLE_REDUCTION(bitvec_and_int,int,"%d",ret[i]&=value[i];)
SIMPLE_REDUCTION(bitvec_and_bool,bool,"%d",ret[i]&=value[i];)

SIMPLE_REDUCTION(bitvec_or,int,"%d",ret[i]|=value[i];)
SIMPLE_REDUCTION(bitvec_or_int,int,"%d",ret[i]|=value[i];)
SIMPLE_REDUCTION(bitvec_or_bool,bool,"%d",ret[i]|=value[i];)

SIMPLE_REDUCTION(bitvec_xor,int,"%d",ret[i]^=value[i];)
SIMPLE_REDUCTION(bitvec_xor_int,int,"%d",ret[i]^=value[i];)
SIMPLE_REDUCTION(bitvec_xor_bool,bool,"%d",ret[i]^=value[i];)

//Select one random message to pass on
static CkReductionMsg *random(int nMsg,CkReductionMsg **msg) {
  int idx = (int)(CrnDrand()*(nMsg-1) + 0.5);
  return CkReductionMsg::buildNew(msg[idx]->getLength(),
                                  (void *)msg[idx]->getData(),
                                  CkReduction::random, msg[idx]);
}

/////////////// concat ////////////////
/*
This reducer simply appends the data it recieves from each element,
without any housekeeping data to separate them.
*/
static CkReductionMsg *concat(int nMsg,CkReductionMsg **msg)
{
  RED_DEB(("/ PE_%d: reduction_concat invoked on %d messages\n",CkMyPe(),nMsg));
  //Figure out how big a message we'll need
  int i,retSize=0;
  for (i=0;i<nMsg;i++)
      retSize+=msg[i]->getSize();

  RED_DEB(("|- concat'd reduction message will be %d bytes\n",retSize));

  //Allocate a new message
  CkReductionMsg *ret=CkReductionMsg::buildNew(retSize,NULL);

  //Copy the source message data into the return message
  char *cur=(char *)(ret->getData());
  for (i=0;i<nMsg;i++) {
    int messageBytes=msg[i]->getSize();
    memcpy((void *)cur,(void *)msg[i]->getData(),messageBytes);
    cur+=messageBytes;
  }
  RED_DEB(("\\ PE_%d: reduction_concat finished-- %d messages combined\n",CkMyPe(),nMsg));
  return ret;
}

/////////////// set ////////////////
/*
This reducer appends the data it recieves from each element
along with some housekeeping data indicating contribution boundaries.
The message data is thus a list of reduction_set_element structures
terminated by a dummy reduction_set_element with a sourceElement of -1.
*/

//This rounds an integer up to the nearest multiple of sizeof(double)
static const int alignSize=sizeof(double);
static int SET_ALIGN(int x) {return ~(alignSize-1)&((x)+alignSize-1);}

//This gives the size (in bytes) of a reduction_set_element
static int SET_SIZE(int dataSize)
{return SET_ALIGN(sizeof(int)+dataSize);}

//This returns a pointer to the next reduction_set_element in the list
static CkReduction::setElement *SET_NEXT(CkReduction::setElement *cur)
{
  char *next=((char *)cur)+SET_SIZE(cur->dataSize);
  return (CkReduction::setElement *)next;
}

//Combine the data passed by each element into an list of reduction_set_elements.
// Each element may contribute arbitrary data (with arbitrary length).
static CkReductionMsg *set(int nMsg,CkReductionMsg **msg)
{
  RED_DEB(("/ PE_%d: reduction_set invoked on %d messages\n",CkMyPe(),nMsg));
  //Figure out how big a message we'll need
  int i,retSize=0;
  for (i=0;i<nMsg;i++) {
    if (!msg[i]->isFromUser())
    //This message is composite-- it will just be copied over (less terminating -1)
      retSize+=(msg[i]->getSize()-sizeof(int));
    else //This is a message from an element-- it will be wrapped in a reduction_set_element
      retSize+=SET_SIZE(msg[i]->getSize());
  }
  retSize+=sizeof(int);//Leave room for terminating -1.

  RED_DEB(("|- composite set reduction message will be %d bytes\n",retSize));

  //Allocate a new message
  CkReductionMsg *ret=CkReductionMsg::buildNew(retSize,NULL);

  //Copy the source message data into the return message
  CkReduction::setElement *cur=(CkReduction::setElement *)(ret->getData());
  for (i=0;i<nMsg;i++)
    if (!msg[i]->isFromUser())
    {//This message is composite-- just copy it over (less terminating -1)
                        int messageBytes=msg[i]->getSize()-sizeof(int);
                        RED_DEB(("|\tc msg[%d] is %d bytes\n",i,msg[i]->getSize()));
                        memcpy((void *)cur,(void *)msg[i]->getData(),messageBytes);
                        cur=(CkReduction::setElement *)(((char *)cur)+messageBytes);
    }
    else //This is a message from an element-- wrap it in a reduction_set_element
    {
      RED_DEB(("|\tu msg[%d] is %d bytes\n",i,msg[i]->getSize()));
      cur->dataSize=msg[i]->getSize();
      memcpy((void *)cur->data,(void *)msg[i]->getData(),msg[i]->getSize());
      cur=SET_NEXT(cur);
    }
  cur->dataSize=-1;//Add a terminating -1.
  RED_DEB(("\\ PE_%d: reduction_set finished-- %d messages combined\n",CkMyPe(),nMsg));
  return ret;
}

//Utility routine: get the next reduction_set_element in the list
// if there is one, or return NULL if there are none.
//To get all the elements, just keep feeding this procedure's output back to
// its input until it returns NULL.
CkReduction::setElement *CkReduction::setElement::next(void)
{
  CkReduction::setElement *n=SET_NEXT(this);
  if (n->dataSize==-1)
    return NULL;//This is the end of the list
  else
    return n;//This is just another element
}


///////// statisticsElement

CkReduction::statisticsElement::statisticsElement(double initialValue)
  : count(1)
  , mean(initialValue)
  , m2(0.0)
{}

// statistics reducer
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
// Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979),
// "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances." (PDF),
// Technical Report STAN-CS-79-773, Department of Computer Science, Stanford University.
static CkReductionMsg* statistics(int nMsgs, CkReductionMsg** msg)
{
  int nElem = msg[0]->getLength() / sizeof(CkReduction::statisticsElement);
  CkReduction::statisticsElement* ret = (CkReduction::statisticsElement*)(msg[0]->getData());
  for (int m = 1; m < nMsgs; m++)
  {
    CkReduction::statisticsElement* value = (CkReduction::statisticsElement*)(msg[m]->getData());
    for (int i = 0; i < nElem; i++)
    {
      double a_count = ret[i].count;
      ret[i].count += value[i].count;
      double delta = value[i].mean - ret[i].mean;
      ret[i].mean += delta * value[i].count / ret[i].count;
      ret[i].m2 += value[i].m2 + delta * delta * value[i].count * a_count / ret[i].count;
    }
  }
  return CkReductionMsg::buildNew(
    nElem*sizeof(CkReduction::statisticsElement),
    (void *)ret,
    CkReduction::invalid,
    msg[0]);
}

///////// tupleElement

CkReduction::tupleElement::tupleElement()
  : dataSize(0)
  , data(NULL)
  , reducer(CkReduction::invalid)
  , owns_data(false)
{}
CkReduction::tupleElement::tupleElement(size_t dataSize_, void* data_, CkReduction::reducerType reducer_)
  : dataSize(dataSize_)
  , data((char*)data_)
  , reducer(reducer_)
  , owns_data(false)
{
}
CkReduction::tupleElement::tupleElement(CkReduction::tupleElement&& rhs_move)
  : dataSize(rhs_move.dataSize)
  , data(rhs_move.data)
  , reducer(rhs_move.reducer)
  , owns_data(rhs_move.owns_data)
{
  rhs_move.dataSize = 0;
  rhs_move.data = 0;
  rhs_move.reducer = CkReduction::invalid;
  rhs_move.owns_data = false;
}
CkReduction::tupleElement& CkReduction::tupleElement::operator=(CkReduction::tupleElement&& rhs_move)
{
  if (owns_data)
    delete[] data;
  dataSize = rhs_move.dataSize;
  data = rhs_move.data;
  reducer = rhs_move.reducer;
  owns_data = rhs_move.owns_data;
  rhs_move.dataSize = 0;
  rhs_move.data = 0;
  rhs_move.reducer = CkReduction::invalid;
  rhs_move.owns_data = false;
  return *this;
}
CkReduction::tupleElement::~tupleElement()
{
  if (owns_data)
    delete[] data;
}

void CkReduction::tupleElement::pup(PUP::er &p) {
  p|dataSize;
  // TODO - it might be better to pack these raw, then we don't have to
  //  transform & copy them out on unpacking, we could just use the message's
  //  memory directly
  if (p.isUnpacking()) {
    data = new char[dataSize];
    owns_data = true;
  }
  PUParray(p, data, dataSize);
  if (p.isUnpacking()){
    int temp;
    p|temp;
    reducer=(CkReduction::reducerType)temp;
  } else {
    int temp=(int)reducer;
    p|temp;
  }
}

CkReductionMsg* CkReductionMsg::buildFromTuple(CkReduction::tupleElement* reductions, int num_reductions)
{
  PUP::sizer ps;
  ps|num_reductions;
  PUParray(ps, reductions, num_reductions);

  CkReductionMsg* msg = CkReductionMsg::buildNew(ps.size(), NULL, CkReduction::tuple);
  PUP::toMem p(msg->data);
  p|num_reductions;
  PUParray(p, reductions, num_reductions);
  if (p.size() != ps.size()) CmiAbort("Size mismatch packing CkReduction::tupleElement::tupleToBuffer\n");
  return msg;
}

void CkReductionMsg::toTuple(CkReduction::tupleElement** out_reductions, int* num_reductions)
{
  PUP::fromMem p(this->getData());
  p|(*num_reductions);
  *out_reductions = new CkReduction::tupleElement[*num_reductions];
  PUParray(p, *out_reductions, *num_reductions);
}

// tuple reducer
CkReductionMsg* CkReduction::tupleReduction(int num_messages, CkReductionMsg** messages)
{
  CkReduction::tupleElement** tuple_data = new CkReduction::tupleElement*[num_messages];
  int num_reductions = 0;
  for (int message_idx = 0; message_idx < num_messages; ++message_idx)
  {
    int itr_num_reductions = 0;
    messages[message_idx]->toTuple(&tuple_data[message_idx], &itr_num_reductions);

    // each message must submit the same reductions
    if (num_reductions == 0)
      num_reductions = itr_num_reductions;
    else if (num_reductions != itr_num_reductions)
      CmiAbort("num_reductions mismatch in CkReduction::tupleReduction");
  }

  DEB_TUPLE(("tupleReduction {\n  num_messages=%d,\n  num_reductions=%d,\n  length=%d\n",
           num_messages, num_reductions, messages[0]->getLength()));

  CkReduction::tupleElement* return_data = new CkReduction::tupleElement[num_reductions];
  // using a raw buffer to avoid CkReductionMsg constructor/destructor, we want to manage
  //  the inner memory of these temps ourselves to avoid unneeded copies
  char* simulated_messages_buffer = new char[sizeof(CkReductionMsg) * num_reductions * num_messages];
  CkReductionMsg** simulated_messages = new CkReductionMsg*[num_messages];

  // imagine the given data in a 2D layout where the messages are rows and reductions are columns
  // here we grab each column and run that reduction

  for (int reduction_idx = 0; reduction_idx < num_reductions; ++reduction_idx)
  {
    DEB_TUPLE(("  reduction_idx=%d {\n", reduction_idx));
    CkReduction::reducerType reducerType = CkReduction::invalid;
    for (int message_idx = 0; message_idx < num_messages; ++message_idx)
    {
      CkReduction::tupleElement* reductions = (CkReduction::tupleElement*)(tuple_data[message_idx]);
      CkReduction::tupleElement& element = reductions[reduction_idx];
      DEB_TUPLE(("    msg %d, sf=%d, length=%d : { dataSize=%d, data=%p, reducer=%d },\n",
                 message_idx, messages[message_idx]->sourceFlag, messages[message_idx]->getLength(), element.dataSize, element.data, element.reducer));

      reducerType = element.reducer;

      size_t sim_idx = (reduction_idx * num_messages + message_idx) * sizeof(CkReductionMsg);
      CkReductionMsg& simulated_message = *(CkReductionMsg*)&simulated_messages_buffer[sim_idx];
      simulated_message.dataSize = element.dataSize;
      simulated_message.data = element.data;
      simulated_message.reducer = element.reducer;
      simulated_message.sourceFlag = messages[message_idx]->sourceFlag;
      simulated_message.userFlag = messages[message_idx]->userFlag;
      simulated_message.gcount = messages[message_idx]->gcount;
      simulated_message.migratableContributor = messages[message_idx]->migratableContributor;
#if CMK_BIGSIM_CHARM
      simulated_message.log = NULL;
#endif
      simulated_messages[message_idx] = &simulated_message;
    }

    // run the reduction and copy the result back to our data structure
    const auto& reducerFp = CkReduction::reducerTable[reducerType].fn;
    CkReductionMsg* result = reducerFp(num_messages, simulated_messages);
    DEB_TUPLE(("    result_len=%d\n  },\n", result->getLength()));
    return_data[reduction_idx] = CkReduction::tupleElement(result->getLength(), result->getData(), reducerType);
    // TODO - leak - the built in reducers all reuse message memory, so this is not safe to delete
    // delete result;
  }

  CkReductionMsg* retval = CkReductionMsg::buildFromTuple(return_data, num_reductions);
  DEB_TUPLE(("} tupleReduction msg_size=%d\n", retval->getSize()));

  for (int message_idx = 0; message_idx < num_messages; ++message_idx)
    delete[] tuple_data[message_idx];
  delete[] tuple_data;
  delete[] return_data;
  delete[] simulated_messages_buffer;
  // note that although this is a 2d array, we don't need to delete the inner objects,
  //  their memory is tracked in simulated_messages_buffer
  delete[] simulated_messages;
  return retval;
}



/////////////////// Reduction Function Table /////////////////////
CkReduction::CkReduction() {} //Dummy private constructor

//Add the given reducer to the list.  Returns the new reducer's
// reducerType.  Must be called in the same order on every node.
CkReduction::reducerType CkReduction::addReducer(reducerFn fn, bool streamable)
{
  reducerTable[nReducers].fn=fn;
  reducerTable[nReducers].streamable=streamable;
  return (reducerType)nReducers++;
}

/*Reducer table: maps reducerTypes to reducerStructs.
It's indexed by reducerType, so the order in this table
must *exactly* match the reducerType enum declaration.
The names don't have to match, but it helps.
*/
int CkReduction::nReducers=CkReduction::lastSystemReducer;

CkReduction::reducerStruct CkReduction::reducerTable[CkReduction::MAXREDUCERS]={
    CkReduction::reducerStruct(::invalid_reducer, true),
    CkReduction::reducerStruct(::nop, true),
    //Compute the sum the numbers passed by each element.
    CkReduction::reducerStruct(::sum_char, true),
    CkReduction::reducerStruct(::sum_short, true),
    CkReduction::reducerStruct(::sum_int, true),
    CkReduction::reducerStruct(::sum_long, true),
    CkReduction::reducerStruct(::sum_long_long, true),
    CkReduction::reducerStruct(::sum_uchar, true),
    CkReduction::reducerStruct(::sum_ushort, true),
    CkReduction::reducerStruct(::sum_uint, true),
    CkReduction::reducerStruct(::sum_ulong, true),
    CkReduction::reducerStruct(::sum_ulong_long, true),
    // The floating point sums are marked as unstreamable to avoid
    // implictly stating that they will always be precision oblivious.
    CkReduction::reducerStruct(::sum_float, false),
    CkReduction::reducerStruct(::sum_double, false),

    //Compute the product the numbers passed by each element.
    CkReduction::reducerStruct(::product_char, true),
    CkReduction::reducerStruct(::product_short, true),
    CkReduction::reducerStruct(::product_int, true),
    CkReduction::reducerStruct(::product_long, true),
    CkReduction::reducerStruct(::product_long_long, true),
    CkReduction::reducerStruct(::product_uchar, true),
    CkReduction::reducerStruct(::product_ushort, true),
    CkReduction::reducerStruct(::product_uint, true),
    CkReduction::reducerStruct(::product_ulong, true),
    CkReduction::reducerStruct(::product_ulong_long, true),
    CkReduction::reducerStruct(::product_float, true),
    CkReduction::reducerStruct(::product_double, true),

    //Compute the largest number passed by any element.
    CkReduction::reducerStruct(::max_char, true),
    CkReduction::reducerStruct(::max_short, true),
    CkReduction::reducerStruct(::max_int, true),
    CkReduction::reducerStruct(::max_long, true),
    CkReduction::reducerStruct(::max_long_long, true),
    CkReduction::reducerStruct(::max_uchar, true),
    CkReduction::reducerStruct(::max_ushort, true),
    CkReduction::reducerStruct(::max_uint, true),
    CkReduction::reducerStruct(::max_ulong, true),
    CkReduction::reducerStruct(::max_ulong_long, true),
    CkReduction::reducerStruct(::max_float, true),
    CkReduction::reducerStruct(::max_double, true),

    //Compute the smallest number passed by any element.
    CkReduction::reducerStruct(::min_char, true),
    CkReduction::reducerStruct(::min_short, true),
    CkReduction::reducerStruct(::min_int, true),
    CkReduction::reducerStruct(::min_long, true),
    CkReduction::reducerStruct(::min_long_long, true),
    CkReduction::reducerStruct(::min_uchar, true),
    CkReduction::reducerStruct(::min_ushort, true),
    CkReduction::reducerStruct(::min_uint, true),
    CkReduction::reducerStruct(::min_ulong, true),
    CkReduction::reducerStruct(::min_ulong_long, true),
    CkReduction::reducerStruct(::min_float, true),
    CkReduction::reducerStruct(::min_double, true),

    //Compute the logical AND of the values passed by each element.
    // The resulting value will be zero if any source value is zero.
    CkReduction::reducerStruct(::logical_and, true),
    CkReduction::reducerStruct(::logical_and_int, true),
    CkReduction::reducerStruct(::logical_and_bool, true),

    //Compute the logical OR of the values passed by each element.
    // The resulting value will be 1 if any source value is nonzero.
    CkReduction::reducerStruct(::logical_or, true),
    CkReduction::reducerStruct(::logical_or_int, true),
    CkReduction::reducerStruct(::logical_or_bool, true),

    //Compute the logical XOR of the values passed by each element.
    // The resulting value will be 1 if an odd number of source values is nonzero.
    CkReduction::reducerStruct(::logical_xor_int, true),
    CkReduction::reducerStruct(::logical_xor_bool, true),

    // Compute the logical bitvector AND of the values passed by each element.
    CkReduction::reducerStruct(::bitvec_and, true),
    CkReduction::reducerStruct(::bitvec_and_int, true),
    CkReduction::reducerStruct(::bitvec_and_bool, true),

    // Compute the logical bitvector OR of the values passed by each element.
    CkReduction::reducerStruct(::bitvec_or, true),
    CkReduction::reducerStruct(::bitvec_or_int, true),
    CkReduction::reducerStruct(::bitvec_or_bool, true),
    
    // Compute the logical bitvector XOR of the values passed by each element.
    CkReduction::reducerStruct(::bitvec_xor, true),
    CkReduction::reducerStruct(::bitvec_xor_int, true),
    CkReduction::reducerStruct(::bitvec_xor_bool, true),

    // Select one of the messages at random to pass on
    CkReduction::reducerStruct(::random, true),

    //Concatenate the (arbitrary) data passed by each element
    // This reduction is marked as unstreamable because of the n^2
    // work required to stream it
    CkReduction::reducerStruct(::concat, false),

    //Combine the data passed by each element into an list of setElements.
    // Each element may contribute arbitrary data (with arbitrary length).
    // This reduction is marked as unstreamable because of the n^2
    // work required to stream it
    CkReduction::reducerStruct(::set, false),

    CkReduction::reducerStruct(::statistics, true),
    CkReduction::reducerStruct(CkReduction::tupleReduction, false),
};








/********** Code added by Sayantan *********************/
/** Locking is a big problem in the nodegroup code for smp.
 So a few assumptions have had to be made. There is one lock
 called lockEverything. It protects all the data structures 
 of the nodegroup reduction mgr. I tried grabbing it separately 
 for each datastructure, modifying it and then releasing it and
 then grabbing it again, for the next change.
 That doesn't really help because the interleaved execution of 
 different threads makes the state of the reduction manager 
 inconsistent. 
 
 1. Grab lockEverything before calling finishreduction or startReduction
    or doRecvMsg
 2. lockEverything is grabbed only in entry methods reductionStarting
    or RecvMesg or  addcontribution.
 ****/
 
/**nodegroup reduction manager . Most of it is similar to the guy above***/
NodeGroup::NodeGroup(void) {
  __nodelock=CmiCreateLock();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    mlogData->objID.type = TypeNodeGroup;
    mlogData->objID.data.group.onPE = CkMyNode();
#endif

}
NodeGroup::~NodeGroup() {
  CmiDestroyLock(__nodelock);
  CkpvAccess(_destroyingNodeGroup) = true;
}
void NodeGroup::pup(PUP::er &p)
{
  CkNodeReductionMgr::pup(p);
  p|reductionInfo;
}

//CK_REDUCTION_CLIENT_DEF(CProxy_NodeGroup,(CkNodeReductionMgr *)CkLocalBranch(_ck_gid));

void CProxy_NodeGroup::ckSetReductionClient(CkCallback *cb) const {
  DEBR(("in CksetReductionClient for CProxy_NodeGroup %d\n",CkLocalNodeBranch(_ck_gid)));
 ((CkNodeReductionMgr *)CkLocalNodeBranch(_ck_gid))->ckSetReductionClient(cb);
  //ckLocalNodeBranch()->ckSetReductionClient(cb);
 }

CK_REDUCTION_CONTRIBUTE_METHODS_DEF(NodeGroup,
				    ((CkNodeReductionMgr *)this),
				    reductionInfo,false)

/* this contribute also adds up the count across all messages it receives.
  Useful for summing up number of array elements who have contributed ****/ 
void NodeGroup::contributeWithCounter(CkReductionMsg *msg,int count)
	{((CkNodeReductionMgr *)this)->contributeWithCounter(&reductionInfo,msg,count);}



//#define BINOMIAL_TREE

CkNodeReductionMgr::CkNodeReductionMgr()//Constructor
  : thisProxy(thisgroup)
{
#ifdef BINOMIAL_TREE
  init_BinomialTree();
#else
  init_BinaryTree();
#endif
  storedCallback=NULL;
  redNo=0;
  inProgress=false;
  
  startRequested=false;
  gcount=CkNumNodes();
  lcount=1;
  nContrib=nRemote=0;
  lockEverything = CmiCreateLock();


  creating=false;
  interrupt = 0;
  DEBR((AA "In NodereductionMgr constructor at %d \n" AB,this));
	/*
		FAULT_EVAC
	*/
	blocked = false;
	maxModificationRedNo = INT_MAX;
	killed=0;
	additionalGCount = newAdditionalGCount = 0;
}

CkNodeReductionMgr::~CkNodeReductionMgr()
{
  CmiDestroyLock(lockEverything);
}

void CkNodeReductionMgr::flushStates()
{
 if(CkMyRank() == 0){
  // CmiPrintf("[%d] CkNodeReductionMgr::flushState\n", CkMyPe());
  redNo=0;
  inProgress=false;

  startRequested=false;
  gcount=CkNumNodes();
  lcount=1;
  nContrib=nRemote=0;

  creating=false;
  interrupt = 0;
  while (!msgs.isEmpty()) { delete msgs.deq(); }
  while (!futureMsgs.isEmpty()) delete futureMsgs.deq();
  while (!futureRemoteMsgs.isEmpty()) delete futureRemoteMsgs.deq();
  while (!futureLateMigrantMsgs.isEmpty()) delete futureLateMigrantMsgs.deq();
  }
}

//////////// Reduction Manager Client API /////////////

//Add the given client function.  Overwrites any previous client.
void CkNodeReductionMgr::ckSetReductionClient(CkCallback *cb)
{
  DEBR((AA "Setting reductionClient in NodeReductionMgr %d at %d\n" AB,cb,this));
  if(cb->isInvalid()){
  	DEBR((AA "Invalid Callback passed to setReductionClient in nodeReductionMgr\n" AB));
  }else{
	DEBR((AA "Valid Callback passed to setReductionClient in nodeReductionMgr\n" AB));
  }

  if (CkMyNode()!=0)
	  CkError("WARNING: ckSetReductionClient should only be called from processor zero!\n");
  delete storedCallback;
  storedCallback=cb;
}



void CkNodeReductionMgr::contribute(contributorInfo *ci,CkReductionMsg *m)
{
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    Chare *oldObj =CpvAccess(_currentObj);
    CpvAccess(_currentObj) = this;
#endif

  //m->ci=ci;
  m->redNo=ci->redNo++;
  m->sourceFlag=-1;//A single contribution
  m->gcount=0;
  DEBR(("[%d,%d] NodeGroup %d> localContribute called for redNo %d \n",CkMyNode(),CkMyPe(),thisgroup.idx,m->redNo));
  addContribution(m);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = oldObj;
#endif
}


void CkNodeReductionMgr::contributeWithCounter(contributorInfo *ci,CkReductionMsg *m,int count)
{
#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&m->log);
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    Chare *oldObj =CpvAccess(_currentObj);
    CpvAccess(_currentObj) = this;
#endif
  //m->ci=ci;
  m->redNo=ci->redNo++;
  m->gcount=count;
  DEBR(("[%d,%d] contributewithCounter started for %d at %0.6f{{{\n",CkMyNode(),CkMyPe(),m->redNo,CmiWallTimer()));
  addContribution(m);
  DEBR(("[%d,%d] }}}contributewithCounter finished for %d at %0.6f\n",CkMyNode(),CkMyPe(),m->redNo,CmiWallTimer()));

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = oldObj;
#endif
}


//////////// Reduction Manager Remote Entry Points /////////////

//Sent down the reduction tree (used by barren PEs)
void CkNodeReductionMgr::ReductionStarting(CkReductionNumberMsg *m)
{
  DEBR((AA "[%d] Received reductionStarting redNo %d\n" AB, CkMyPe(), m->num));
  CmiLock(lockEverything);
	/*
		FAULT_EVAC
	*/
  if(blocked){
	delete m;
  	CmiUnlock(lockEverything);
	return ;
  }
  int srcNode = CmiNodeOf((UsrToEnv(m))->getSrcPe());
  if (isPresent(m->num) && !inProgress)
  {
    DEBR((AA "Starting Node reduction #%d at parent's request\n" AB,m->num));
    startReduction(m->num,srcNode);
    finishReduction();
  } else if (isFuture(m->num)){
  	DEBR(("[%d][%d] Message num %d Present redNo %d \n",CkMyNode(),CkMyPe(),m->num,redNo));
  }
  else //is Past
    DEBR((AA "Ignoring node parent's late request to start #%d\n" AB,m->num));
  CmiUnlock(lockEverything);
  delete m;

}


void CkNodeReductionMgr::doRecvMsg(CkReductionMsg *m){
	DEBR(("[%d,%d] doRecvMsg called for  %d at %.6f[[[[[\n",CkMyNode(),CkMyPe(),m->redNo,CkWallTimer()));
	/*
		FAULT_EVAC
	*/
	if(blocked){
		DEBR(("[%d] This node is blocked, so remote message is being buffered as no %d\n",CkMyNode(),bufferedRemoteMsgs.length()));
		bufferedRemoteMsgs.enq(m);
		return;
	}
	
	if (isPresent(m->redNo)) { //Is a regular, in-order reduction message
	    //DEBR((AA "Recv'd remote contribution %d for #%d at %d\n" AB,nRemote,m->redNo,this));
	    startReduction(m->redNo,CkMyNode());
	    msgs.enq(m);
	    nRemote++;
	    finishReduction();
	}
	else {
	    if (isFuture(m->redNo)) {
	    	   // DEBR((AA "Recv'd early remote contribution %d for #%d\n" AB,nRemote,m->redNo));
		    futureRemoteMsgs.enq(m);
	    }else{
		   CkPrintf("BIG Problem Present %d Mesg RedNo %d \n",redNo,m->redNo);	
		   CkAbort("Recv'd late remote contribution!\n");
	    }
	}
	DEBR(("[%d,%d]]]]] doRecvMsg called for  %d at %.6f\n",CkMyNode(),CkMyPe(),m->redNo,CkWallTimer()));
}

//Sent up the reduction tree with reduced data
void CkNodeReductionMgr::RecvMsg(CkReductionMsg *m)
{
#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&m->log);
#endif
#ifndef CMK_CPV_IS_SMP
#if CMK_IMMEDIATE_MSG
	if(interrupt == 1){
		//CkPrintf("$$$$$$$$$How did i wake up in the middle of someone else's entry method ?\n");
		CpvAccess(_qd)->process(-1);
		CmiDelayImmediate();
		return;
	}
#endif	
#endif
   interrupt = 1;	
   CmiLock(lockEverything);   
   DEBR(("[%d,%d] Recv'd REMOTE contribution for %d at %.6f[[[\n",CkMyNode(),CkMyPe(),m->redNo,CkWallTimer()));
   doRecvMsg(m);
   CmiUnlock(lockEverything);    
   interrupt = 0;
   DEBR(("[%d,%d] ]]]]]]Recv'd REMOTE contribution for %d at %.6f\n",CkMyNode(),CkMyPe(),m->redNo,CkWallTimer()));
}

void CkNodeReductionMgr::startReduction(int number,int srcNode)
{
	if (isFuture(number)) CkAbort("Can't start reductions out of order!\n");
	if (isPast(number)) CkAbort("Can't restart reduction that's already finished!\n");
	if (inProgress){
  		DEBR((AA "This Node reduction is already in progress\n" AB));
		return;//This reduction already started
	}
	if (creating) //Don't start yet-- we're creating elements
	{
		DEBR((AA " Node Postponing start request #%d until we're done creating\n" AB,redNo));
		startRequested=true;
		return;
	}
	
	//If none of these cases, we need to start the reduction--
	DEBR((AA "Starting Node reduction #%d on %p srcNode %d\n" AB,redNo,this,srcNode));
	inProgress=true;

	if(!_isNotifyChildInRed) return;

	//Sent start requests to our kids (in case they don't already know)
	
	for (int k=0;k<treeKids();k++)
	{
#ifdef BINOMIAL_TREE
		DEBR((AA "Asking child Node %d to start #%d\n" AB,kids[k],redNo));
		thisProxy[kids[k]].ReductionStarting(new CkReductionNumberMsg(redNo));
#else
		if(kids[k] != srcNode){
			DEBR((AA "Asking child Node %d to start #%d\n" AB,kids[k],redNo));
			thisProxy[kids[k]].ReductionStarting(new CkReductionNumberMsg(redNo));
		}	
#endif
	}

	DEBR((AA "Asking all local groups to start #%d\n" AB,redNo));
	// in case, nodegroup does not has the attached red group,
	// it has to restart groups again
	startLocalGroupReductions(number);
/*
	if (startLocalGroupReductions(number) == 0)
          thisProxy[CkMyNode()].restartLocalGroupReductions(number);
*/
}

// restart local groups until succeed
void CkNodeReductionMgr::restartLocalGroupReductions(int number) {
  CmiLock(lockEverything);    
  if (startLocalGroupReductions(number) == 0)
    thisProxy[CkMyNode()].restartLocalGroupReductions(number);
  CmiUnlock(lockEverything);    
}

void CkNodeReductionMgr::doAddContribution(CkReductionMsg *m){
	/*
		FAULT_EVAC
	*/
	if(blocked){
		DEBR(("[%d] This node is blocked, so local message is being buffered as no %d\n",CkMyNode(),bufferedMsgs.length()));
		bufferedMsgs.enq(m);
		return;
	}
	
	if (isFuture(m->redNo)) {//An early contribution-- add to future Q
		DEBR((AA "Contributor gives early node contribution-- for #%d\n" AB,m->redNo));
		futureMsgs.enq(m);
	} else {// An ordinary contribution
		DEBR((AA "Recv'd local node contribution %d for #%d at %d\n" AB,nContrib,m->redNo,this));
		//    CmiPrintf("[%d,%d] Redcv'd Local Contribution for redNo %d number %d at %0.6f \n",CkMyNode(),CkMyPe(),m->redNo,nContrib+1,CkWallTimer());
		startReduction(m->redNo,CkMyNode());
		msgs.enq(m);
		nContrib++;
		finishReduction();
	}
}

//Handle a message from one element for the reduction
void CkNodeReductionMgr::addContribution(CkReductionMsg *m)
{
  interrupt = 1;
  CmiLock(lockEverything);
  doAddContribution(m);
  CmiUnlock(lockEverything);
  interrupt = 0;
}

void CkNodeReductionMgr::LateMigrantMsg(CkReductionMsg *m){
        CmiLock(lockEverything);   
	/*
		FAULT_EVAC
	*/
	if(blocked){
		DEBR(("[%d] This node is blocked, so local message is being buffered as no %d\n",CkMyNode(),bufferedMsgs.length()));
		bufferedMsgs.enq(m);
                CmiUnlock(lockEverything);   
		return;
	}
	
	if (isFuture(m->redNo)) {//An early contribution-- add to future Q
		DEBR((AA "Latemigrant gives early node contribution-- for #%d\n" AB,m->redNo));
//		CkPrintf("[%d,%d] NodeGroup %d> Latemigrant gives early node contribution %d in redNo %d\n",CkMyNode(),CkMyPe(),thisgroup.idx,m->redNo,redNo);
		futureLateMigrantMsgs.enq(m);
	} else {// An ordinary contribution
		DEBR((AA "Recv'd late migrant contribution %d for #%d at %d\n" AB,nContrib,m->redNo,this));
//		CkPrintf("[%d,%d] NodeGroup %d> Latemigrant contribution %d in redNo %d\n",CkMyNode(),CkMyPe(),thisgroup.idx,m->redNo,redNo);
		msgs.enq(m);
		finishReduction();
	}
        CmiUnlock(lockEverything);   
}





/** check if the nodegroup reduction is finished at this node. In that case send it
up the reduction tree **/

void CkNodeReductionMgr::finishReduction(void)
{
  DEBR((AA "in Nodegrp finishReduction %d treeKids %d \n" AB,inProgress,treeKids()));
  /***Check if reduction is finished in the next few ifs***/
  if ((!inProgress) || creating){
  	DEBR((AA "Either not in Progress or creating\n" AB));
  	return;
  }

  bool partialReduction = false;

  if (nContrib<(lcount)){
    if (msgs.length() > 1 && CkReduction::reducerTable[msgs.peek()->reducer].streamable) {
      partialReduction = true;
    }
    else {
      DEBR((AA "Nodegrp Need more local messages %d %d\n" AB,nContrib,(lcount)));
      return;//Need more local messages
    }
  }
  if (nRemote<treeKids()){
    if (msgs.length() > 1 && CkReduction::reducerTable[msgs.peek()->reducer].streamable) {
      partialReduction = true;
    }
    else {
      DEBR((AA "Nodegrp Need more Remote messages %d %d\n" AB,nRemote,treeKids()));
      return;//Need more remote messages
    }
  }
  if (nRemote>treeKids()){

	  interrupt = 0;
	   CkAbort("Nodegrp Excess remote reduction message received!\n");
  }

  DEBR((AA "Reducing node data...\n" AB));

  /**reduce all messages received at this node **/
  CkReductionMsg *result=reduceMessages();

  if (partialReduction) {
    msgs.enq(result);
    return;
  }

  if (hasParent())
  {//Pass data up tree to parent
	if(CmiNodeAlive(CkMyNode()) || killed == 0){
    	DEBR((AA "Passing reduced data up to parent node %d. \n" AB,treeParent()));
    	DEBR(("[%d,%d] Passing data up to parentNode %d at %.6f for redNo %d with ncontrib %d\n",CkMyNode(),CkMyPe(),treeParent(),CkWallTimer(),redNo,nContrib));
		/*
			FAULT_EVAC
		*/
			result->gcount += additionalGCount;//if u have replaced some node add its gcount to ur count
	    thisProxy[treeParent()].RecvMsg(result);
	}

  }
  else
  {
		if(result->isMigratableContributor() && result->gcount+additionalGCount != result->sourceFlag){
		  DEBR(("[%d,%d] NodeGroup %d> Node Reduction %d not done yet gcounts %d sources %d migratable %d \n",CkMyNode(),CkMyPe(),thisgroup.idx,redNo,result->gcount,result->sourceFlag,result->isMigratableContributor()));
			msgs.enq(result);
			return;
		}
		result->gcount += additionalGCount;
	  /** if the reduction is finished and I am the root of the reduction tree
	  then call the reductionhandler and other stuff ***/
		

		DEBR(("[%d,%d]------------------- END OF REDUCTION %d with %d remote contributions passed to client function at %.6f\n",CkMyNode(),CkMyPe(),redNo,nRemote,CkWallTimer()));
    CkSetRefNum(result, result->getUserFlag());
    if (!result->callback.isInvalid()){
      DEBR(("[%d,%d] message Callback used \n",CkMyNode(),CkMyPe()));
	    result->callback.send(result);
    }
    else if (storedCallback!=NULL){
      DEBR(("[%d,%d] stored Callback used \n",CkMyNode(),CkMyPe()));
	    storedCallback->send(result);
    }
    else{
    		DEBR((AA "Invalid Callback \n" AB));
	    CkAbort("No reduction client!\n"
		    "You must register a client with either SetReductionClient or during contribute.\n");
		}
  }

  // DEBR((AA "Reduction %d finished in group!\n" AB,redNo));
  //CkPrintf("[%d,%d]Reduction %d finished with %d\n",CkMyNode(),CkMyPe(),redNo,nContrib);
  redNo++;
	updateTree();
  int i;
  inProgress=false;
  startRequested=false;
  nRemote=nContrib=0;

  //Look through the future queue for messages we can now handle
  int n=futureMsgs.length();

  for (i=0;i<n;i++)
  {
    interrupt = 1;

    CkReductionMsg *m=futureMsgs.deq();

    interrupt = 0;
    if (m!=NULL){ //One of these addContributions may have finished us.
      DEBR(("[%d,%d] NodeGroup %d> Mesg with redNo %d might be useful in new reduction %d \n",CkMyNode(),CkMyPe(),thisgroup.idx,m->redNo,redNo));
      doAddContribution(m);//<- if *still* early, puts it back in the queue
    }
  }

  interrupt = 1;

  n=futureRemoteMsgs.length();

  interrupt = 0;
  for (i=0;i<n;i++)
  {
    interrupt = 1;

    CkReductionMsg *m=futureRemoteMsgs.deq();

    interrupt = 0;
    if (m!=NULL)
      doRecvMsg(m);//<- if *still* early, puts it back in the queue
  }
  
  n = futureLateMigrantMsgs.length();
  for(i=0;i<n;i++){
    CkReductionMsg *m = futureLateMigrantMsgs.deq();
    if(m != NULL){
      if(m->redNo == redNo){
        msgs.enq(m);
      }else{
        futureLateMigrantMsgs.enq(m);
      }
    }
  }
}

//////////// Reduction Manager Utilities /////////////

void CkNodeReductionMgr::init_BinaryTree(){
	parent = (CkMyNode()-1)/TREE_WID;
	int firstkid = CkMyNode()*TREE_WID+1;
	numKids=CkNumNodes()-firstkid;
  if (numKids>TREE_WID) numKids=TREE_WID;
  if (numKids<0) numKids=0;

	for(int i=0;i<numKids;i++){
		kids.push_back(firstkid+i);
		newKids.push_back(firstkid+i);
	}
}

void CkNodeReductionMgr::init_BinomialTree(){
	int depth = (int )ceil((log((double )CkNumNodes())/log((double)2)));
	/*upperSize = (unsigned )pow((double)2,depth);*/
	upperSize = (unsigned) 1 << depth;
	label = upperSize-CkMyNode()-1;
	int p=label;
	int count=0;
	while( p > 0){
		if(p % 2 == 0)
			break;
		else{
			p = p/2;
			count++;
		}
	}
	/*parent = label + rint(pow((double)2,count));*/
	parent = label + (1<<count);
	parent = upperSize -1 -parent;
	int temp;
	if(count != 0){
		numKids = 0;
		for(int i=0;i<count;i++){
			/*temp = label - rint(pow((double)2,i));*/
			temp = label - (1<<i);
			temp = upperSize-1-temp;
			if(temp <= CkNumNodes()-1){
		//		kids[numKids] = temp;
				kids.push_back(temp);
				numKids++;
			}
		}
	}else{
		numKids = 0;
	//	kids = NULL;
	}
}


int CkNodeReductionMgr::treeRoot(void)
{
  return 0;
}
bool CkNodeReductionMgr::hasParent(void) //Root Node
{
  return (bool)(CkMyNode()!=treeRoot());
}
int CkNodeReductionMgr::treeParent(void) //My parent Node
{
  return parent;
}

int CkNodeReductionMgr::firstKid(void) //My first child Node
{
  return CkMyNode()*TREE_WID+1;
}
int CkNodeReductionMgr::treeKids(void)//Number of children in tree
{
#ifdef BINOMIAL_TREE
	return numKids;
#else
/*  int nKids=CkNumNodes()-firstKid();
  if (nKids>TREE_WID) nKids=TREE_WID;
  if (nKids<0) nKids=0;
  return nKids;*/
	return numKids;
#endif
}

//Combine (& free) the current message vector msgs.
CkReductionMsg *CkNodeReductionMgr::reduceMessages(void)
{
#if CMK_BIGSIM_CHARM
  _TRACE_BG_END_EXECUTE(1);
  void* _bgParentLog = NULL;
  _TRACE_BG_BEGIN_EXECUTE_NOMSG("NodeReduce", &_bgParentLog, 0);
#endif
  CkReductionMsg *ret=NULL;

  //Look through the vector for a valid reducer, swapping out placeholder messages
  CkReduction::reducerType r=CkReduction::invalid;
  int msgs_gcount=0;//Reduced gcount
  int msgs_nSources=0;//Reduced nSources
  CMK_REFNUM_TYPE msgs_userFlag=(CMK_REFNUM_TYPE)-1;
  CkCallback msgs_callback;
  CkCallback msgs_secondaryCallback;
  int i;
  int nMsgs=0;
  CkReductionMsg *m;
  CkReductionMsg **msgArr=new CkReductionMsg*[msgs.length()];
  bool isMigratableContributor;
	

  while(NULL!=(m=msgs.deq()))
  {
    DEBR((AA "***** gcount=%d; sourceFlag=%d ismigratable %d \n" AB,m->gcount,m->nSources(),m->isMigratableContributor()));	  
    msgs_gcount+=m->gcount;
    if (m->sourceFlag!=0)
    { //This is a real message from an element, not just a placeholder
      msgs_nSources+=m->nSources();
#if CMK_BIGSIM_CHARM
      _TRACE_BG_ADD_BACKWARD_DEP(m->log);
#endif

      if (nMsgs == 0 || m->reducer != CkReduction::nop) {
        msgArr[nMsgs++]=m;
        r=m->reducer;
        if (!m->callback.isInvalid()){
#if CMK_ERROR_CHECKING
          if(nMsgs > 1 && !(msgs_callback == m->callback))
            CkAbort("mis-matched client callbacks in reduction messages\n");
#endif  
          msgs_callback=m->callback;
	}
        if(!m->secondaryCallback.isInvalid()){
          msgs_secondaryCallback = m->secondaryCallback;
        }
        if (m->userFlag!=(CMK_REFNUM_TYPE)-1)
          msgs_userFlag=m->userFlag;
	isMigratableContributor= m->isMigratableContributor();
      }
      else {
#if CMK_ERROR_CHECKING
        if(!(msgs_callback == m->callback))
          CkAbort("mis-matched client callbacks in reduction messages\n");
#endif  
        delete m;
      }
    }
    else
    { //This is just a placeholder message-- replace it
      delete m;
    }
  }

  if (nMsgs==0||r==CkReduction::invalid)
  //No valid reducer in the whole vector
    ret=CkReductionMsg::buildNew(0,NULL);
  else
  {//Use the reducer to reduce the messages
    if(nMsgs == 1){
      ret = msgArr[0];
    }else{
      if (msgArr[0]->reducer == CkReduction::nop) {
        // nMsgs > 1 indicates that reduction type is not nop
        // this means any data with reducer type nop was submitted
        // only so that counts would agree, and can be removed
        delete msgArr[0];
        msgArr[0] = msgArr[nMsgs - 1];
        nMsgs--;
      }
      CkReduction::reducerFn f=CkReduction::reducerTable[r].fn;
      ret=(*f)(nMsgs,msgArr);
    }
    ret->reducer=r;
  }

	
#if USE_CRITICAL_PATH_HEADER_ARRAY
#if CRITICAL_PATH_DEBUG > 3
	CkPrintf("[%d] combining critical path information from messages in CkNodeReductionMgr::reduceMessages(). numMsgs=%d\n", CkMyPe(), nMsgs);
#endif
	MergeablePathHistory path(CkpvAccess(currentlyExecutingPath));
	path.updateMax(UsrToEnv(ret));
	// Combine the critical paths from all the reduction messages into the header for the new result
	for (i=0;i<nMsgs;i++){
	  if (msgArr[i]!=ret){
	    //	    CkPrintf("[%d] other path = %lf\n", CkMyPe(), UsrToEnv(msgArr[i])->pathHistory.getTime() );
	    path.updateMax(UsrToEnv(msgArr[i]));
	  } else {
	    //	    CkPrintf("[%d] other path is ret = %lf\n", CkMyPe(), UsrToEnv(msgArr[i])->pathHistory.getTime() );
	  }
	}
#if CRITICAL_PATH_DEBUG > 3
	CkPrintf("[%d] result path = %lf\n", CkMyPe(), path.getTime() );
#endif

#endif


	//Go back through the vector, deleting old messages
  for (i=0;i<nMsgs;i++) if (msgArr[i]!=ret) delete msgArr[i];
  delete [] msgArr;
  //Set the message counts
  ret->redNo=redNo;
  ret->gcount=msgs_gcount;
  ret->userFlag=msgs_userFlag;
  ret->callback=msgs_callback;
  ret->secondaryCallback = msgs_secondaryCallback;
  ret->sourceFlag=msgs_nSources;
  ret->setMigratableContributor(isMigratableContributor);
  DEBR((AA "Node Reduced gcount=%d; sourceFlag=%d\n" AB,ret->gcount,ret->sourceFlag));
#if CMK_BIGSIM_CHARM
  _TRACE_BG_TLINE_END(&ret->log);
#endif

  return ret;
}

void CkNodeReductionMgr::pup(PUP::er &p)
{
//We do not store the client function pointer or the client function parameter,
//it is the responsibility of the programmer to correctly restore these
  IrrGroup::pup(p);
  p(redNo);
  p(inProgress); p(creating); p(startRequested);
  p(lcount);
  p(nContrib); p(nRemote);
  p(interrupt);
  p|msgs;
  p|futureMsgs;
  p|futureRemoteMsgs;
  p|futureLateMigrantMsgs;
  p|parent;
  p|additionalGCount;
  p|newAdditionalGCount;
  if(p.isUnpacking()) {
    gcount=CkNumNodes();
    thisProxy = thisgroup;
    lockEverything = CmiCreateLock();
#ifdef BINOMIAL_TREE
    init_BinomialTree();
#else
    init_BinaryTree();
#endif		
  }
  p | blocked;
  p | maxModificationRedNo;

#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
  int isnull = (storedCallback == NULL);
  p | isnull;
  if (!isnull) {
    if (p.isUnpacking()) {
      storedCallback = new CkCallback;
    }
    p|*storedCallback;
  }
#endif

}

/*
	FAULT_EVAC
	Evacuate - is called when this processor realizes it might crash. In that case, it tries to change 
	the reduction tree. It also needs to decide a reduction number after which it shall use the new 
	reduction tree. 
*/
void CkNodeReductionMgr::evacuate(){
	DEBREVAC(("[%d] Evacuate called on nodereductionMgr \n",CkMyNode()));
	if(treeKids() == 0){
	/*
		if the node going down is a leaf
	*/
		oldleaf=true;
		DEBREVAC(("[%d] Leaf Node marks itself for deletion when evacuation is complete \n",CkMyNode()));
		/*
			Need to ask parent for the reduction number that it has seen. 
			Since it is a leaf, the tree does not need to be rewired. 
			We reuse the oldparent type of tree modification message to get 
			the parent to block and tell us about the highest reduction number it has seen.
			
		*/
		int data[2];
		data[0]=CkMyNode();
		data[1]=getTotalGCount()+additionalGCount;
		thisProxy[treeParent()].modifyTree(LEAFPARENT,2,data);
		newParent = treeParent();
	}else{
		DEBREVAC(("[%d]%d> Internal Node sends messages to change the redN tree \n",CkMyNode(),thisgroup.idx));
		oldleaf= false;
	/*
		It is not a leaf. It needs to rewire the tree around itself.
		It also needs to decide on a reduction No after which the new tree will be used
		Till it decides on the new tree and the redNo at which it becomes valid,
		all received messages will be buffered
	*/
		newParent = kids[0];
		for(int i=numKids-1;i>=0;i--){
			newKids.remove(i);
		}
		/*
			Ask everybody for the highest reduction number they have seen and
			also tell them about the new tree
		*/
		/*
			Tell parent about its new child;
		*/
		int oldParentData[2];
		oldParentData[0] = CkMyNode();
		oldParentData[1] = newParent;
		thisProxy[parent].modifyTree(OLDPARENT,2,oldParentData);

		/*
			Tell the other children about their new parent
		*/
		int childrenData=newParent;
		for(int i=1;i<numKids;i++){
			thisProxy[kids[i]].modifyTree(OLDCHILDREN,1,&childrenData);
		}
		
		/*
			Tell newParent (1st child) about its new children,
			the current node and its children except the newParent
		*/
		int *newParentData = new int[numKids+2];
		for(int i=1;i<numKids;i++){
			newParentData[i] = kids[i];
		}
		newParentData[0] = CkMyNode();
		newParentData[numKids] = parent;
		newParentData[numKids+1] = getTotalGCount()+additionalGCount;
		thisProxy[newParent].modifyTree(NEWPARENT,numKids+2,newParentData);
	}
	readyDeletion = false;
	blocked = true;
	numModificationReplies = 0;
	tempModificationRedNo = findMaxRedNo();
}

/*
	Depending on the code, use the data to change the tree
	1. OLDPARENT : replace the old child with a new one
	2. OLDCHILDREN: replace the parent
	3. NEWPARENT:  add the children and change the parent
	4. LEAFPARENT: delete the old child
*/

void CkNodeReductionMgr::modifyTree(int code,int size,int *data){
	DEBREVAC(("[%d]%d> Received modifyTree request with code %d \n",CkMyNode(),thisgroup.idx,code));
	int sender;
	newKids = kids;
	readyDeletion = false;
	newAdditionalGCount = additionalGCount;
	switch(code){
		case OLDPARENT: 
			for(int i=0;i<numKids;i++){
				if(newKids[i] == data[0]){
					newKids[i] = data[1];
					break;
				}
			}
			sender = data[0];
			newParent = parent;
			break;
		case OLDCHILDREN:
			newParent = data[0];
			sender = parent;
			break;
		case NEWPARENT:
			for(int i=0;i<size-2;i++){
				newKids.push_back(data[i]);
			}
			newParent = data[size-2];
			newAdditionalGCount += data[size-1];
			sender = parent;
			break;
		case LEAFPARENT:
			for(int i=0;i<numKids;i++){
				if(newKids[i] == data[0]){
					newKids.remove(i);
					break;
				}
			}
			sender = data[0];
			newParent = parent;
			newAdditionalGCount += data[1];
			break;
	};
	blocked = true;
	int maxRedNo = findMaxRedNo();
	
	thisProxy[sender].collectMaxRedNo(maxRedNo);
}

void CkNodeReductionMgr::collectMaxRedNo(int maxRedNo){
	/*
		Find out the maximum redNo that has been seen by 
		the affected nodes
	*/
	numModificationReplies++;
	if(maxRedNo > tempModificationRedNo){
		tempModificationRedNo = maxRedNo;
	}
	if(numModificationReplies == numKids+1){
		maxModificationRedNo = tempModificationRedNo;
		/*
			when all the affected nodes have replied, tell them the maximum.
			Unblock yourself. deal with the buffered messages local and remote
		*/
		if(maxModificationRedNo == -1){
			printf("[%d]%d> This array has not started reductions yet \n",CkMyNode(),thisgroup.idx);
		}else{
			DEBREVAC(("[%d]%d> maxModificationRedNo for this nodegroup %d \n",CkMyNode(),thisgroup.idx,maxModificationRedNo));
		}
		thisProxy[parent].unblockNode(maxModificationRedNo);
		for(int i=0;i<numKids;i++){
			thisProxy[kids[i]].unblockNode(maxModificationRedNo);
		}
		blocked = false;
		updateTree();
		clearBlockedMsgs();
	}
}

void CkNodeReductionMgr::unblockNode(int maxRedNo){
	maxModificationRedNo = maxRedNo;
	updateTree();
	blocked = false;
	clearBlockedMsgs();
}


void CkNodeReductionMgr::clearBlockedMsgs(){
	int len = bufferedMsgs.length();
	for(int i=0;i<len;i++){
		CkReductionMsg *m = bufferedMsgs.deq();
		doAddContribution(m);
	}
	len = bufferedRemoteMsgs.length();
	for(int i=0;i<len;i++){
		CkReductionMsg *m = bufferedRemoteMsgs.deq();
		doRecvMsg(m);
	}

}
/*
	if the reduction number exceeds the maxModificationRedNo, change the tree
	to become the new one
*/

void CkNodeReductionMgr::updateTree(){
	if(redNo > maxModificationRedNo){
		parent = newParent;
		kids = newKids;
		maxModificationRedNo = INT_MAX;
		numKids = kids.size();
		readyDeletion = true;
		additionalGCount = newAdditionalGCount;
		DEBREVAC(("[%d]%d> Updating Tree numKids %d -> ",CkMyNode(),thisgroup.idx,numKids));
		for(int i=0;i<(int)(newKids.size());i++){
			DEBREVAC(("%d ",newKids[i]));
		}
		DEBREVAC(("\n"));
	//	startReduction(redNo,CkMyNode());
	}else{
		if(maxModificationRedNo != INT_MAX){
			DEBREVAC(("[%d]%d> Updating delayed because redNo %d maxModificationRedNo %d \n",CkMyNode(),thisgroup.idx,redNo,maxModificationRedNo));
			startReduction(redNo,CkMyNode());
			finishReduction();
		}	
	}
}


void CkNodeReductionMgr::doneEvacuate(){
	DEBREVAC(("[%d] doneEvacuate called \n",CkMyNode()));
/*	if(oldleaf){
		
			It used to be a leaf
			Then as soon as future messages have been emptied you can 
			send the parent a message telling them that they are not going
			to receive anymore messages from this child
		
		DEBR(("[%d] At the end of evacuation emptying future messages %d \n",CkMyNode(),futureMsgs.length()));
		while(futureMsgs.length() != 0){
			int n = futureMsgs.length();
			for(int i=0;i<n;i++){
				CkReductionMsg *m = futureMsgs.deq();
				if(isPresent(m->redNo)){
					msgs.enq(m);
				}else{
					futureMsgs.enq(m);
				}
			}
			CkReductionMsg *result = reduceMessages();
			thisProxy[treeParent()].RecvMsg(result);
			redNo++;
		}
		DEBR(("[%d] Asking parent %d to remove myself from list \n",CkMyNode(),treeParent()));
		thisProxy[treeParent()].DeleteChild(CkMyNode());
	}else{*/
		if(readyDeletion){
			thisProxy[treeParent()].DeleteChild(CkMyNode());
		}else{
			thisProxy[newParent].DeleteNewChild(CkMyNode());
		}
//	}
}

void CkNodeReductionMgr::DeleteChild(int deletedChild){
	DEBREVAC(("[%d]%d> Deleting child %d \n",CkMyNode(),thisgroup.idx,deletedChild));
	for(int i=0;i<numKids;i++){
		if(kids[i] == deletedChild){
			kids.remove(i);
			break;
		}
	}
	numKids = kids.length();
	finishReduction();
}

void CkNodeReductionMgr::DeleteNewChild(int deletedChild){
	for(int i=0;i<(int)(newKids.length());i++){
		if(newKids[i] == deletedChild){
			newKids.remove(i);
			break;
		}
	}
	DEBREVAC(("[%d]%d> Deleting  new child %d readyDeletion %d newKids %d -> ",CkMyNode(),thisgroup.idx,deletedChild,readyDeletion,newKids.size()));
	for(int i=0;i<(int)(newKids.size());i++){
		DEBREVAC(("%d ",newKids[i]));
	}
	DEBREVAC(("\n"));
	finishReduction();
}

int CkNodeReductionMgr::findMaxRedNo(){
	int max = redNo;
	for(int i=0;i<futureRemoteMsgs.length();i++){
		if(futureRemoteMsgs[i]->redNo  > max){
			max = futureRemoteMsgs[i]->redNo;
		}
	}
	/*
		if redNo is max (that is no future message) and the current reduction has not started
		then tree can be changed before the reduction redNo can be started
	*/ 
	if(redNo == max && msgs.length() == 0){
		DEBREVAC(("[%d] Redn %d has not received any contributions \n",CkMyNode(),max));
		max--;
	}
	return max;
}

// initnode call. check the size of reduction table
void CkReductionMgr::sanitycheck()
{
#if CMK_ERROR_CHECKING
  int count = 0;
  while (CkReduction::reducerTable[count].fn != NULL) count++;
  CmiAssert(CkReduction::nReducers == count);
#endif
}

#include "CkReduction.def.h"
