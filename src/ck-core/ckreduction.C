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

#if 0
//Debugging messages:
// Reduction mananger internal information:
#define DEBR(x) CkPrintf x
#define AA "Red PE%d Node%d #%d (%d,%d)> "
#define AB ,CkMyPe(),CkMyNode(),redNo,nRemote,nContrib

#define DEBN(x) CkPrintf x
#define AAN "Red Node%d "
#define ABN ,CkMyNode()

// For status and data messages from the builtin reducer functions.
#define RED_DEB(x) //CkPrintf x

#else
//No debugging info-- empty defines
#define DEBR(x) //CkPrintf x
#define DEBN(x) //CkPrintf x
#define RED_DEB(x) //CkPrintf x
#endif

Group::Group()
{
	creatingContributors();
	contributorStamped(&reductionInfo);
	contributorCreated(&reductionInfo);
	doneCreatingContributors();
	nodeProxyPtr = new CProxy_CkArrayReductionMgr(CkpvAccess(_currentGroupRednMgr));


}


CK_REDUCTION_CONTRIBUTE_METHODS_DEF(Group,
				    ((CkReductionMgr *)this),
				    reductionInfo);
CK_REDUCTION_CLIENT_DEF(CProxy_Group,(CkReductionMgr *)CkLocalBranch(_ck_gid));


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
class CkReductionNumberMsg:public CMessage_CkReductionNumberMsg {
public:
  int num;
  CkReductionNumberMsg(int n) {num=n;}
};

/*
One CkReductionMgr runs a non-overlapping set of reductions.
It collects messages from all local contributors, then sends
the reduced message up the reduction tree to node zero, where
they're passed to the user's client function.
*/

CkReductionMgr::CkReductionMgr()//Constructor
  : thisProxy(thisgroup)
{
  storedCallback=NULL;
  redNo=0;
  completedRedNo = -1;
  inProgress=CmiFalse;
  creating=CmiFalse;
  startRequested=CmiFalse;
  gcount=lcount=0;
  nContrib=nRemote=0;
  DEBR((AA"In reductionMgr constructor at %d \n"AB,this));
}

//////////// Reduction Manager Client API /////////////

//Add the given client function.  Overwrites any previous client.
void CkReductionMgr::ckSetReductionClient(CkCallback *cb)
{
  DEBR((AA"Setting reductionClient in ReductionMgr"AB));

  if (CkMyPe()!=0)
	  CkError("WARNING: ckSetReductionClient should only be called from processor zero!\n");
  delete storedCallback;
  storedCallback=cb;
  CkCallback *callback =new CkCallback(CkIndex_CkReductionMgr::ArrayReductionHandler(0),thishandle);
  (*nodeProxyPtr).ckSetReductionClient(callback);
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
  DEBR((AA"Creating contributors...\n"AB));
  creating=CmiTrue;
}
void CkReductionMgr::doneCreatingContributors(void)
{
  DEBR((AA"Done creating contributors...\n"AB));
  creating=CmiFalse;
  if (startRequested) startReduction(redNo);
  finishReduction();
}

//A new contributor will be created
void CkReductionMgr::contributorStamped(contributorInfo *ci)
{
  DEBR((AA"Contributor %p stamped\n"AB,ci));
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
  DEBR((AA"Contributor %p created\n"AB,ci));
  //We've got another contributor
  lcount++;
  //He may not need to contribute to some of our reductions:
  for (int r=redNo;r<ci->redNo;r++)
    adj(r).lcount--;//He won't be contributing to r here
}

/*Don't expect any more contributions from this one.
This is rather horrifying because we now have to make
sure the global element count accurately reflects all the
contributions the element made before it died-- these may stretch
far into the future.  The adj() vector is what saves us here.
*/
void CkReductionMgr::contributorDied(contributorInfo *ci)
{
  DEBR((AA"Contributor %p(%d) died\n"AB,ci,ci->redNo));
  //We lost a contributor
  gcount--;

  if (ci->redNo<redNo)
  {//Must have been migrating during reductions-- root is waiting for his
  // contribution, which will never come.
    DEBR((AA"Dying guy %p must have been migrating-- he's at #%d!\n"AB,ci,ci->redNo));
    for (int r=ci->redNo;r<redNo;r++)
      thisProxy[treeRoot()].MigrantDied(new CkReductionNumberMsg(r));
  }

  //Add to the global count for all his future messages (wherever they are)
  int r;
  for (r=redNo;r<ci->redNo;r++)
  {//He already contributed to this reduction, but won't show up in global count.
    DEBR((AA"Dead guy %p left contribution for #%d\n"AB,ci,r));
    adj(r).gcount++;
  }

  lcount--;
  //He's already contributed to several reductions here
  for (r=redNo;r<ci->redNo;r++)
    adj(r).lcount++;//He'll be contributing to r here

  finishReduction();
}
//Migrating away (note that global count doesn't change)
void CkReductionMgr::contributorLeaving(contributorInfo *ci)
{
  DEBR((AA"Contributor %p(%d) migrating away\n"AB,ci,ci->redNo));
  lcount--;//We lost a local
  //He's already contributed to several reductions here
  for (int r=redNo;r<ci->redNo;r++)
    adj(r).lcount++;//He'll be contributing to r here

  finishReduction();
}
//Migrating in (note that global count doesn't change)
void CkReductionMgr::contributorArriving(contributorInfo *ci)
{
  DEBR((AA"Contributor %p(%d) migrating in\n"AB,ci,ci->redNo));
  lcount++;//We gained a local
  //He has already contributed (elsewhere) to several reductions:
  for (int r=redNo;r<ci->redNo;r++)
    adj(r).lcount--;//He won't be contributing to r here

}

//Contribute-- the given msg can contain any data.  The reducerType
// field of the message must be valid.
// Each contributor must contribute exactly once to the each reduction.
void CkReductionMgr::contribute(contributorInfo *ci,CkReductionMsg *m)
{
  m->ci=ci;
  m->redNo=ci->redNo++;
  m->sourceFlag=-1;//A single contribution
  m->gcount=0;
  addContribution(m);
//  CkPrintf("$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
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
  if (isPresent(m->num) && !inProgress)
  {
    DEBR((AA"Starting reduction #%d at parent's request\n"AB,m->num));
    startReduction(m->num);
    finishReduction();
  } else if (isFuture(m->num)){
//   CkPrintf("[%d] arrays Mesg No %d redNo %d \n",CkMyPe(),m->num,redNo);
 //   CkAbort("My reduction tree parent somehow got ahead of me! in arrays\n");
    }
  else //is Past
    DEBR((AA"Ignoring parent's late request to start #%d\n"AB,m->num));
  delete m;
}
//Sent to root of reduction tree with reduction contribution
// of migrants that missed the main reduction.
void CkReductionMgr::LateMigrantMsg(CkReductionMsg *m)
{
	int len = finalMsgs.length();
	finalMsgs.enq(m);
//	CkPrintf("[%d]Late Migrant Detected for %d ,  (%d %d )\n",CkMyPe(),m->redNo,len,finalMsgs.length());
	endArrayReduction();
}

//A late migrating contributor will never contribute to this reduction
void CkReductionMgr::MigrantDied(CkReductionNumberMsg *m)
{
  if (hasParent() || m->num < completedRedNo) CkAbort("Late MigrantDied message recv'd!\n");
  DEBR((AA"Migrant died before contributing to #%d\n"AB,m->num));
 // CkPrintf("[%d,%d]Migrant Died called\n",CkMyNode(),CkMyPe());	 		  
  adj(m->num).gcount--;//He won't be contributing to this one.
  finishReduction();
}
//Sent up the reduction tree with reduced data
void CkReductionMgr::RecvMsg(CkReductionMsg *m)
{
  if (isPresent(m->redNo)) { //Is a regular, in-order reduction message
    DEBR((AA"Recv'd remote contribution %d for #%d at %d\n"AB,nRemote,m->redNo,this));
    startReduction(m->redNo);
    msgs.push_back(m);
    nRemote++;
    finishReduction();
  }
  else if (isFuture(m->redNo)) {
    DEBR((AA"Recv'd early remote contribution %d for #%d\n"AB,nRemote,m->redNo));
    futureRemoteMsgs.enq(m);
  }
  else CkAbort("Recv'd late remote contribution!\n");
}
//////////// Reduction Manager State /////////////
void CkReductionMgr::startReduction(int number)
{
  if (isFuture(number)){ /*CkAbort("Can't start reductions out of order!\n");*/ return;}
  if (isPast(number)) {/*CkAbort("Can't restart reduction that's already finished!\n");*/return;}
  if (inProgress){
  	DEBR((AA"This reduction is already in progress\n"AB));
  	return;//This reduction already started
  }
  if (creating) //Don't start yet-- we're creating elements
  {
    DEBR((AA"Postponing start request #%d until we're done creating\n"AB,redNo));
    startRequested=CmiTrue;
    return;
  }

//If none of these cases, we need to start the reduction--
  DEBR((AA"Starting reduction #%d\n"AB,redNo));
  inProgress=CmiTrue;
  //Sent start requests to our kids (in case they don't already know)
  //CkPrintf("[%d] Start Reduction Mesg %d  red No %d firstKid %d\n",CkMyPe(),number,redNo,firstKid());

  //making it a broadcast done only by PE 0

  //if(CkMyPe()==0){
	thisProxy.ReductionStarting(new CkReductionNumberMsg(number));
  //}

  /*for (int k=0;k<treeKids();k++)
  {
    DEBR((AA"Asking child PE %d to start #%d\n"AB,firstKid()+k,redNo));
    //if(firstKid()+k == 0){
    	CkPrintf("[%d]Sent ReductionStarting to %d \n",CkMyPe(),firstKid()+k);
    //}
    thisProxy[firstKid()+k].ReductionStarting(new CkReductionNumberMsg(number));
  }*/
}
/*Handle a message from one element for the reduction*/
void CkReductionMgr::addContribution(CkReductionMsg *m)
{
  if (isPast(m->redNo))
  {//We've moved on-- forward late contribution straight to root
    DEBR((AA"Migrant %p gives late contribution for #%d!\n"AB,m->ci,m->redNo));
   // if (!hasParent()) //Root moved on too soon-- should never happen
   //   CkAbort("Late reduction contribution received at root!\n");
    thisProxy[treeRoot()].LateMigrantMsg(m);
  }
  else if (isFuture(m->redNo)) {//An early contribution-- add to future Q
    DEBR((AA"Contributor %p gives early contribution-- for #%d\n"AB,m->ci,m->redNo));
    futureMsgs.enq(m);
  } else {// An ordinary contribution
    DEBR((AA"Recv'd local contribution %d for #%d at %d\n"AB,nContrib,m->redNo,this));
   // CkPrintf("[%d] Local Contribution for %d in Mesg %d at %.6f\n",CkMyPe(),redNo,m->redNo,CmiWallTimer());
    startReduction(m->redNo);
    msgs.push_back(m);
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
  DEBR((AA"in finishReduction %d \n"AB,inProgress));
  if ((!inProgress) | creating){
  	DEBR((AA"Either not in Progress or creating\n"AB));
  	return;
  }
  //CkPrintf("[%d]finishReduction called for redNo %d with nContrib %d at %.6f\n",CkMyPe(),redNo, nContrib,CmiWallTimer());
  if (nContrib<(lcount+adj(redNo).lcount)){
	DEBR((AA"Need more local messages %d %d\n"AB,nContrib,(lcount+adj(redNo).lcount)));
	 return;//Need more local messages
  }
  DEBR((AA"Reducing data...\n"AB));
  CkReductionMsg *result=reduceMessages();

  //CkPrintf("[%d] Got all local Messages in finishReduction %d in redNo %d\n",CkMyPe(),nContrib,redNo);

  /* reduce the messages and then store the callback specified in the message ***/
  CkReductionMsg *ret =CkReductionMsg::buildNew(result->getSize(),result->getData(),result->getReducer());
  ret->redNo = redNo;
  ret->userFlag= result->userFlag;
  ret->sourceFlag = result->sourceFlag;
  ret->gcount=result->gcount+gcount+adj(redNo).gcount;
  ret->callback = CkCallback(CkIndex_CkReductionMgr::ArrayReductionHandler(NULL),0,thisProxy);
  secondaryStoredCallback = new CkCallback();
  *secondaryStoredCallback = result->callback;
  callbackQ.enq(secondaryStoredCallback);
  
  //CkPrintf("[%d]Passing things up the NodeGroup reduction tree for redNo %d at %0.6f\n",CkMyPe(),redNo,CmiWallTimer());
  (*nodeProxyPtr)[CkMyNode()].ckLocalBranch()->contributeArrayReduction(ret);


  //House Keeping Operations will have to check later what needs to be changed
  redNo++;
  //Shift the count adjustment vector down one slot (to match new redNo)
  int i;
  
  if(hasParent()){
	int i;
	completedRedNo++;
  	for (i=1;i<adjVec.length();i++)
	   adjVec[i-1]=adjVec[i];
	adjVec.length()--;  
  }
  inProgress=CmiFalse;
  startRequested=CmiFalse;
  nRemote=nContrib=0;

  //Look through the future queue for messages we can now handle
  int n=futureMsgs.length();
  for (i=0;i<n;i++)
  {
    CkReductionMsg *m=futureMsgs.deq();
    if (m!=NULL) //One of these addContributions may have finished us.
      addContribution(m);//<- if *still* early, puts it back in the queue
  }
	
}




//////////// Reduction Manager Utilities /////////////
int CkReductionMgr::treeRoot(void)
{
  return 0;
}
CmiBool CkReductionMgr::hasParent(void) //Root PE
{
  return (CmiBool)(CkMyPe()!=treeRoot());
}
int CkReductionMgr::treeParent(void) //My parent PE
{
  return (CkMyPe()-1)/TREE_WID;
}
int CkReductionMgr::firstKid(void) //My first child PE
{
  return CkMyPe()*TREE_WID+1;
}
int CkReductionMgr::treeKids(void)//Number of children in tree
{
  int nKids=CkNumPes()-firstKid();
  if (nKids>TREE_WID) nKids=TREE_WID;
  if (nKids<0) nKids=0;
  return nKids;
}

//Return the countAdjustment struct for the given redNo:
countAdjustment &CkReductionMgr::adj(int number)
{
  number-=completedRedNo;
  number--;
  if (number<0) CkAbort("Requested adjustment to prior reduction!\n");
  //Pad the adjustment vector with zeros until it's at least number long
  while (adjVec.length()<=number)
    adjVec.push_back(countAdjustment());
  return adjVec[number];
}

//Combine (& free) the current message vector msgs.
CkReductionMsg *CkReductionMgr::reduceMessages(void)
{
  CkReductionMsg *ret=NULL;
  int nMsgs=msgs.length();

  //Look through the vector for a valid reducer, swapping out placeholder messages
  CkReduction::reducerType r=CkReduction::invalid;
  int msgs_gcount=0;//Reduced gcount
  int msgs_nSources=0;//Reduced nSources
  int msgs_userFlag=-1;
  CkCallback msgs_callback;
  int i;
  for (i=0;i<nMsgs;i++)
  {
    CkReductionMsg *m=msgs[i];
    msgs_gcount+=m->gcount;
    if (m->sourceFlag!=0)
    { //This is a real message from an element, not just a placeholder
      msgs_nSources+=m->nSources();
      r=m->reducer;
      if (!m->callback.isInvalid())
        msgs_callback=m->callback;
      if (m->userFlag!=-1)
        msgs_userFlag=m->userFlag;
    }
    else
    { //This is just a placeholder message-- replace it
      msgs[i--]=msgs[--nMsgs];
      delete m;
    }
  }

  if (nMsgs==0||r==CkReduction::invalid)
  //No valid reducer in the whole vector
    ret=CkReductionMsg::buildNew(0,NULL);
  else
  {//Use the reducer to reduce the messages
    CkReduction::reducerFn f=CkReduction::reducerTable[r];
    CkReductionMsg **msgArr=&msgs[0];//<-- HACK!
    ret=(*f)(nMsgs,msgArr);
    ret->reducer=r;
  }

  //Go back through the vector, deleting old messages
  for (i=0;i<nMsgs;i++) delete msgs[i];

  //Set the message counts
  ret->redNo=redNo;
  ret->gcount=msgs_gcount;
  ret->userFlag=msgs_userFlag;
  ret->callback=msgs_callback;
  ret->sourceFlag=msgs_nSources;
  DEBR((AA"Reduced gcount=%d; sourceFlag=%d\n"AB,ret->gcount,ret->sourceFlag));

  //Empty out the message vector
  msgs.length()=0;
  return ret;
}


//Checkpointing utilities
//pack-unpack method for CkReductionMsg
//if packing pack the message and then unpack and return it
//if unpacking allocate memory for it read it off disk and then unapck
//and return it
CkReductionMsg* CkReductionMgr::pupCkReductionMsg(CkReductionMsg *m, PUP::er &p)
{
  int len;
  envelope *env;

  if (p.isPacking()) {
  env = UsrToEnv(CkReductionMsg::pack(m));
    len = env->getTotalsize();
  }
  p(len);
  if (p.isUnpacking())
  env = (envelope *) CmiAlloc(len);
  p((void *)env, len);

  return CkReductionMsg::unpack(EnvToUsr(env));
}

//pack-unpack method for reduction message vector
void CkReductionMgr::pupMsgVector(CkVec<CkReductionMsg *> &_msgs, PUP::er &p)
{
  int nMsgs;
  CkReductionMsg *m;

  if (p.isPacking()) nMsgs = _msgs.length();
  p(nMsgs);

  for(int i = 0; i < nMsgs; i++) {
    m = p.isPacking() ? _msgs[i] : 0;
    _msgs.insert(i, pupCkReductionMsg(m, p));
  }
}

//pack-unpack method for reduction message Qs
void CkReductionMgr::pupMsgQ(CkQ<CkReductionMsg *> &_msgs, PUP::er &p)
{
  int nMsgs;
  CkReductionMsg *m;

  if (p.isPacking()) nMsgs = _msgs.length();
  p(nMsgs);

  for(int i = 0; i < nMsgs; i++) {
    m = p.isPacking() ? _msgs.deq() : 0;
    _msgs.enq(pupCkReductionMsg(m, p));
  }
}

//pack-unpack method for count adjustment vector
void CkReductionMgr::pupAdjVec(CkVec<countAdjustment> &vec, PUP::er &p)
{
  int nAdjs;

  if (p.isPacking()) nAdjs = vec.length();
  p(nAdjs);

  for(int i = 0; i < nAdjs; i++) {
  if (p.isUnpacking()) vec.push_back(countAdjustment());
  p(vec[i].gcount);
  p(vec[i].lcount);
  p(vec[i].mainRecvd);
  }
}

void CkReductionMgr::pup(PUP::er &p)
{
//We do not store the client function pointer or the client function parameter,
//it is the responsibility of the programmer to correctly restore these
  CkGroupInitCallback::pup(p);
  p(redNo);
  p(completedRedNo);
  p(inProgress); p(creating); p(startRequested);
  p(gcount); p(lcount);
  p(nContrib); p(nRemote);
  pupMsgVector(msgs, p);
  pupMsgQ(futureMsgs, p);
  pupMsgQ(futureRemoteMsgs, p);
  pupMsgQ(finalMsgs, p);
  pupAdjVec(adjVec, p);
}


//Callback for doing Reduction through NodeGroups added by Sayantan

void CkReductionMgr::ArrayReductionHandler(CkReductionMsg *m){

	int total = m->gcount+adj(m->redNo).gcount;
	finalMsgs.enq(m);
	//CkPrintf("ArrayReduction Handler Invoked for %d \n",m->redNo);
	adj(m->redNo).mainRecvd = 1;
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
  	int msgs_userFlag=-1;
  	CkCallback msgs_callback;
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
      				r=m->reducer;
      				if (!m->callback.isInvalid())
        			msgs_callback=m->callback;
      				if (m->userFlag!=-1)
        				msgs_userFlag=m->userFlag;
				tempMsgs.push_back(m);
    			}
		}else{
			finalMsgs.enq(m);
		}

	}
	numMsgs = tempMsgs.length();
	//CkPrintf("[%d]Total = %d %d Sources = %d Number of Messages %d\n",CkMyPe(),msgs_gcount,  adj(completedRedNo+1).gcount,msgs_nSources,numMsgs);
	if(numMsgs == 0){
		return;
	}
	if(adj(completedRedNo+1).mainRecvd == 0){
		for(i=0;i<numMsgs;i++){
			finalMsgs.enq(tempMsgs[i]);
		}
		return;
	}
	if(msgs_gcount  > msgs_nSources){
		for(i=0;i<numMsgs;i++){
			finalMsgs.enq(tempMsgs[i]);
		}
		return;
	}



	if (nMsgs==0||r==CkReduction::invalid)
  		//No valid reducer in the whole vector
    		ret=CkReductionMsg::buildNew(0,NULL);
  	else{//Use the reducer to reduce the messages
    		CkReduction::reducerFn f=CkReduction::reducerTable[r];
		// has to be corrected elements from above need to be put into a temporary vector
    		CkReductionMsg **msgArr=&tempMsgs[0];//<-- HACK!
    		ret=(*f)(numMsgs,msgArr);
    		ret->reducer=r;

  	}

	for(i = 0;i<numMsgs;i++){
		delete tempMsgs[i];
	}



	//CkPrintf("Length of finalMsgs after endReduction %d \n",finalMsgs.length());
	//CkPrintf("Data size of result = %d Length of finalMsg %d \n",ret->getLength(),finalMsgs.length());

	ret->redNo=completedRedNo+1;
  	ret->gcount=msgs_gcount;
  	ret->userFlag=msgs_userFlag;
  	ret->callback=msgs_callback;
  	ret->sourceFlag=msgs_nSources;
	
	secondaryStoredCallback = callbackQ.deq();
	if (!secondaryStoredCallback->isInvalid())
	    secondaryStoredCallback->send(ret);
    else if (storedCallback!=NULL)
	    storedCallback->send(ret);
    else
	    CkAbort("No reduction client!\n"
		    "You must register a client with either SetReductionClient or during contribute.\n");
	completedRedNo++;
//        CkPrintf("[%d,%d]------------END OF ARRAY REDUCTION %d at %.6f\n",CkMyNode(),CkMyPe(),completedRedNo,CkWallTimer());
	for (i=1;i<adjVec.length();i++)
    		adjVec[i-1]=adjVec[i];
	adjVec.length()--;
}





/////////////////////////////////////////////////////////////////////////

////////////////////////////////
//CkReductionMsg support

//ReductionMessage default private constructor-- does nothing
CkReductionMsg::CkReductionMsg(){}

//This define gives the distance from the start of the CkReductionMsg
// object to the start of the user data area (just below last object field)
#define ARM_DATASTART (sizeof(CkReductionMsg)-sizeof(double))

//"Constructor"-- builds and returns a new CkReductionMsg.
//  the "data" array you specify will be copied into this object.
CkReductionMsg *CkReductionMsg::
  buildNew(int NdataSize,const void *srcData,
    CkReduction::reducerType reducer)
{
  int len[1];
  len[0]=NdataSize;
  CkReductionMsg *ret = new(len,0)CkReductionMsg();

  ret->dataSize=NdataSize;
  if (srcData!=NULL)
    memcpy(ret->data,srcData,NdataSize);
  ret->userFlag=-1;
  ret->reducer=reducer;
  ret->ci=NULL;
  ret->sourceFlag=-1000;
  ret->gcount=0;
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
  in->data = NULL;
  return (void*) in;
}

CkReductionMsg* CkReductionMsg::unpack(void *in)
{
  CkReductionMsg *ret = (CkReductionMsg *)in;
  DEBR(("CkReductionMsg::unpack %d %d %d %d\n",
    ret->sourceFlag,ret->redNo,ret->gcount,ret->dataSize));
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
      RED_DEB(("|\tmsg%d (from %d) [%d]="typeStr"\n",m,msg[m]->sourceFlag,i,value[i]));\
      loop\
    }\
  }\
  RED_DEB(("\\ PE_%d: " #name " finished\n",CkMyPe()));\
  return CkReductionMsg::buildNew(nElem*sizeof(dataType),(void *)ret);\
}

//Use this macro for reductions that have the same type for all inputs
#define SIMPLE_POLYMORPH_REDUCTION(nameBase,loop) \
  SIMPLE_REDUCTION(nameBase##_int,int,"%d",loop) \
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

//Compute the logical OR of the integers passed by each element.
// The resulting integer will be 1 if any source integer is nonzero; else 0.
SIMPLE_REDUCTION(logical_or,int,"%d",
  if (value[i]!=0)
           ret[i]=1;
  ret[i]=!!ret[i];//Make sure ret[i] is 0 or 1
)

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
                        RED_DEB(("|\tmsg[%d] is %d bytes\n",i,msg[i]->getSize()));
                        memcpy((void *)cur,(void *)msg[i]->getData(),messageBytes);
                        cur=(CkReduction::setElement *)(((char *)cur)+messageBytes);
    }
    else //This is a message from an element-- wrap it in a reduction_set_element
    {
      RED_DEB(("|\tmsg[%d] is %d bytes\n",i,msg[i]->getSize()));
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

/////////////////// Reduction Function Table /////////////////////
CkReduction::CkReduction() {} //Dummy private constructor

//Add the given reducer to the list.  Returns the new reducer's
// reducerType.  Must be called in the same order on every node.
CkReduction::reducerType CkReduction::addReducer(reducerFn fn)
{
  reducerTable[nReducers]=fn;
  return (reducerType)nReducers++;
}

/*Reducer table: maps reducerTypes to reducerFns.
It's indexed by reducerType, so the order in this table
must *exactly* match the reducerType enum declaration.
The names don't have to match, but it helps.
*/
int CkReduction::nReducers=CkReduction::lastSystemReducer;

CkReduction::reducerFn CkReduction::reducerTable[CkReduction::MAXREDUCERS]={
    ::invalid_reducer,
  //Compute the sum the numbers passed by each element.
    ::sum_int,::sum_float,::sum_double,

  //Compute the product the numbers passed by each element.
    ::product_int,::product_float,::product_double,

  //Compute the largest number passed by any element.
    ::max_int,::max_float,::max_double,

  //Compute the smallest number passed by any element.
    ::min_int,::min_float,::min_double,

  //Compute the logical AND of the integers passed by each element.
  // The resulting integer will be zero if any source integer is zero.
    ::logical_and,

  //Compute the logical OR of the integers passed by each element.
  // The resulting integer will be 1 if any source integer is nonzero.
    ::logical_or,

  //Concatenate the (arbitrary) data passed by each element
    ::concat,

  //Combine the data passed by each element into an list of setElements.
  // Each element may contribute arbitrary data (with arbitrary length).
    ::set
};








/********** Code added by Sayantan *********************/


/**nodegroup reduction manager . Most of it is similar to the guy above***/
NodeGroup::NodeGroup(void) {
  __nodelock=CmiCreateLock();


}
NodeGroup::~NodeGroup() {
  CmiDestroyLock(__nodelock);
}
void NodeGroup::pup(PUP::er &p)
{
  IrrGroup::pup(p);
}

//CK_REDUCTION_CLIENT_DEF(CProxy_NodeGroup,(CkNodeReductionMgr *)CkLocalBranch(_ck_gid));

void CProxy_NodeGroup::ckSetReductionClient(CkCallback *cb) const {
  DEBR(("in CksetReductionClient for CProxy_NodeGroup %d\n",CkLocalNodeBranch(_ck_gid)));
 ((CkNodeReductionMgr *)CkLocalNodeBranch(_ck_gid))->ckSetReductionClient(cb);
 }

CK_REDUCTION_CONTRIBUTE_METHODS_DEF(NodeGroup,
				    ((CkNodeReductionMgr *)this),
				    reductionInfo);

/* this contribute also adds up the count across all messages it receives.
  Useful for summing up number of array elements who have contributed ****/ 
void NodeGroup::contributeWithCounter(CkReductionMsg *msg,int count)
	{((CkNodeReductionMgr *)this)->contributeWithCounter(&reductionInfo,msg,count);}



//#define BINOMIAL_TREE

CkNodeReductionMgr::CkNodeReductionMgr()//Constructor
  : thisProxy(thisgroup)
{
 if(CkMyRank() == 0){
#ifdef BINOMIAL_TREE
  init_BinomialTree();
#endif
  storedCallback=NULL;
  redNo=0;
  inProgress=CmiFalse;
  
  startRequested=CmiFalse;
  gcount=CkNumNodes();
  lcount=1;
  nContrib=nRemote=0;
  lockEverything = CmiCreateLock();


  creating=CmiFalse;
  interrupt = 0;
  DEBR((AA"In NodereductionMgr constructor at %d \n"AB,this));
  }
}

//////////// Reduction Manager Client API /////////////

//Add the given client function.  Overwrites any previous client.
void CkNodeReductionMgr::ckSetReductionClient(CkCallback *cb)
{
  DEBR((AA"Setting reductionClient in NodeReductionMgr %d at %d\n"AB,cb,this));
  if(cb->isInvalid()){
  	DEBR((AA"Invalid Callback passed to setReductionClient in nodeReductionMgr\n"AB));
  }else{
	DEBR((AA"Valid Callback passed to setReductionClient in nodeReductionMgr\n"AB));
  }

  if (CkMyNode()!=0)
	  CkError("WARNING: ckSetReductionClient should only be called from processor zero!\n");
  delete storedCallback;
  storedCallback=cb;
}



void CkNodeReductionMgr::contribute(contributorInfo *ci,CkReductionMsg *m)
{

  m->ci=ci;
  m->redNo=ci->redNo++;
  m->sourceFlag=-1;//A single contribution
  m->gcount=0;

  addContribution(m);

}


void CkNodeReductionMgr::contributeWithCounter(contributorInfo *ci,CkReductionMsg *m,int count)
{


  m->ci=ci;
  m->redNo=ci->redNo++;

  m->gcount=count;

  addContribution(m);

}


//////////// Reduction Manager Remote Entry Points /////////////

//Sent down the reduction tree (used by barren PEs)
void CkNodeReductionMgr::ReductionStarting(CkReductionNumberMsg *m)
{
  
  if (isPresent(m->num) && !inProgress)
  {
    DEBR((AA"Starting reduction #%d at parent's request\n"AB,m->num));
    startReduction(m->num);
    finishReduction();
  } else if (isFuture(m->num)){
  	//CkPrintf("[%d][%d] Message num %d Present redNo %d \n",CkMyNode(),CkMyPe(),m->num,redNo);
    	CkAbort("My reduction tree parent somehow got ahead of me! in nodegroups\n");
    }
  else //is Past
    DEBR((AA"Ignoring parent's late request to start #%d\n"AB,m->num));
  delete m;
 
}

//Sent up the reduction tree with reduced data
void CkNodeReductionMgr::RecvMsg(CkReductionMsg *m)
{

#ifndef CMK_CPV_IS_SMP
#if CMK_IMMEDIATE_MSG
	if(interrupt == 1){
		CmiDelayImmediate();
		return;
	}
#endif	
#endif
	 
  if (isPresent(m->redNo)) { //Is a regular, in-order reduction message
    DEBR((AA"Recv'd remote contribution %d for #%d at %d\n"AB,nRemote,m->redNo,this));
 //  CkPrintf("[%d,%d] Recv'd REMOTE contribution for %d at %.6f\n",CkMyNode(),CkMyPe(),m->redNo,CkWallTimer());

    
    
    startReduction(m->redNo);
    
    
    
    interrupt = 1;	    
    CmiLock(lockEverything);
    msgs.push_back(m);        
    nRemote++;
    CmiUnlock(lockEverything);
    interrupt = 0;

    
    
    finishReduction();
    

  }
  else if (isFuture(m->redNo)) {
    DEBR((AA"Recv'd early remote contribution %d for #%d\n"AB,nRemote,m->redNo));
    
    interrupt = 1;
    CmiLock(lockEverything);
    futureRemoteMsgs.enq(m);
    CmiUnlock(lockEverything);
    interrupt = 0;
  }
  else{
    //CkPrintf("BIG Problem Present %d Mesg RedNo %d \n",redNo,m->redNo);
   CkAbort("Recv'd late remote contribution!\n");
   }

}

void CkNodeReductionMgr::startReduction(int number)
{
	  if (isFuture(number)) CkAbort("Can't start reductions out of order!\n");
	  if (isPast(number)) CkAbort("Can't restart reduction that's already finished!\n");
	  if (inProgress){
  		DEBR((AA"This reduction is already in progress\n"AB));		
	  	return;//This reduction already started
	  }
	  if (creating) //Don't start yet-- we're creating elements
	  {
	    DEBR((AA"Postponing start request #%d until we're done creating\n"AB,redNo));
	    
	    interrupt = 1;
	    CmiLock(lockEverything);
	    startRequested=CmiTrue;
	    CmiUnlock(lockEverything);
	    interrupt = 0;
	    return;
	  }
	
	//If none of these cases, we need to start the reduction--
	  DEBR((AA"Starting reduction #%d\n"AB,redNo));
	  inProgress=CmiTrue;
	  //Sent start requests to our kids (in case they don't already know)
	 
	  
	
	  for (int k=0;k<treeKids();k++)
	  {
#ifdef BINOMIAL_TREE
		DEBR((AA"Asking child PE %d to start #%d\n"AB,kids[k],redNo));
		thisProxy[kids[k]].ReductionStarting(new CkReductionNumberMsg(redNo));
#else
		DEBR((AA"Asking child PE %d to start #%d\n"AB,firstKid()+k,redNo));
		thisProxy[firstKid()+k].ReductionStarting(new CkReductionNumberMsg(redNo));
#endif
	  }
	  	    	    
}

//Handle a message from one element for the reduction
void CkNodeReductionMgr::addContribution(CkReductionMsg *m)
{
   
   if (isFuture(m->redNo)) {//An early contribution-- add to future Q
    DEBR((AA"Contributor %p gives early contribution-- for #%d\n"AB,m->ci,m->redNo));
    interrupt = 1;
    CmiLock(lockEverything);
    futureMsgs.enq(m);
    CmiUnlock(lockEverything);
    interrupt = 0;
  } else {// An ordinary contribution
    DEBR((AA"Recv'd local contribution %d for #%d at %d\n"AB,nContrib,m->redNo,this));
//    CmiPrintf("[%d,%d] Redcv'd Local Contribution for redNo %d number %d at %0.6f \n",CkMyNode(),CkMyPe(),m->redNo,nContrib+1,CkWallTimer());
    startReduction(m->redNo);
    interrupt = 1;
    CmiLock(lockEverything);
    msgs.push_back(m);
    nContrib++;
    CmiUnlock(lockEverything);
    interrupt = 0;
    finishReduction();
  }

}
/** check if the nodegroup reduction is finished at this node. In that case send it 
up the reduction tree **/


void CkNodeReductionMgr::finishReduction(void)
{
  
  DEBR((AA"in finishReduction %d \n"AB,inProgress));
  /***Check if reduction is finished in the next few ifs***/
  if ((!inProgress) | creating){
  	DEBR((AA"Either not in Progress or creating\n"AB));	
  	return;
  }
  interrupt = 1;
  CmiLock(lockEverything);
  if (nContrib<(lcount)){
	DEBR((AA"Need more local messages %d %d\n"AB,nContrib,(lcount)));
	CmiUnlock(lockEverything);
	interrupt = 0;
	 return;//Need more local messages
  }    
  if (nRemote<treeKids()){
	DEBR((AA"Need more Remote messages %d %d\n"AB,nRemote,treeKids()));
	CmiUnlock(lockEverything);
	interrupt = 0;
	return;//Need more remote messages
  }
  if (nRemote>treeKids()){
	  CmiUnlock(lockEverything);
	  interrupt = 0;
	   CkAbort("Excess remote reduction message received!\n");
  }

  
  DEBR((AA"Reducing data...\n"AB));

  /**reduce all messages received at this node **/
  CkReductionMsg *result=reduceMessages();
 
  if (hasParent())
  {//Pass data up tree to parent
    DEBR((AA"Passing reduced data up to parent node %d. \n"AB,treeParent()));
    //CkPrintf("[%d,%d] Passing data up to parentNode %d at %.6f \n",CkMyNode(),CkMyPe(),treeParent(),CkWallTimer());
    thisProxy[treeParent()].RecvMsg(result);

  }
  else
  {
	  /** if the reduction is finished and I am the root of the reduction tree 
	  then call the reductionhandler and other stuff ***/
    DEBR((AA"Passing result to client function\n"AB));
 //   CkPrintf("[%d,%d]------------------- END OF REDUCTION %d with %d remote contributions passed to client function at %.6f\n",CkMyNode(),CkMyPe(),redNo,nRemote,CkWallTimer());
    if (!result->callback.isInvalid())
	    result->callback.send(result);
    else if (storedCallback!=NULL)
	    storedCallback->send(result);
    else{
    		DEBR((AA"Invalid Callback at %d %d\n"AB,result->callback,storedCallback));
	    CkAbort("No reduction client!\n"
		    "You must register a client with either SetReductionClient or during contribute.\n");
	}
  }

  DEBR((AA"Reduction %d finished!\n"AB,redNo));
  //CkPrintf("[%d,%d]Reduction %d finished with %d\n",CkMyNode(),CkMyPe(),redNo,nContrib);

  redNo++;

  int i;

  inProgress=CmiFalse;
  startRequested=CmiFalse;
  nRemote=nContrib=0;
  

  CmiUnlock(lockEverything);
  interrupt = 0;
  //Look through the future queue for messages we can now handle
  
  interrupt = 1;
  CmiLock(lockEverything);
  int n=futureMsgs.length(); 
  CmiUnlock(lockEverything);
  interrupt = 0;
  
  for (i=0;i<n;i++)
  {
    interrupt = 1;	  
    CmiLock(lockEverything);
    CkReductionMsg *m=futureMsgs.deq();
    CmiUnlock(lockEverything);
    interrupt = 0;
    if (m!=NULL) //One of these addContributions may have finished us.
      addContribution(m);//<- if *still* early, puts it back in the queue
  }
  
  interrupt = 1;
  CmiLock(lockEverything);
  n=futureRemoteMsgs.length();
  CmiUnlock(lockEverything);
  interrupt = 0;
  for (i=0;i<n;i++)
  {
    interrupt = 1;	  
    CmiLock(lockEverything);
    CkReductionMsg *m=futureRemoteMsgs.deq();
    CmiUnlock(lockEverything);
    interrupt = 0;
    if (m!=NULL)
      RecvMsg(m);//<- if *still* early, puts it back in the queue
  }

  
}


//////////// Reduction Manager Utilities /////////////



void CkNodeReductionMgr::init_BinomialTree(){
	int depth = (int )ceil((log(CkNumNodes())/log(2)));
	upperSize = (unsigned )pow(2,depth);
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
	parent = label + rint(pow(2,count));
	parent = upperSize -1 -parent;
	int temp;
	if(count != 0){
		kids = new int[count];
		numKids = 0;
		for(int i=0;i<count;i++){
			temp = label - rint(pow(2,i));
			temp = upperSize-1-temp;
			if(temp <= CkNumNodes()-1){
				kids[numKids] = temp;
				numKids++;
			}
		}
	}else{
		numKids = 0;
		kids = NULL;
	}
}



int CkNodeReductionMgr::treeRoot(void)
{
  return 0;
}
CmiBool CkNodeReductionMgr::hasParent(void) //Root PE
{
  return (CmiBool)(CkMyNode()!=treeRoot());
}
int CkNodeReductionMgr::treeParent(void) //My parent PE
{
#ifdef BINOMIAL_TREE
	return parent;
#else
  return (CkMyNode()-1)/TREE_WID;
#endif
}

int CkNodeReductionMgr::firstKid(void) //My first child PE
{
  return CkMyNode()*TREE_WID+1;
}
int CkNodeReductionMgr::treeKids(void)//Number of children in tree
{
#ifdef BINOMIAL_TREE
	return numKids;
#else
  int nKids=CkNumNodes()-firstKid();
  if (nKids>TREE_WID) nKids=TREE_WID;
  if (nKids<0) nKids=0;
  return nKids;
#endif
}



//Combine (& free) the current message vector msgs.
CkReductionMsg *CkNodeReductionMgr::reduceMessages(void)
{

  CkReductionMsg *ret=NULL;
  int nMsgs=msgs.length();

  //Look through the vector for a valid reducer, swapping out placeholder messages
  CkReduction::reducerType r=CkReduction::invalid;
  int msgs_gcount=0;//Reduced gcount
  int msgs_nSources=0;//Reduced nSources
  int msgs_userFlag=-1;
  CkCallback msgs_callback;
  int i;
  for (i=0;i<nMsgs;i++)
  {
    CkReductionMsg *m=msgs[i];
    msgs_gcount+=m->gcount;
    if (m->sourceFlag!=0)
    { //This is a real message from an element, not just a placeholder
      msgs_nSources+=m->nSources();
      r=m->reducer;
      if (!m->callback.isInvalid())
        msgs_callback=m->callback;
      if (m->userFlag!=-1)
        msgs_userFlag=m->userFlag;
    }
    else
    { //This is just a placeholder message-- replace it
      msgs[i--]=msgs[--nMsgs];
      delete m;
    }
  }

  if (nMsgs==0||r==CkReduction::invalid)
  //No valid reducer in the whole vector
    ret=CkReductionMsg::buildNew(0,NULL);
  else
  {//Use the reducer to reduce the messages
    CkReduction::reducerFn f=CkReduction::reducerTable[r];
    CkReductionMsg **msgArr=&msgs[0];//<-- HACK!
    ret=(*f)(nMsgs,msgArr);
    ret->reducer=r;
  }

  //Go back through the vector, deleting old messages
  for (i=0;i<nMsgs;i++) delete msgs[i];

  //Set the message counts
  ret->redNo=redNo;
  ret->gcount=msgs_gcount;
  ret->userFlag=msgs_userFlag;
  ret->callback=msgs_callback;
  ret->sourceFlag=msgs_nSources;
  DEBR((AA"Reduced gcount=%d; sourceFlag=%d\n"AB,ret->gcount,ret->sourceFlag));

  //Empty out the message vector
  msgs.length()=0;

  return ret;
}


//Checkpointing utilities
//pack-unpack method for CkReductionMsg
//if packing pack the message and then unpack and return it
//if unpacking allocate memory for it read it off disk and then unapck
//and return it
CkReductionMsg* CkNodeReductionMgr::pupCkReductionMsg(CkReductionMsg *m, PUP::er &p)
{
  int len;
  envelope *env;

  if (p.isPacking()) {
  env = UsrToEnv(CkReductionMsg::pack(m));
    len = env->getTotalsize();
  }
  p(len);
  if (p.isUnpacking())
  env = (envelope *) CmiAlloc(len);
  p((void *)env, len);

  return CkReductionMsg::unpack(EnvToUsr(env));
}

//pack-unpack method for reduction message vector
void CkNodeReductionMgr::pupMsgVector(CkVec<CkReductionMsg *> &_msgs, PUP::er &p)
{
  int nMsgs;
  CkReductionMsg *m;

  if (p.isPacking()) nMsgs = _msgs.length();
  p(nMsgs);

  for(int i = 0; i < nMsgs; i++) {
    m = p.isPacking() ? _msgs[i] : 0;
    _msgs.insert(i, pupCkReductionMsg(m, p));
  }
}

//pack-unpack method for reduction message Qs
void CkNodeReductionMgr::pupMsgQ(CkQ<CkReductionMsg *> &_msgs, PUP::er &p)
{
  int nMsgs;
  CkReductionMsg *m;

  if (p.isPacking()) nMsgs = _msgs.length();
  p(nMsgs);

  for(int i = 0; i < nMsgs; i++) {
    m = p.isPacking() ? _msgs.deq() : 0;
    _msgs.enq(pupCkReductionMsg(m, p));
  }
}


void CkNodeReductionMgr::pup(PUP::er &p)
{
//We do not store the client function pointer or the client function parameter,
//it is the responsibility of the programmer to correctly restore these
  IrrGroup::pup(p);
  p(redNo);
  p(inProgress); p(creating); p(startRequested);
  p(gcount); p(lcount);
  p(nContrib); p(nRemote);
  p(interrupt);
  pupMsgVector(msgs, p);
  pupMsgQ(futureMsgs, p);
  pupMsgQ(futureRemoteMsgs, p);
 // pupAdjVec(adjVec, p);
}



#include "CkReduction.def.h"
