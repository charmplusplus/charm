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
serially.
  -CkReductionMsg, the message carrying reduction data
used by the reduction manager.


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
#define AA "Red PE%d #%d (%d,%d)> "
#define AB ,CkMyPe(),redNo,nRemote,nContrib

// For status and data messages from the builtin reducer functions.
#define RED_DEB(x) CkPrintf x

#else
//No debugging info-- empty defines
#define DEBR(x) //CkPrintf x
#define RED_DEB(x) //CkPrintf x
#endif



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
  : thisproxy(thisgroup)
{
  storedClient=NULL;
  storedClientParam=NULL;
  redNo=0;
  inProgress=CmiFalse;
  creating=CmiFalse;
  startRequested=CmiFalse;
  gcount=lcount=0;
  nContrib=nRemote=0;
  DEBR((AA"In reductionMgr constructor\n"AB));
}

//////////// Reduction Manager Client API /////////////

//Add the given client function.  Overwrites any previous client.
void CkReductionMgr::setClient(clientFn client,void *param)
{
  storedClient=client;
  storedClientParam=param;
}

///////////////////////////// Contributor ////////////////////////
//Contributors keep a copy of this structure:

/*Contributor migration support:
*/
void CkReductionMgr::contributorInfo::pup(PUP::er &p)
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

//Initializes a new contributor
void CkReductionMgr::contributorCreated(contributorInfo *ci)
{
  DEBR((AA"Contributor %p created\n"AB,ci));
  //We've got another contributor
  gcount++;
  lcount++;
  if (inProgress) 
  {
    ci->redNo=redNo+1;//Created *during* reduction => contribute to *next* reduction
    adj(redNo).gcount--;//He'll wrongly be counted in the global count at end
    adj(redNo).lcount--;//He won't contribute to the current reduction
  } else
    ci->redNo=redNo;//Created *before* reduction => contribute to *that* reduction
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
      thisproxy[treeRoot()].MigrantDied(new CkReductionNumberMsg(r));
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
}

//////////// Reduction Manager Remote Entry Points /////////////

//Sent down the reduction tree (used by barren PEs)
void CkReductionMgr::ReductionStarting(CkReductionNumberMsg *m)
{
  if (isPresent(m->num) && !inProgress)
  {
    DEBR((AA"Starting reduction #%d at parent's request\n"AB,m->num));
    startReduction(m->num);
    finishReduction();
  } else if (isFuture(m->num)) 
    CkAbort("My reduction tree parent somehow got ahead of me!\n");
  else //is Past
    DEBR((AA"Ignoring parent's late request to start #%d\n"AB,m->num));
  delete m;
}
//Sent to root of reduction tree with reduction contribution
// of migrants that missed the main reduction.
void CkReductionMgr::LateMigrantMsg(CkReductionMsg *m)
{
  addContribution(m);
}

//A late migrating contributor will never contribute to this reduction
void CkReductionMgr::MigrantDied(CkReductionNumberMsg *m)
{
  if (hasParent() || isPast(m->num)) CkAbort("Late MigrantDied message recv'd!\n");
  DEBR((AA"Migrant died before contributing to #%d\n"AB,m->num));
  adj(m->num).gcount--;//He won't be contributing to this one.
  finishReduction();
}
//Sent up the reduction tree with reduced data
void CkReductionMgr::RecvMsg(CkReductionMsg *m)
{
  if (isPresent(m->redNo)) { //Is a regular, in-order reduction message
    DEBR((AA"Recv'd remote contribution %d for #%d\n"AB,nRemote,m->redNo));
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
  if (isFuture(number)) CkAbort("Can't start reductions out of order!\n");
  if (isPast(number)) CkAbort("Can't restart reduction that's already finished!\n");
  if (inProgress) return;//This reduction already started
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
  for (int k=0;k<treeKids();k++)
  {
    DEBR((AA"Asking child PE %d to start #%d\n"AB,firstKid()+k,redNo));
    thisproxy[firstKid()+k].ReductionStarting(new CkReductionNumberMsg(redNo));
  }
}
//Handle a message from one element for the reduction
void CkReductionMgr::addContribution(CkReductionMsg *m)
{
  if (isPast(m->redNo))
  {//We've moved on-- forward late contribution straight to root
    DEBR((AA"Migrant %p gives late contribution for #%d!\n"AB,m->ci,m->redNo));
    if (!hasParent()) //Root moved on too soon-- should never happen
      CkAbort("Late reduction contribution received at root!\n");
    thisproxy[treeRoot()].LateMigrantMsg(m);
  } 
  else if (isFuture(m->redNo)) {//An early contribution-- add to future Q
    DEBR((AA"Contributor %p gives early contribution-- for #%d\n"AB,m->ci,m->redNo));
    futureMsgs.enq(m);
  } else {// An ordinary contribution  
    DEBR((AA"Recv'd local contribution %d for #%d\n"AB,nContrib,m->redNo));
    startReduction(m->redNo);
    msgs.push_back(m);
    nContrib++;
    finishReduction();
  }
}
void CkReductionMgr::finishReduction(void) 
{
  if ((!inProgress) | creating) return;
  if (nContrib<(lcount+adj(redNo).lcount)) return;//Need more local messages
  if (nRemote<treeKids()) return;//Need more remote messages
  if (nRemote>treeKids()) CkAbort("Excess remote reduction message received!\n");
  
  DEBR((AA"Reducing data...\n"AB));
  CkReductionMsg *result=reduceMessages();

  if (hasParent())
  {//Pass data up tree to parent
    DEBR((AA"Passing reduced data up to parent node %d.\n"AB,treeParent()));
    DEBR((AA"Message gcount is %d+%d+%d.\n"AB,result->gcount,gcount,adj(redNo).gcount));
    result->gcount+=gcount+adj(redNo).gcount;
    thisproxy[treeParent()].RecvMsg(result);
  }
  else 
  {//We are root-- pass data to client
    DEBR((AA"Final gcount is %d+%d+%d.\n"AB,result->gcount,gcount,adj(redNo).gcount));
    int totalElements=result->gcount+gcount+adj(redNo).gcount;
    if (totalElements>result->nSources()) 
    {
      DEBR((AA"Only got %d of %d contributions (c'mon, migrators!)\n"AB,result->nSources(),totalElements));
      msgs.push_back(result);
      return; // Wait for migrants to contribute
    } else if (totalElements<result->nSources()) {
      DEBR((AA"Got %d of %d contributions\n"AB,result->nSources(),totalElements));
      CkAbort("ERROR! Too many contributions at root!\n");
    }
    DEBR((AA"Passing result to client function\n"AB));
    if (storedClient==NULL) 
      CkAbort("No reduction client function installed!\n");
    (*storedClient)(storedClientParam,result->dataSize,result->data);
    delete result;
  }
  
  DEBR((AA"Reduction %d finished!\n"AB,redNo));
  redNo++;
  //Shift the count adjustment vector down one slot (to match new redNo)
  int i;
  for (i=1;i<adjVec.length();i++)
    adjVec[i-1]=adjVec[i];
  adjVec.length()--;
  inProgress=CmiFalse;
  startRequested=CmiFalse;
  nRemote=nContrib=0;

  //Look through the future queue for messages we can now handle
  int n=futureMsgs.length();
  for (i=0;i<n;i++)
  {
    CkReductionMsg *m=futureMsgs.deq();
    addContribution(m);//<- if *still* early, puts it back in the queue
  }
  n=futureRemoteMsgs.length();
  for (i=0;i<n;i++)
  {
    CkReductionMsg *m=futureRemoteMsgs.deq();
    RecvMsg(m);//<- if *still* early, puts it back in the queue
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
CkReductionMgr::countAdjustment &CkReductionMgr::adj(int number)
{
  number-=redNo;
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
  int i;
  for (i=0;i<nMsgs;i++)
  {
    CkReductionMsg *m=msgs[i];
    msgs_gcount+=m->gcount;
    msgs_nSources+=m->nSources();
    if (m->reducer!=CkReduction::invalid)
      r=m->reducer;
    if (m->sourceFlag==0)
    {//Replace this placeholder message
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
void CkReductionMgr::pupAdjVec(CkVec<CkReductionMgr::countAdjustment> &vec, PUP::er &p)
{
  int nAdjs;

  if (p.isPacking()) nAdjs = vec.length();
  p(nAdjs);

  for(int i = 0; i < nAdjs; i++) {
  if (p.isUnpacking()) vec.push_back(countAdjustment());
  p(vec[i].gcount);
  p(vec[i].lcount);
  }
}

void CkReductionMgr::pup(PUP::er &p)
{
//We do not store the client function pointer or the client function parameter,
//it is the responsibility of the programmer to correctly restore these
  CkGroupInitCallback::pup(p);
  p(redNo);
  p(inProgress); p(creating); p(startRequested);
  p(gcount); p(lcount);
  p(nContrib); p(nRemote);
  pupMsgVector(msgs, p);
  pupMsgQ(futureMsgs, p);
  pupMsgQ(futureRemoteMsgs, p);
  pupAdjVec(adjVec, p);
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
  buildNew(int NdataSize,void *srcData,
    CkReduction::reducerType reducer)
{
  int len[1];
  len[0]=NdataSize;
  CkReductionMsg *ret = new(len,0)CkReductionMsg();
  
  ret->dataSize=NdataSize;
  if (srcData!=NULL)
    memcpy(ret->data,srcData,NdataSize);
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
    ret+=*(int *)(msg[i]->data);
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
{CkAbort("ERROR! Called the invalid reducer!\n");return NULL;}

#define SIMPLE_REDUCTION(name,dataType,typeStr,loop) \
static CkReductionMsg *name(int nMsg,CkReductionMsg **msg)\
{\
  RED_DEB(("/ PE_%d: " #name " invoked on %d messages\n",CkMyPe(),nMsg));\
  int m,i;\
  int nElem=msg[0]->dataSize/sizeof(dataType);\
  dataType *ret=(dataType *)(msg[0]->data);\
  for (m=1;m<nMsg;m++)\
  {\
    dataType *value=(dataType *)(msg[m]->data);\
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
      retSize+=msg[i]->dataSize;

  RED_DEB(("|- concat'd reduction message will be %d bytes\n",retSize));
  
  //Allocate a new message
  CkReductionMsg *ret=CkReductionMsg::buildNew(retSize,NULL);
  
  //Copy the source message data into the return message
  char *cur=(char *)(ret->data);
  for (i=0;i<nMsg;i++) {
    int messageBytes=msg[i]->dataSize;
    memcpy((void *)cur,(void *)msg[i]->data,messageBytes);
    cur+=messageBytes;
  }
  RED_DEB(("\\ PE_%d: reduction_concat finished-- %d messages combined\n",CkMyPe(),nMsg));
  return ret;
}

/////////////// set ////////////////
/*
This reducer appends the data it recieves from each element
along with some housekeeping data indicating the source element 
and data size.
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
  for (i=0;i<nMsg;i++)
    if (msg[i]->sourceFlag>0)
    //This message is composite-- it will just be copied over (less terminating -1)
      retSize+=(msg[i]->dataSize-sizeof(int));
    else //This is a message from an element-- it will be wrapped in a reduction_set_element
      retSize+=SET_SIZE(msg[i]->dataSize);
  retSize+=sizeof(int);//Leave room for terminating -1.

  RED_DEB(("|- composite set reduction message will be %d bytes\n",retSize));
  
  //Allocate a new message
  CkReductionMsg *ret=CkReductionMsg::buildNew(retSize,NULL);
  
  //Copy the source message data into the return message
  CkReduction::setElement *cur=(CkReduction::setElement *)(ret->data);
  for (i=0;i<nMsg;i++)
    if (msg[i]->sourceFlag>0)
    {//This message is composite-- just copy it over (less terminating -1)
                        int messageBytes=msg[i]->dataSize-sizeof(int);
                        RED_DEB(("|\tmsg[%d] is %d bytes from %d sources\n",i,msg[i]->dataSize,msg[i]->nSources()));
                        memcpy((void *)cur,(void *)msg[i]->data,messageBytes);
                        cur=(CkReduction::setElement *)(((char *)cur)+messageBytes);
    }
    else //This is a message from an element-- wrap it in a reduction_set_element
    {
      RED_DEB(("|\tmsg[%d] is %d bytes\n",i,msg[i]->dataSize));
      cur->dataSize=msg[i]->dataSize;
      memcpy((void *)cur->data,(void *)msg[i]->data,msg[i]->dataSize);
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
int CkReduction::nReducers=16;//Number of reducers currently in table below

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

#include "CkReduction.def.h"
