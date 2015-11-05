#include "charm++.h"
#include "ck.h"
#include "CkArrayReductionMgr.decl.h"
#define ARRREDDEBUG 0

void noopitar(const char*, ...)
{}

#if ARRREDDEBUG
#define ARPRINT CkPrintf
#else
#define ARPRINT noopitar
#endif

void CkArrayReductionMgr::init()
{
	//ARPRINT("Array ReductionMgr Constructor called %d\n",thisgroup);
	redNo=0;
	size = CkMyNodeSize();
	count = 0;
	lockCount = CmiCreateLock();
	ctorDoneFlag = 1;
	alreadyStarted = -1;
}

CkArrayReductionMgr::CkArrayReductionMgr(){
	init();
	attachedGroup.setZero();
}

CkArrayReductionMgr::CkArrayReductionMgr(int dummy, CkGroupID gid){
	init();
	attachedGroup = gid;
}

void CkArrayReductionMgr::flushStates(){
  if(CkMyRank()== 0){
    // CmiPrintf("[%d] CkArrayReductionMgr::flushState\n", CkMyPe());
    redNo=0;
    count = 0;
    while (!my_msgs.isEmpty())  delete my_msgs.deq();
    while (!my_futureMsgs.isEmpty()) delete my_futureMsgs.deq();
    reductionInfo.redNo = 0;
    CkNodeReductionMgr::flushStates();
  }
}

void CkArrayReductionMgr::collectAllMessages(){
  // if the queue isn't full, but there is at least one message and the
  // reduction is streamable, do a partial reduction
  bool partialReduction = count < size && my_msgs.length() > 0 &&             \
    CkReduction::reducerTable[my_msgs.peek()->reducer].streamable;

	if(count == size || partialReduction) {
		ARPRINT("[%d] CollectAll messages  for %d with %d on %p\n",CkMyNode(),redNo,count,this);
		CkReductionMsg *result = reduceMessages();
                if (partialReduction) {
                  my_msgs.enq(result);
                  return;
                }
		result->redNo = redNo;
		/**keep a count of elements that contributed to the original reduction***/
		contributeWithCounter(result,result->gcount);
		count=0;
		redNo++;
		int n=my_futureMsgs.length();
		for(int i=0;i<n;i++){
			CkReductionMsg *elementMesg = my_futureMsgs.deq();
			if(elementMesg->getRedNo() == redNo){
				my_msgs.enq(elementMesg);
				count++;
				collectAllMessages();
			}else{
				my_futureMsgs.enq(elementMesg);
			}
		}
	}
}

void CkArrayReductionMgr::contributeArrayReduction(CkReductionMsg *m){
	ARPRINT("[%d]Contribute Array Reduction called for RedNo %d group %d \n",CkMyNode(),m->getRedNo(),thisgroup.idx);
	/** store the contribution untill all procs have contributed. At that point reduce and
	carry out a reduction among nodegroups*/
#if CMK_BIGSIM_CHARM
	 _TRACE_BG_TLINE_END(&(m->log));
#endif
	CmiLock(lockCount);
	if(m->getRedNo() == redNo){
		my_msgs.enq(m);
		count++;
		collectAllMessages();
	}else{
		//ARPRINT("[%d][%d]Out of sequence messages for %d Present redNo %d \n",CkMyNode(),CkMyPe(),m->getRedNo(),redNo);
		my_futureMsgs.enq(m);
	}
	CmiUnlock(lockCount);
}

CkReductionMsg *CkArrayReductionMgr::reduceMessages(void){
#if CMK_BIGSIM_CHARM
        _TRACE_BG_END_EXECUTE(1);
	void* _bgParentLog = NULL;
	_TRACE_BG_BEGIN_EXECUTE_NOMSG("ArrayReduce", &_bgParentLog, 0);
#endif
	CkReductionMsg *ret=NULL;

	//Look through the vector for a valid reducer, swapping out placeholder messages
	CkReduction::reducerType r=CkReduction::invalid;
	int msgs_gcount=0;//Reduced gcount
	int msgs_nSources=0;//Reduced nSources
	int msgs_userFlag=-1;
	CkCallback msgs_callback;
	CkCallback msgs_secondaryCallback;
	int i;
	int nMsgs=0;
	CkReductionMsg *m;
	CkReductionMsg **msgArr=new CkReductionMsg*[my_msgs.length()];
	bool isMigratableContributor;

	while(NULL!=(m=my_msgs.deq()))
	{

		msgs_gcount+=m->gcount;
		if (m->sourceFlag!=0)
		{ //This is a real message from an element, not just a placeholder
			msgArr[nMsgs++]=m;
			msgs_nSources+=m->nSources();
			r=m->reducer;
			if (!m->callback.isInvalid())
				msgs_callback=m->callback;
			if(!m->secondaryCallback.isInvalid())
				msgs_secondaryCallback = m->secondaryCallback;
			if (m->userFlag!=-1)
				msgs_userFlag=m->userFlag;

			isMigratableContributor=m->isMigratableContributor();
#if CMK_BIGSIM_CHARM
			_TRACE_BG_ADD_BACKWARD_DEP(m->log);
#endif
				
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
			CkReduction::reducerFn f=CkReduction::reducerTable[r].fn;
    	ret=(*f)(nMsgs,msgArr);
		}
                ret->reducer=r;
        }

	//Go back through the vector, deleting old messages
  	for (i=0;i<nMsgs;i++) {
          if (msgArr[i] != ret) delete msgArr[i];
    }
	delete [] msgArr;

	//Set the message counts
  	ret->redNo=redNo;
  	ret->gcount=msgs_gcount;
  	ret->userFlag=msgs_userFlag;
  	ret->callback=msgs_callback;
		ret->secondaryCallback = msgs_secondaryCallback;
  	ret->sourceFlag=msgs_nSources;
		ret->setMigratableContributor(isMigratableContributor);	
	return ret;
}

void CkArrayReductionMgr::pup(PUP::er &p){
	NodeGroup::pup(p);
	p(redNo);p(count);
	p|my_msgs;
	p|my_futureMsgs;
	p|attachedGroup;
	if(p.isUnpacking()) {
	  size = CkMyNodeSize();
	  lockCount = CmiCreateLock();
	}
}

void CkArrayReductionMgr::setAttachedGroup(CkGroupID groupID){
	attachedGroup = groupID;
	ARPRINT("[%d] setAttachedGroup called with attachedGroup %d \n",CkMyNode(),attachedGroup);
	if (alreadyStarted != -1) {
		((CkNodeReductionMgr *)this)->restartLocalGroupReductions(alreadyStarted);
		alreadyStarted = -1;
	}
}


void CkArrayReductionMgr::startNodeGroupReduction(int number,CkGroupID groupID){
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    Chare *oldObj =CpvAccess(_currentObj);
    CpvAccess(_currentObj) = this;
#endif
	ARPRINT("[%d] startNodeGroupReductions for red No %d my group %d attached group %d on %p \n",CkMyNode(),number,thisgroup.idx, attachedGroup.idx,this);
	if(attachedGroup.isZero()){
		setAttachedGroup(groupID);
	}
	startLocalGroupReductions(number);
	CkReductionNumberMsg *msg = new CkReductionNumberMsg(number);
	envelope::setSrcPe((char *)UsrToEnv(msg),CkMyNode());
	((CkNodeReductionMgr *)this)->ReductionStarting(msg);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = oldObj;
#endif
}

int CkArrayReductionMgr::startLocalGroupReductions(int number){	
	ARPRINT("[%d] startLocalGroupReductions for red No %d my group %d attached group %d number of rednMgrs %d on %p \n",CkMyNode(),number,thisgroup.idx, attachedGroup.idx,size,this);
	if(attachedGroup.isZero()){
		alreadyStarted = number;
		return 0;
	}
	int firstPE = CkNodeFirst(CkMyNode());
	for(int i=0;i<size;i++){
		CProxy_CkReductionMgr reductionMgrProxy(attachedGroup);
		reductionMgrProxy[firstPE+i].ReductionStarting(new CkReductionNumberMsg(number));
	}
	return 1;
}

int CkArrayReductionMgr::getTotalGCount(){
	int firstPE = CkNodeFirst(CkMyNode());
	int totalGCount=0;
	for(int i=0;i<size;i++){
		CProxy_CkReductionMgr reductionMgrProxy(attachedGroup);
		CkReductionMgr *mgrPtr = reductionMgrProxy[firstPE+i].ckLocalBranch();
		CkAssert(mgrPtr != NULL);
		totalGCount += mgrPtr->getGCount();
	}
	return totalGCount;
}

CkArrayReductionMgr::~CkArrayReductionMgr() {
  CmiDestroyLock(lockCount);
}

#include "CkArrayReductionMgr.def.h"
