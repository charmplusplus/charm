#include "charm++.h"
#include "ck.h"
#include "CkArrayReductionMgr.decl.h"
#define ARRREDDEBUG 0

#if ARRREDDEBUG
#define ARPRINT ARPRINT
#else
#define ARPRINT //ARPRINT
#endif

CkArrayReductionMgr::CkArrayReductionMgr(){
	//ARPRINT("Array ReductionMgr Constructor called %d\n",thisgroup);
	if(CkMyRank()== 0){
	redNo=0;
	size = CmiMyNodeSize();
	count = 0;
	lockCount = CmiCreateLock();
	}
};

void CkArrayReductionMgr::collectAllMessages(){
	if(count == size){
		ARPRINT("[%d] CollectAll messages  for %d with %d\n",CkMyNode(),redNo,count);
		CkReductionMsg *result = reduceMessages();
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
	ARPRINT("[%d]Contribute Array Reduction called for RedNo %d\n",CkMyNode(),m->getRedNo());
	/** store the contribution untill all procs have contributed. At that point reduce and
	carry out a reduction among nodegroups*/
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
};
CkReductionMsg *CkArrayReductionMgr::reduceMessages(void){
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
		CkReduction::reducerFn f=CkReduction::reducerTable[r];
    		ret=(*f)(nMsgs,msgArr);
    		ret->reducer=r;
  	}

	//Go back through the vector, deleting old messages
  	for (i=0;i<nMsgs;i++) delete msgArr[i];

	//Set the message counts
  	ret->redNo=redNo;
  	ret->gcount=msgs_gcount;
  	ret->userFlag=msgs_userFlag;
  	ret->callback=msgs_callback;
	ret->secondaryCallback = msgs_secondaryCallback;
  	ret->sourceFlag=msgs_nSources;

	return ret;
}

void CkArrayReductionMgr::pup(PUP::er &p){
	NodeGroup::pup(p);
	p(size);p(redNo);p(count);
	p|my_msgs;
	p|my_futureMsgs;
}

#include "CkArrayReductionMgr.def.h"
