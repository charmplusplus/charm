#include "charm++.h"
#include "ck.h"
#include "CkArrayReductionMgr.decl.h"



CkArrayReductionMgr::CkArrayReductionMgr(){
	//CkPrintf("Array ReductionMgr Constructor called %d\n",thisgroup);
	if(CkMyRank()== 0){
	redNo=0;
	size = CmiMyNodeSize();
	count = 0;
	lockCount = CmiCreateLock();
	}
};

void CkArrayReductionMgr::contributeArrayReduction(CkReductionMsg *m){
	//CkPrintf("[%d]Contribute Array Reduction called for RedNo %d\n",CkMyNode(),m->getRedNo());
	/** store the contribution untill all procs have contributed. At that point reduce and
	carry out a reduction among nodegroups*/
	CmiLock(lockCount);
	if(m->getRedNo() == redNo){
		my_msgs.push_back(m);
		count++;
		if(count == size){
			//CkPrintf("[%d] About to call contributewithCounter for %d with %d\n",CkMyNode(),redNo,count);
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
					my_msgs.push_back(elementMesg);
					count++;
				}else{
					my_futureMsgs.enq(elementMesg);
				}
			}

		}
	}else{
		CkPrintf("[%d][%d]Out of sequence messages for %d Present redNo %d \n",CkMyNode(),CkMyPe(),m->getRedNo(),redNo);
		my_futureMsgs.enq(m);
	}
	CmiUnlock(lockCount);
};
CkReductionMsg *CkArrayReductionMgr::reduceMessages(void){
	CkReductionMsg *ret=NULL;
  	int nMsgs=my_msgs.length();
	CkReduction::reducerType r=CkReduction::invalid;
  	int msgs_gcount=0;//Reduced gcount
  	int msgs_nSources=0;//Reduced nSources
  	int msgs_userFlag=-1;
  	CkCallback msgs_callback;
  	int i;
  	for (i=0;i<nMsgs;i++)
  	{
    		CkReductionMsg *m=my_msgs[i];
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
      			my_msgs[i--]=my_msgs[--nMsgs];
      			delete m;
    		}
  	}

  	if (nMsgs==0||r==CkReduction::invalid)
  		//No valid reducer in the whole vector
    		ret=CkReductionMsg::buildNew(0,NULL);
  	else
  	{//Use the reducer to reduce the messages
    		CkReduction::reducerFn f=CkReduction::reducerTable[r];
    		CkReductionMsg **msgArr=&my_msgs[0];//<-- HACK!
    		ret=(*f)(nMsgs,msgArr);
    		ret->reducer=r;
  	}

  //Go back through the vector, deleting old messages
  	for (i=0;i<nMsgs;i++) delete my_msgs[i];

  //Set the message counts
  	ret->redNo=redNo;
  	ret->gcount=msgs_gcount;
  	ret->userFlag=msgs_userFlag;
  	ret->callback=msgs_callback;
  	ret->sourceFlag=msgs_nSources;


  //Empty out the message vector
  	my_msgs.length()=0;
  return ret;

}


CkReductionMsg* CkArrayReductionMgr::pupCkReductionMsg(CkReductionMsg *m, PUP::er &p)
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


void CkArrayReductionMgr::pupMsgVector(CkVec<CkReductionMsg *> &_msgs, PUP::er &p)
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
void CkArrayReductionMgr::pupMsgQ(CkQ<CkReductionMsg *> &_msgs, PUP::er &p)
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
void CkArrayReductionMgr::pup(PUP::er &p){
	p(size);p(redNo);p(count);
	pupMsgVector(my_msgs,p);
	pupMsgQ(my_futureMsgs,p);
}

#include "CkArrayReductionMgr.def.h"
