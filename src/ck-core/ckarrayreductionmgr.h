#ifndef _CKARRAYREDNMGR_H
#define _CKARRAYREDNMGR_H
class CkArrayReductionMgr : public NodeGroup{
	/** Group Reduction Managers use this class
	to carry out their reductions. This class receives
	contributions from group/array redn mgrs when all
	data from elements on a processor has been collected
	This is then contributed to this group.*/
	private:
		int size;
		int redNo;
		int count;
		CkVec<CkReductionMsg *> my_msgs;
		CkQ<CkReductionMsg *> my_futureMsgs;
		CmiNodeLock lockCount;
		void collectAllMessages();
	public:
		CkArrayReductionMgr();
		void contributeArrayReduction(CkReductionMsg *m);
		CkReductionMsg *reduceMessages(void);
		virtual void pup(PUP::er &p);
		void pupMsgVector(CkVec<CkReductionMsg *> &msgs, PUP::er &p);
      		void pupMsgQ(CkQ<CkReductionMsg *> &msgs, PUP::er &p);
		CkReductionMsg* pupCkReductionMsg(CkReductionMsg *m, PUP::er &p);
};
#endif

