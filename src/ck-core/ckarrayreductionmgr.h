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
		CkMsgQ<CkReductionMsg> my_msgs;
		CkMsgQ<CkReductionMsg> my_futureMsgs;
		CmiNodeLock lockCount;
		void collectAllMessages();
	public:
		CkArrayReductionMgr();
		CkArrayReductionMgr(CkMigrateMessage *m):NodeGroup(m) {}
		void contributeArrayReduction(CkReductionMsg *m);
		CkReductionMsg *reduceMessages(void);
		virtual void pup(PUP::er &p);
};
#endif

