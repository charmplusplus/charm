#ifndef _CKARRAYREDNMGR_H
#define _CKARRAYREDNMGR_H
class CkArrayReductionMgr : public NodeGroup{
	/** This class receives	contributions from all the CkReductionMgr s in a node, after each of them has 
	collected all the contributions on one processor. The data from the different processors in a node
	is collected and sent to the CkNodeReductionMgr for this node
	*/
	private:
		int size;
		int redNo;
		int count;
		CkGroupID attachedGroup;
		CkMsgQ<CkReductionMsg> my_msgs;
		CkMsgQ<CkReductionMsg> my_futureMsgs;
		CmiNodeLock lockCount;
		int alreadyStarted;
		void init();
		void collectAllMessages();
	public:
		volatile int ctorDoneFlag;
		CkArrayReductionMgr();
		CkArrayReductionMgr(int dummy, CkGroupID gid);
		CkArrayReductionMgr(CkMigrateMessage *m):NodeGroup(m) {}
		void contributeArrayReduction(CkReductionMsg *m);
		CkReductionMsg *reduceMessages(void);
                void flushStates();
		virtual void pup(PUP::er &p);
		void setAttachedGroup(CkGroupID groupID);
		void startNodeGroupReduction(int number,CkGroupID groupID);
		virtual int startLocalGroupReductions(int number);
		virtual int getTotalGCount();
                ~CkArrayReductionMgr();
};
#endif

