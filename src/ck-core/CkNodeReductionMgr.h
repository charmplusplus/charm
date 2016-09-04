#ifndef _CKNODERED_MGR_H
#define _CKNODERED_MGR_H

#include "CkReductionMgr.h"

/**
 * One CkReductionMgr runs a non-overlapping set of reductions.
 * It collects messages from all local contributors, then sends
 * the reduced message up the reduction tree to node zero, where
 * they're passed to the user's client function.
 */
class CkNodeReductionMgr : public IrrGroup {
public:
	CkNodeReductionMgr(void);
	CkNodeReductionMgr(CkMigrateMessage *m) : IrrGroup(m) {
          storedCallback = NULL;
        }
        ~CkNodeReductionMgr();

	typedef CkReductionClientFn clientFn;

	/**
	 * Add the given client function.  Overwrites any previous client.
	 * This manager will dispose of the callback when replaced or done.
	 */
	void ckSetReductionClient(CkCallback *cb);

//Contribute-- the given msg can contain any data.  The reducerType
// field of the message must be valid.
// Each contributor must contribute exactly once to each reduction.
	void contribute(contributorInfo *ci,CkReductionMsg *msg);
	void contributeWithCounter(contributorInfo *ci,CkReductionMsg *m,int count);
//Communication (library-private)
	//Sent up the reduction tree with reduced data
	void RecvMsg(CkReductionMsg *m);
	void doRecvMsg(CkReductionMsg *m);
	void LateMigrantMsg(CkReductionMsg *m);

	virtual void flushStates();	// flush state varaibles

	virtual int getTotalGCount(){return 0;};

private:
//Data members
	//Stored callback function (may be NULL if none has been set)
	CkCallback *storedCallback;

	int redNo;//Number of current reduction (incremented at end)
	bool inProgress;//Is a reduction started, but not complete?
	bool creating;//Are elements still being created?
	bool startRequested;//Should we start the next reduction when creation finished?
	int gcount;
	int lcount;//Number of local contributors

	//Current local and remote contributions
	int nContrib,nRemote;
	//Contributions queued for the current reduction
	CkMsgQ<CkReductionMsg> msgs;
	//Contributions queued for future reductions (sent to us too early)
	CkMsgQ<CkReductionMsg> futureMsgs;
	//Remote messages queued for future reductions (sent to us too early)
	CkMsgQ<CkReductionMsg> futureRemoteMsgs;
	//Late migrant messages queued for future reductions
	CkMsgQ<CkReductionMsg> futureLateMigrantMsgs;
	
	//My Big LOCK
	CmiNodeLock lockEverything;

	bool interrupt; /* flag for use in non-smp: false means interrupt can occur, true means not (also acts as a lock) */

	/*vector storing the children of this node*/
	CkVec<int> kids;
	
//State:
	void startReduction(int number,int srcPE);
	void doAddContribution(CkReductionMsg *m);
	void finishReduction(void);
protected:	
	void addContribution(CkReductionMsg *m);

private:

//Reduction tree utilities
/* for binomial trees*/
	unsigned upperSize;
	unsigned label;
	int parent;
	int numKids;
//	int *kids;
	void init_BinomialTree();

	
	void init_BinaryTree();
	enum {TREE_WID=2};
	int treeRoot(void);//Root PE
	bool hasParent(void);
	int treeParent(void);//My parent PE
	int firstKid(void);//My first child PE
	int treeKids(void);//Number of children in tree

	//Combine (& free) the current message vector.
	CkReductionMsg *reduceMessages(void);

	//Map reduction number to a time
	bool isPast(int num) const {return (bool)(num<redNo);}
	bool isPresent(int num) const {return (bool)(num==redNo);}
	bool isFuture(int num) const {return (bool)(num>redNo);}

	/*FAULT_EVAC*/
	bool oldleaf;
	bool blocked;
	int newParent;
	int additionalGCount,newAdditionalGCount; //gcount that gets passed to u from the node u replace
	CkVec<int> newKids;
	CkMsgQ<CkReductionMsg> bufferedMsgs;
	CkMsgQ<CkReductionMsg> bufferedRemoteMsgs;
	enum {OLDPARENT,OLDCHILDREN,NEWPARENT,LEAFPARENT};
	int numModificationReplies;
	int maxModificationRedNo;
	int tempModificationRedNo;
	bool readyDeletion;
	bool killed;
	
//Checkpointing utilities
 public:
	virtual void pup(PUP::er &p);
	/*FAULT_EVAC*/
	virtual void evacuate();
	virtual void doneEvacuate();
	void DeleteChild(int deletedChild);
	void DeleteNewChild(int deletedChild);
	void collectMaxRedNo(int maxRedNo);
	void unblockNode(int maxRedNo);
	void modifyTree(int code,int size,int *data);
private:	
	int findMaxRedNo();
	void updateTree();
	void clearBlockedMsgs();
};


#endif
