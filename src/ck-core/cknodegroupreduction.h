#ifndef _CKNODEGROUPREDUCTION_H
#define _CKNODEGROUPREDUCTION_H
/** Node Group Reductions added by Sayantan
    This thing is extremely similar to the guy above****/


    /**some data classes used by both ckreductionmgr and cknodereductionmgr**/
class contributorInfo {
	public:
		int redNo;//Current reduction number
		contributorInfo() {redNo=0;}
		//Migration utilities:
		void pup(PUP::er &p);
	};

class countAdjustment {
	public:
		int gcount;//Adjustment to global count (applied at reduction end)
		int lcount;//Adjustment to local count (applied continually)
		int mainRecvd;
		countAdjustment(int ignored=0) {gcount=lcount=0;mainRecvd=0;}
		void pup(PUP::er& p){ p|gcount; p|lcount; p|mainRecvd; }
};

/** @todo: Fwd decl for a temporary class. Remove after
 * delegated cross-array reductions are implemented more optimally
 */
namespace ck { namespace impl { class XArraySectionReducer; } }

class CkReductionMsg;
//CkReduction is just a "namespace class" for the user-visible
// parts of the reduction system.
class CkReduction {
public:
	/*These are the reducers you can use,
	  in addition to any user-defined reducers.*/
	typedef enum {
	//A placeholder invalid reduction type
		invalid=0,
	//Compute the sum the numbers passed by each element.
		sum_int,sum_float,sum_double,

	//Compute the product the numbers passed by each element.
		product_int,product_float,product_double,

	//Compute the largest number passed by any element.
		max_int,max_float,max_double,

	//Compute the smallest number passed by any element.
		min_int,min_float,min_double,

	//Compute the logical AND of the integers passed by each element.
	// The resulting integer will be zero if any source integer is zero.
		logical_and,

	//Compute the logical OR of the integers passed by each element.
	// The resulting integer will be 1 if any source integer is nonzero.
		logical_or,

                // Compute the logical bitvector AND of the integers passed by each element.
                bitvec_and,

                // Compute the logical bitvector OR of the integers passed by each element.
                bitvec_or,

	// Select one message at random to pass on
		random,

	//Concatenate the (arbitrary) data passed by each element
		concat,

	//Combine the data passed by each element into an list of setElements.
	// Each element may contribute arbitrary data (with arbitrary length).
		set,

	//Last system-defined reducer number (user-defined reducers follow)
		lastSystemReducer
	} reducerType;

	//This structure is used with the set reducer above,
	// and contains the data from one contribution.
	class setElement {
	public:
	        int dataSize;//The length of the data array below
	        char data[1];//The (dataSize-long) array of data
		//Utility routine: get the next setElement,
		// or return NULL if there are none.
		setElement *next(void);
	};

//Support for adding new reducerTypes:
	//A reducerFunction is used to combine several contributions
	//into a single summed contribution:
	//  nMsg gives the number of messages to reduce.
	//  msgs[i] contains a contribution or summed contribution.
	typedef CkReductionMsg *(*reducerFn)(int nMsg,CkReductionMsg **msgs);

	//Add the given reducer to the list.  Returns the new reducer's
	// reducerType.  Must be called in the same order on every node.
	static reducerType addReducer(reducerFn fn);

private:
	friend class CkReductionMgr;
 	friend class CkNodeReductionMgr;
	friend class CkArrayReductionMgr;
	friend class CkMulticastMgr;
    friend class ck::impl::XArraySectionReducer;
//System-level interface
	//This is the maximum number of possible reducers,
	// including both builtin and user-defined types
	enum {MAXREDUCERS=256};

	//Reducer table: maps reducerTypes to reducerFns.
	static reducerFn reducerTable[MAXREDUCERS];
	static int nReducers;//Number of reducers currently in table above

	//Don't instantiate a CkReduction object-- it's just a namespace.
	CkReduction();
};





#define CK_REDUCTION_CONTRIBUTE_METHODS_DECL \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	CMK_REFNUM_TYPE userFlag=-1); \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	const CkCallback &cb,CMK_REFNUM_TYPE userFlag=-1); \
  void contribute(CkReductionMsg *msg); \
  void contribute(const CkCallback &cb,CMK_REFNUM_TYPE userFlag=-1);\
  void contribute(CMK_REFNUM_TYPE userFlag=-1);\





class CkNodeReductionMgr : public IrrGroup {
public:
	CProxy_CkNodeReductionMgr thisProxy;
public:
	CkNodeReductionMgr(void);
	CkNodeReductionMgr(CkMigrateMessage *m) : IrrGroup(m) {
          storedCallback = NULL;
        }

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
        void restartLocalGroupReductions(int number);
	//Sent down the reduction tree (used by barren PEs)
	void ReductionStarting(CkReductionNumberMsg *m);
	//Sent up the reduction tree with reduced data
	void RecvMsg(CkReductionMsg *m);
	void doRecvMsg(CkReductionMsg *m);
	void LateMigrantMsg(CkReductionMsg *m);

	virtual void flushStates();	// flush state varaibles
	virtual int startLocalGroupReductions(int number){ return 1;} // can be used to start reductions on all the 
	//CkReductionMgrs on a particular node. It is overwritten by CkArrayReductionMgr to make the actual calls
	// since it knows the CkReductionMgrs on a node.

	virtual int getTotalGCount(){return 0;};

private:
//Data members
	//Stored callback function (may be NULL if none has been set)
	CkCallback *storedCallback;

	int redNo;//Number of current reduction (incremented at end)
	CmiBool inProgress;//Is a reduction started, but not complete?
	CmiBool creating;//Are elements still being created?
	CmiBool startRequested;//Should we start the next reduction when creation finished?
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

	int interrupt; /* flag for use in non-smp 0 means interrupt can occur 1 means not (also acts as a lock)*/

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
	CmiBool hasParent(void);
	int treeParent(void);//My parent PE
	int firstKid(void);//My first child PE
	int treeKids(void);//Number of children in tree

	//Combine (& free) the current message vector.
	CkReductionMsg *reduceMessages(void);

	//Map reduction number to a time
	CmiBool isPast(int num) const {return (CmiBool)(num<redNo);}
	CmiBool isPresent(int num) const {return (CmiBool)(num==redNo);}
	CmiBool isFuture(int num) const {return (CmiBool)(num>redNo);}

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
	int killed;	
	
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



//A NodeGroup that contribute to reductions
class NodeGroup : public CkNodeReductionMgr {
  protected:
    contributorInfo reductionInfo;//My reduction information
  public:
    CmiNodeLock __nodelock;
    NodeGroup();
    NodeGroup(CkMigrateMessage* m):CkNodeReductionMgr(m) { __nodelock=CmiCreateLock(); }
    
    ~NodeGroup();
    inline const CkGroupID &ckGetGroupID(void) const {return thisgroup;}
    inline CkGroupID CkGetNodeGroupID(void) const {return thisgroup;}
    virtual int isNodeGroup() { return 1; }

    virtual void pup(PUP::er &p);
    virtual void flushStates() {
    	CkNodeReductionMgr::flushStates();
        reductionInfo.redNo = 0;
    }

    CK_REDUCTION_CONTRIBUTE_METHODS_DECL
    void contributeWithCounter(CkReductionMsg *msg,int count);
};


#endif

