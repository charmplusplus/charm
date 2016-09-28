/*
Charm++ File: Reduction Library
added 3/27/2000 by Orion Sky Lawlor, olawlor@acm.org
modified 02/21/2003 by Sayantan Chakravorty


A reduction takes some sort of inputs (contributions)
from some set of objects scattered across all PE's,
and combines (reduces) all the contributions onto one
PE.  This library provides several different kinds of
combination routines (reducers), and various utilities
for supporting them.

The calls needed to use the reduction manager are:
-Create with CProxy_CkReduction::ckNew.

*/
#ifndef _CKREDUCTION_H
#define _CKREDUCTION_H

#include "CkReduction.decl.h"
#include "CkArrayReductionMgr.decl.h"

#if CMK_BIGSIM_CHARM || CMK_MULTICORE || !CMK_SMP
#define GROUP_LEVEL_REDUCTION           1
#endif

#ifdef _PIPELINED_ALLREDUCE_
#define FRAG_SIZE 131072
#define FRAG_THRESHOLD 131072
#endif

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
#define MAX_INT 5000000
#define _MLOG_REDUCE_P2P_ 0
#endif

//This message is sent between group objects on a single PE
// to let each know the other has been created.
class CkGroupCallbackMsg:public CMessage_CkGroupCallbackMsg {
public:
	typedef void (*callbackType)(void *param);
	CkGroupCallbackMsg(callbackType Ncallback,void *Nparam)
		{callback=Ncallback;param=Nparam;}
	void call(void) {(*callback)(param);}
private:
	callbackType callback;
	void *param;
};

class CkGroupInitCallback : public IrrGroup {
public:
	CkGroupInitCallback(void);
	CkGroupInitCallback(CkMigrateMessage *m):IrrGroup(m) {}
	void callMeBack(CkGroupCallbackMsg *m);
	void pup(PUP::er& p){ IrrGroup::pup(p); }
};


class CkGroupReadyCallback : public IrrGroup {
private:
  int _isReady;
  CkQ<CkGroupCallbackMsg *> _msgs;
  void callBuffered(void);
public:
	CkGroupReadyCallback(void);
	CkGroupReadyCallback(CkMigrateMessage *m):IrrGroup(m) {}
	void callMeBack(CkGroupCallbackMsg *m);
	int isReady(void) { return _isReady; }
protected:
	void setReady(void) {_isReady = 1; callBuffered(); }
	void setNotReady(void) {_isReady = 0; }
};

class CkReductionNumberMsg:public CMessage_CkReductionNumberMsg {
public:
  int num;
  CkReductionNumberMsg(int n) {num=n;}
};


class CkReductionInactiveMsg:public CMessage_CkReductionInactiveMsg {
  public:
    int id, redno;
    CkReductionInactiveMsg(int i, int r) {id=i; redno = r;}
};


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
    countAdjustment(int ignored=0) {(void)ignored; gcount=0; lcount=0; mainRecvd=0;}
	void pup(PUP::er& p){ p|gcount; p|lcount; p|mainRecvd; }
};

/** @todo: Fwd decl for a temporary class. Remove after
 * delegated cross-array reductions are implemented more optimally
 */
namespace ck { namespace impl { class XArraySectionReducer; } }

//CkReduction is just a "namespace class" for the user-visible
// parts of the reduction system.
class CkReduction {
public:
	/*These are the reducers you can use,
	  in addition to any user-defined reducers.*/

        /*  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                 remember to update CkReduction::reducerTable

            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  */

	typedef enum {
	//A placeholder invalid reduction type
		invalid=0,
                nop,
	//Compute the sum the numbers passed by each element.
		sum_char,sum_short,sum_int,sum_long,sum_long_long,
                sum_uchar,sum_ushort,sum_uint,sum_ulong,
                sum_ulong_long,sum_float,sum_double,

	//Compute the product the numbers passed by each element.
		product_char,product_short,product_int,product_long,product_long_long,
                product_uchar,product_ushort,product_uint,product_ulong,
                product_ulong_long,product_float,product_double,

	//Compute the largest number passed by any element.
		max_char,max_short,max_int,max_long,max_long_long,
                max_uchar,max_ushort,max_uint,max_ulong,
                max_ulong_long,max_float,max_double,

	//Compute the smallest number passed by any element.
		min_char,min_short,min_int,min_long,min_long_long,
                min_uchar,min_ushort,min_uint,min_ulong,
                min_ulong_long,min_float,min_double,

	//Compute the logical AND of the values passed by each element.
	// The resulting value will be zero if any source value is zero.
		logical_and, // Deprecated: same as logical_and_int
                logical_and_int,logical_and_bool,

	//Compute the logical OR of the values passed by each element.
	// The resulting value will be 1 if any source value is nonzero.
		logical_or, // Deprecated: same as logical_or_int
                logical_or_int,logical_or_bool,

	//Compute the logical XOR of the values passed by each element.
	// The resulting value will be 1 if an odd number of source value is nonzero.
                logical_xor_int,logical_xor_bool,

                // Compute the logical bitvector AND of the values passed by each element.
                bitvec_and, // Deprecated: same as bitvec_and_int
                bitvec_and_int,bitvec_and_bool,

                // Compute the logical bitvector OR of the values passed by each element.
                bitvec_or, // Deprecated: same as bitvec_or_int
                bitvec_or_int,bitvec_or_bool,

                // Compute the logical bitvector XOR of the values passed by each element.
                bitvec_xor, // Deprecated: same as bitvec_xor_int
                bitvec_xor_int,bitvec_xor_bool,

	// Select one message at random to pass on
		random,

	//Concatenate the (arbitrary) data passed by each element
		concat,

	//Combine the data passed by each element into an list of setElements.
	// Each element may contribute arbitrary data (with arbitrary length).
        set,

        // Calculate the count, mean, and variance / standard deviation of the data
        statistics,

        // Combine multiple data/reducer pairs into one reduction
        tuple,

	//Last system-defined reducer number (user-defined reducers follow)
		lastSystemReducer
	} reducerType;

	//This structure is used with the set reducer above,
	// and contains the data from one contribution.
	class setElement {
	public:
	        int dataSize;//The allocated length of the `data' array, in bytes
	        char data[1];//The beginning of the array of data
		//Utility routine: get the next setElement,
		// or return NULL if there are none.
		setElement *next(void);
    };

    // Structure containing the payload of a statistics reduction
    struct statisticsElement {
        int count;
        double mean;
        double m2;
        statisticsElement(double initialValue);
        double variance() const { return count > 1 ? m2 / (double(count) - 1.0) : 0.0; }
        double stddev() const { return sqrt(variance()); }
    };

    struct tupleElement {
        size_t dataSize;
        char* data;
        CkReduction::reducerType reducer;
        bool owns_data;
        tupleElement();
        tupleElement(size_t dataSize, void* data, CkReduction::reducerType reducer);
        tupleElement(CkReduction::tupleElement&& rhs_move);
        tupleElement& operator=(CkReduction::tupleElement&& rhs_move);
        ~tupleElement();

        inline void* getData(void) { return data; }
        void pup(PUP::er &p);
    };

//Support for adding new reducerTypes:
	//A reducerFunction is used to combine several contributions
	//into a single summed contribution:
	//  nMsg gives the number of messages to reduce.
	//  msgs[i] contains a contribution or summed contribution.
	typedef CkReductionMsg *(*reducerFn)(int nMsg,CkReductionMsg **msgs);

  struct reducerStruct {
    reducerFn fn;
    bool streamable;
    reducerStruct(reducerFn f=NULL, bool s=false) : fn(f), streamable(s) {}
  };

	//Add the given reducer to the list.  Returns the new reducer's
	// reducerType.  Must be called in the same order on every node.
	static reducerType addReducer(reducerFn fn, bool streamable=false);

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
	static reducerStruct reducerTable[MAXREDUCERS];
	static int nReducers;//Number of reducers currently in table above

    // tupleReduction needs access to the reducerTable that lives in this namespace
    // so it is not a standalone function in ckreduction.C like other reduction implementations
    static CkReductionMsg* tupleReduction(int nMsgs, CkReductionMsg** msgs);

	//Don't instantiate a CkReduction object-- it's just a namespace.
	CkReduction();
};
PUPbytes(CkReduction::reducerType)

//A CkReductionMsg is sent up the reduction tree-- it
// carries a contribution, or several reduced contributions.
class CkReductionMsg : public CMessage_CkReductionMsg
{
	friend class CkReduction;
	friend class CkReductionMgr;
	friend class CkNodeReductionMgr;
	friend class CkArrayReductionMgr;
	friend class CkMulticastMgr;
#ifdef _PIPELINED_ALLREDUCE_
	friend class ArrayElement;
	friend class AllreduceMgr;
#endif
	friend class ck::impl::XArraySectionReducer;
public:

//Publically-accessible fields:
	//"Constructor"-- builds and returns a new CkReductionMsg.
	//  the "srcData" array you specify will be copied into this object (unless NULL).
	static CkReductionMsg *buildNew(int NdataSize,const void *srcData,
		CkReduction::reducerType reducer=CkReduction::invalid,
                CkReductionMsg *buf = NULL);

	inline int getLength(void) const {return dataSize;}
	inline int getSize(void) const {return dataSize;}
	inline void *getData(void) {return data;}
	inline const void *getData(void) const {return data;}

	inline int getGcount(void){return gcount;}
	inline CkReduction::reducerType getReducer(void){return reducer;}
	inline int getRedNo(void){return redNo;}

	inline CMK_REFNUM_TYPE getUserFlag(void) const {return userFlag;}
	inline void setUserFlag(CMK_REFNUM_TYPE f) { userFlag=f;}

	inline void setCallback(const CkCallback &cb) { callback=cb; }

	//Return true if this message came straight from a contribute call--
	// if it didn't come from a previous reduction function.
	inline int isFromUser(void) const {return sourceFlag==-1;}

	inline bool isMigratableContributor(void) const {return migratableContributor;}
	inline void setMigratableContributor(bool _mig){ migratableContributor = _mig;}

    // Tuple reduction
    static CkReductionMsg* buildFromTuple(CkReduction::tupleElement* reductions, int num_reductions);
    void toTuple(CkReduction::tupleElement** out_reductions, int* num_reductions);

	~CkReductionMsg();

//Implementation-only fields (don't access these directly!)
	//Msg runtime support
	static void *alloc(int msgnum, size_t size, int *reqSize, int priobits);
	static void *pack(CkReductionMsg *);
	static CkReductionMsg *unpack(void *in);

#if CMK_BIGSIM_CHARM
	/* AMPI reductions use bare CkReductionMsg's instead of AmpiMsg's */
	void *event; // the event point that corresponds to this message
	int eventPe; // the PE that the event is located on
#endif

private:
	int dataSize;//Length of array below, in bytes
	void *data;//Reduction data
	CMK_REFNUM_TYPE userFlag; //Some sort of identifying flag, for client use
	CkCallback callback; //What to do when done
	CkCallback secondaryCallback; // the group callback is piggybacked on the nodegrp reduction
	bool migratableContributor; // are the contributors migratable

	int sourceFlag;/*Flag:
		0 indicates this is a placeholder message (meaning: nothing to report)
		-1 indicates this is a single (non-reduced) contribution.
  		>0 indicates this is a reduced contribution.
  	*/
  	int nSources(void) {return sourceFlag<0?-sourceFlag:sourceFlag;}
#if (defined(_FAULT_MLOG_) && _MLOG_REDUCE_P2P_ )
    int sourceProcessorCount;
#endif
    int fromPE;
private:
#if CMK_BIGSIM_CHARM
        void *log;
#endif
	CkReduction::reducerType reducer;
	//contributorInfo *ci;//Source contributor, or NULL if none
	int redNo;//The serial number of this reduction
	int gcount;//Contribution to the global contributor count
        // for section multicast/reduction library
        CkSectionInfo sid;   // section cookie for multicast
        char rebuilt;          // indicate if the multicast tree needs rebuilt
        int nFrags;
        int fragNo;      // fragment of a reduction msg (when pipelined)
                         // value = 0 to nFrags-1
	double dataStorage;//Start of data array (so it's double-aligned)

	int no;

	//Default constructor is private so you must use "buildNew", above
    CkReductionMsg();
};


#define CK_REDUCTION_CONTRIBUTE_METHODS_DECL \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1); \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1); \
  void contribute(CkReductionMsg *msg); \
  void contribute(const CkCallback &cb,CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1);\
  void contribute(CMK_REFNUM_TYPE userFlag=(CMK_REFNUM_TYPE)-1);

#define CK_BARRIER_CONTRIBUTE_METHODS_DECL \
  void barrier(const CkCallback &cb);\

/**
 * One CkReductionMgr runs a non-overlapping set of reductions.
 * It collects messages from all local contributors, then sends
 * the reduced message up the reduction tree to node zero, where
 * they're passed to the user's client function.
 */
class CkNodeReductionMgr : public IrrGroup {
public:
	CProxy_CkNodeReductionMgr thisProxy;
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
        void restartLocalGroupReductions(int number);
	//Sent down the reduction tree (used by barren PEs)
	void ReductionStarting(CkReductionNumberMsg *m);
	//Sent up the reduction tree with reduced data
	void RecvMsg(CkReductionMsg *m);
	void doRecvMsg(CkReductionMsg *m);
	void LateMigrantMsg(CkReductionMsg *m);

	virtual void flushStates();	// flush state varaibles
    virtual int startLocalGroupReductions(int number){ (void)number; return 1;} // can be used to start reductions on all the
	//CkReductionMgrs on a particular node. It is overwritten by CkArrayReductionMgr to make the actual calls
	// since it knows the CkReductionMgrs on a node.

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


class CProxy_CkArrayReductionMgr;
class CkReductionMgr : public CkGroupInitCallback {
public:
        CProxy_CkReductionMgr thisProxy;

public:
	CkReductionMgr(CProxy_CkArrayReductionMgr groupRednMgr);
	CkReductionMgr(CkMigrateMessage *m);
        ~CkReductionMgr();

	typedef CkReductionClientFn clientFn;

	/**
	 * Add the given client function.  Overwrites any previous client.
	 * This manager will dispose of the callback when replaced or done.
	 */
	void ckSetReductionClient(CkCallback *cb);

//Contributors keep a copy of this structure:


//Contributor list maintainance:
	//These just set and clear the "creating" flag to prevent
	// reductions from finishing early because not all elements
	// have been created.
	void creatingContributors(void);
	void doneCreatingContributors(void);
	//Initializes a new contributor
	void contributorStamped(contributorInfo *ci);//Increment global number
	void contributorCreated(contributorInfo *ci);//Increment local number
	void contributorDied(contributorInfo *ci);//Don't expect more contributions
	//Migrating away
	void contributorLeaving(contributorInfo *ci);
	//Migrating in
	void contributorArriving(contributorInfo *ci);

//Contribute-- the given msg can contain any data.  The reducerType
// field of the message must be valid.
// Each contributor must contribute exactly once to each reduction.
	void contribute(contributorInfo *ci,CkReductionMsg *msg);

//Communication (library-private)
	//Sent down the reduction tree (used by barren PEs)
	void ReductionStarting(CkReductionNumberMsg *m);
	//Sent to root of the reduction tree with late migrant data
	void LateMigrantMsg(CkReductionMsg *m);
	//A late migrating contributor will never contribute
	void MigrantDied(CkReductionNumberMsg *m);

	void RecvMsg(CkReductionMsg *m);
  void AddToInactiveList(CkReductionInactiveMsg *m);

	//Call back for using Node added by Sayantan
	void ArrayReductionHandler(CkReductionMsg *m);
	void endArrayReduction();

// simple barrier for FT
        void barrier(CkReductionMsg * msg);
        void Barrier_RecvMsg(CkReductionMsg *m);
        void addBarrier(CkReductionMsg *m);
        void finishBarrier(void);

	virtual bool isReductionMgr(void){ return true; }
	virtual void flushStates();
	/*FAULT_EVAC: used to get the gcount on a processor when 
		it is evacuated.
		TODO: It needs to be fixed as it should return the gcount
		and the adjustment information for objects that might have
		contributed and died.
		The current implementation lets us get by in the case
		when there are no gcount
	*/
	int getGCount(){return gcount;};
        static void sanitycheck();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	void decGCount(){gcount--;}
	void incNumImmigrantRecObjs(){
		numImmigrantRecObjs++;
	}
	void decNumImmigrantRecObjs(){
		numImmigrantRecObjs--;
	}
	void incNumEmigrantRecObjs(){
		numEmigrantRecObjs++;
	}
	void decNumEmigrantRecObjs(){
		numEmigrantRecObjs--;
	}

#endif

private:

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
	int numImmigrantRecObjs;
	int numEmigrantRecObjs;
#endif

#if !GROUP_LEVEL_REDUCTION
	CProxy_CkArrayReductionMgr nodeProxy; //holds the local branch of the nodegroup tree
#endif

//Data members
	//Stored callback function (may be NULL if none has been set)
	CkCallback storedCallback;
	// calback that came along with the contribute
 	CkCallback *secondaryStoredCallback;

	int redNo;//Number of current reduction (incremented at end) to be deposited with NodeGroups
	int completedRedNo;//Number of reduction Completed ie recieved callback from NodeGroups
	bool inProgress;//Is a reduction started, but not complete?
	bool creating;//Are elements still being created?
	bool startRequested;//Should we start the next reduction when creation finished?
	int gcount;//=el't created here - el't deleted here
	int lcount;//Number of local contributors
	int maxStartRequest; // the highest future ReductionStarting message received

	//Current local and remote contributions
	int nContrib,nRemote;
  // Is it inactive
  bool is_inactive;

        // simple barrier
        CkCallback barrier_storedCallback;
        int barrier_gCount;
        int barrier_nSource;
        int barrier_nContrib,barrier_nRemote;

	//Contributions queued for the current reduction
	CkMsgQ<CkReductionMsg> msgs;

	//Contributions queued for future reductions (sent to us too early)
	CkMsgQ<CkReductionMsg> futureMsgs;
	//Remote messages queued for future reductions (sent to us too early)
	CkMsgQ<CkReductionMsg> futureRemoteMsgs;

	CkMsgQ<CkReductionMsg> finalMsgs;
  std::map<int, int> inactiveList;

//State:
	void startReduction(int number,int srcPE);
	void addContribution(CkReductionMsg *m);
	void finishReduction(void);
  void checkIsActive();
  void informParentInactive();
  void checkAndAddToInactiveList(int id, int red_no);
  void checkAndRemoveFromInactiveList(int id, int red_no);
  void sendReductionStartingToKids(int red_no);

//Reduction tree utilities
	unsigned upperSize;
	unsigned label;
	int parent;
	int numKids;
	/*vector storing the children of this node*/
	CkVec<int> newKids;
	CkVec<int> kids;
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


	//This vector of adjustments is indexed by redNo,
	// starting from the current redNo.
	CkVec<countAdjustment> adjVec;
	//Return the countAdjustment struct for the given redNo:
	countAdjustment &adj(int number);
	//Shift the list of countAdjustments down
	void shiftAdjVec(void);

protected:
	//whether to notify children that reduction starts
	bool disableNotifyChildrenStart;
	void resetCountersWhenFlushingStates() { gcount = lcount = 0; }
        bool isDestroying;

//Checkpointing utilities
public:
#if (defined(_FAULT_MLOG_) && _MLOG_REDUCE_P2P_)
    int *perProcessorCounts;
    int processorCount;
    int totalCount;
    int numberReductionMessages(){
            if(totalCount != 0){
                return totalCount;
            }else{
                return MAX_INT;
            }
    }
#endif
	virtual void pup(PUP::er &p);
	static int isIrreducible(){ return 0;}
	void contributeViaMessage(CkReductionMsg *m);
};

//Define methods used to contribute to the given reduction type.
//  Data is copied, not deleted.
/*#define CK_REDUCTION_CONTRIBUTE_METHODS_DECL \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	CMK_REFNUM_TYPE userFlag=-1); \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	const CkCallback &cb,CMK_REFNUM_TYPE userFlag=-1); \
  void contribute(CkReductionMsg *msg);\*/

#define CkReductionTarget(me, method) \
    CkIndex_##me::redn_wrapper_##method(NULL)

#define CK_REDUCTION_CONTRIBUTE_METHODS_DEF(me,myRednMgr,myRednInfo,migratable) \
void me::contribute(int dataSize,const void *data,CkReduction::reducerType type,\
	CMK_REFNUM_TYPE userFlag)\
{\
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);\
	msg->setUserFlag(userFlag);\
	msg->setMigratableContributor(migratable);\
	myRednMgr->contribute(&myRednInfo,msg);\
}\
void me::contribute(int dataSize,const void *data,CkReduction::reducerType type,\
	const CkCallback &cb,CMK_REFNUM_TYPE userFlag)\
{\
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);\
	msg->setUserFlag(userFlag);\
	msg->setCallback(cb);\
	msg->setMigratableContributor(migratable);\
	myRednMgr->contribute(&myRednInfo,msg);\
}\
void me::contribute(CkReductionMsg *msg) \
	{\
	msg->setMigratableContributor(migratable);\
	myRednMgr->contribute(&myRednInfo,msg);\
	}\
void me::contribute(const CkCallback &cb,CMK_REFNUM_TYPE userFlag)\
{\
	CkReductionMsg *msg=CkReductionMsg::buildNew(0,NULL,CkReduction::nop);\
    msg->setUserFlag(userFlag);\
    msg->setCallback(cb);\
    msg->setMigratableContributor(migratable);\
    myRednMgr->contribute(&myRednInfo,msg);\
}\
void me::contribute(CMK_REFNUM_TYPE userFlag)\
{\
    CkReductionMsg *msg=CkReductionMsg::buildNew(0,NULL,CkReduction::nop);\
    msg->setUserFlag(userFlag);\
    msg->setMigratableContributor(migratable);\
    myRednMgr->contribute(&myRednInfo,msg);\
}\

#define CK_BARRIER_CONTRIBUTE_METHODS_DEF(me,myRednMgr,myRednInfo,migratable) \
void me::barrier(const CkCallback &cb)\
{\
    CkReductionMsg *msg=CkReductionMsg::buildNew(0,NULL,CkReduction::nop);\
    msg->setCallback(cb);\
    msg->setMigratableContributor(migratable);\
    myRednMgr->barrier(msg);\
}\


//A group that can contribute to reductions
class Group : public CkReductionMgr
{
	contributorInfo reductionInfo;//My reduction information
 public:
	Group();
	Group(CkMigrateMessage *msg);
	virtual int isNodeGroup() { return 0; }
	virtual void pup(PUP::er &p);
	virtual void flushStates() {
		CkReductionMgr::flushStates();
		reductionInfo.redNo = 0;
 	}
	virtual void CkAddThreadListeners(CthThread tid, void *msg);

	CK_REDUCTION_CONTRIBUTE_METHODS_DECL
        CK_BARRIER_CONTRIBUTE_METHODS_DECL
};

#ifdef _PIPELINED_ALLREDUCE_
class AllreduceMgr
{
public:
	AllreduceMgr() { fragsRecieved=0; size=0; }
	friend class ArrayElement;
	// recieve an allreduce message
	void allreduce_recieve(CkReductionMsg* msg)
	{
		// allred_msgs.enq(msg);
		fragsRecieved++;
		if(fragsRecieved==1)
		{
			data = new char[FRAG_SIZE*msg->nFrags];
		}
		memcpy(data+msg->fragNo*FRAG_SIZE, msg->data, msg->dataSize);
		size += msg->dataSize;
		
		if(fragsRecieved==msg->nFrags) {
			CkReductionMsg* ret = CkReductionMsg::buildNew(size, data);
			cb.send(ret);
			fragsRecieved=0; size=0;
			delete [] data;
		}
		
	}
	// TODO: check for same reduction
	CkCallback cb;	
	int size;
	char* data;
	int fragsRecieved;
	// CkMsgQ<CkReductionMsg> allred_msgs;
};
#endif // _PIPELINED_ALLREDUCE_

#endif //_CKREDUCTION_H
