#ifndef _CKREDUCTION_MGR_H
#define _CKREDUCTION_MGR_H


#include "pup.h"
#include "charm.h"
#include "ckcallback.h"

class CkReductionMsg;
class CkReductionNumberMsg;
class CkReductionInactiveMsg;

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
  countAdjustment(int ignored=0) {(void)ignored; gcount=0; lcount=0;}
  void pup(PUP::er& p){ p|gcount; p|lcount; }
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




class CkReductionMgr : public CkGroupInitCallback {
public:
	CkReductionMgr();
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

//Data members
	//Stored callback function (may be NULL if none has been set)
	CkCallback storedCallback;

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
	bool hasParent(void);
	int treeParent(void);//My parent PE
	int firstKid(void);//My first child PE
	int treeKids(void);//Number of children in tree

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
	static bool isIrreducible(){ return false;}
	void contributeViaMessage(CkReductionMsg *m);
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
        tuple
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
	friend class CkMulticastMgr;
    friend class ck::impl::XArraySectionReducer;
//System-level interface

	//Reducer table: maps reducerTypes to reducerFns.
    static std::vector<reducerStruct>& reducerTable();
    static std::vector<reducerStruct> initReducerTable();

    // tupleReduction needs access to the reducerTable that lives in this namespace
    // so it is not a standalone function in ckreduction.C like other reduction implementations
    static CkReductionMsg* tupleReduction(int nMsgs, CkReductionMsg** msgs);

	//Don't instantiate a CkReduction object-- it's just a namespace.
	CkReduction();
};
PUPbytes(CkReduction::reducerType)


#endif
