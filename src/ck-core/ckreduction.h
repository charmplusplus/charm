/*
Charm++ File: Reduction Library
added 3/27/2000 by Orion Sky Lawlor, olawlor@acm.org

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

class CkReductionMsg; //See definition below

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
	CkGroupInitCallback(CkMigrateMessage *m) {}
	void callMeBack(CkGroupCallbackMsg *m);
};


class CkGroupReadyCallback : public IrrGroup {
private:
  int _isReady;
  CkQ<CkGroupCallbackMsg *> _msgs;
  void callBuffered(void);
public:
	CkGroupReadyCallback(void);
	CkGroupReadyCallback(CkMigrateMessage *m) {}
	void callMeBack(CkGroupCallbackMsg *m);
	int isReady(void) { return _isReady; }
protected:
	void setReady(void) {_isReady = 1; callBuffered(); }
	void setNotReady(void) {_isReady = 0; }
};

class CkReductionMsg;


class CkReductionNumberMsg;

/**
 * One CkReductionMgr runs a non-overlapping set of reductions.
 * It collects messages from all local contributors, then sends
 * the reduced message up the reduction tree to node zero, where
 * they're passed to the user's client function.
 */
class CkReductionMgr : public CkGroupInitCallback {
        CProxy_CkReductionMgr thisProxy;
public:
	CkReductionMgr(void);
	CkReductionMgr(CkMigrateMessage *m) :CkGroupInitCallback(m) {}
	
	typedef CkReductionClientFn clientFn;

	/**
	 * Add the given client function.  Overwrites any previous client.
	 * This manager will dispose of the callback when replaced or done.
	 */
	void ckSetReductionClient(CkCallback *cb);

//Contributors keep a copy of this structure:
	class contributorInfo {
	public:
		int redNo;//Current reduction number
		contributorInfo() {redNo=0;}
		//Migration utilities:
		void pup(PUP::er &p);
	};
	
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
	//Sent up the reduction tree with reduced data
	void RecvMsg(CkReductionMsg *m);
	//Sent to root of the reduction tree with late migrant data
	void LateMigrantMsg(CkReductionMsg *m);
	//A late migrating contributor will never contribute
	void MigrantDied(CkReductionNumberMsg *m);

private:
//Data members
	//Stored callback function (may be NULL if none has been set)
	CkCallback *storedCallback;
	
	int redNo;//Number of current reduction (incremented at end)
	CmiBool inProgress;//Is a reduction started, but not complete?
	CmiBool creating;//Are elements still being created?
	CmiBool startRequested;//Should we start the next reduction when creation finished?
	int gcount;//=el't created here - el't deleted here
	int lcount;//Number of local contributors
	
	//Current local and remote contributions
	int nContrib,nRemote;
	//Contributions queued for the current reduction
	CkVec<CkReductionMsg *> msgs;
	
	//Contributions queued for future reductions (sent to us too early)
	CkQ<CkReductionMsg *> futureMsgs;
	//Remote messages queued for future reductions (sent to us too early)
	CkQ<CkReductionMsg *> futureRemoteMsgs;

//State:
	void startReduction(int number);
	void addContribution(CkReductionMsg *m);
	void finishReduction(void);
	
//Reduction tree utilities
	enum {TREE_WID=4};
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

	class countAdjustment {
	public:
		int gcount;//Adjustment to global count (applied at reduction end)
		int lcount;//Adjustment to local count (applied continually)
		countAdjustment(int ignored=0) {gcount=lcount=0;}
	};
	
	//This vector of adjustments is indexed by redNo,
	// starting from the current redNo.
	CkVec<countAdjustment> adjVec;
	//Return the countAdjustment struct for the given redNo:
	countAdjustment &adj(int number);
	//Shift the list of countAdjustments down
	void shiftAdjVec(void);

//Checkpointing utilities
	//pack-unpack method for CkReductionMsg
        CkReductionMsg* pupCkReductionMsg(CkReductionMsg *m, PUP::er &p);
        void pupMsgVector(CkVec<CkReductionMsg *> &msgs, PUP::er &p);
        void pupMsgQ(CkQ<CkReductionMsg *> &msgs, PUP::er &p);
        void pupAdjVec(CkVec<countAdjustment> &vec, PUP::er &p);
 public:
	virtual void pup(PUP::er &p);
};


//A CkReductionMsg is sent up the reduction tree-- it
// carries a contribution, or several reduced contributions.
class CkReductionMsg : public CMessage_CkReductionMsg
{
	friend class CkReduction;
	friend class CkReductionMgr;
public:

//Publically-accessible fields:	
	//"Constructor"-- builds and returns a new CkReductionMsg.
	//  the "srcData" array you specify will be copied into this object (unless NULL).
	static CkReductionMsg *buildNew(int NdataSize,const void *srcData,
		CkReduction::reducerType reducer=CkReduction::invalid);
	
	inline int getLength(void) const {return dataSize;}
	inline int getSize(void) const {return dataSize;}
	inline void *getData(void) {return data;}
	
	inline int getUserFlag(void) const {return userFlag;}
	inline void setUserFlag(int f) { userFlag=f;}
	
	inline void setCallback(const CkCallback &cb) { callback=cb; }

	//Return true if this message came straight from a contribute call--
	// if it didn't come from a previous reduction function.
	inline int isFromUser(void) const {return sourceFlag==-1;}

//Implementation-only fields (don't access these directly!)
	//Msg runtime support
	static void *alloc(int msgnum, size_t size, int *reqSize, int priobits);
	static void *pack(CkReductionMsg *);
	static CkReductionMsg *unpack(void *in);
	
private:
	int dataSize;//Length of array below, in bytes
	void *data;//Reduction data
	int userFlag; //Some sort of identifying flag, for client use
	CkCallback callback; //What to do when done
	
	int sourceFlag;/*Flag:
		0 indicates this is a placeholder message (meaning: nothing to report)
		-1 indicates this is a single (non-reduced) contribution.
  		>0 indicates this is a reduced contribution.
  	*/
  	int nSources(void) {return abs(sourceFlag);}

private:
	CkReduction::reducerType reducer;
	CkReductionMgr::contributorInfo *ci;//Source contributor, or NULL if none
	int redNo;//The serial number of this reduction
	int gcount;//Contribution to the global contributor count
	double dataStorage;//Start of data array (so it's double-aligned)
	
	//Default constructor is private so you must use "buildNew", above
	CkReductionMsg();
};


//Define methods used to contribute to the given reduction type.
//  Data is copied, not deleted.
#define CK_REDUCTION_CONTRIBUTE_METHODS_DECL \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	int userFlag=-1); \
  void contribute(int dataSize,const void *data,CkReduction::reducerType type, \
	const CkCallback &cb,int userFlag=-1); \
  void contribute(CkReductionMsg *msg);\

#define CK_REDUCTION_CONTRIBUTE_METHODS_DEF(me,myRednMgr,myRednInfo) \
void me::contribute(int dataSize,const void *data,CkReduction::reducerType type,\
	int userFlag)\
{\
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);\
	msg->setUserFlag(userFlag);\
	myRednMgr->contribute(&myRednInfo,msg);\
}\
void me::contribute(int dataSize,const void *data,CkReduction::reducerType type,\
	const CkCallback &cb,int userFlag)\
{\
	CkReductionMsg *msg=CkReductionMsg::buildNew(dataSize,data,type);\
	msg->setUserFlag(userFlag);\
	msg->setCallback(cb);\
	myRednMgr->contribute(&myRednInfo,msg);\
}\
void me::contribute(CkReductionMsg *msg) \
	{myRednMgr->contribute(&myRednInfo,msg);}\


//A group that can contribute to reductions
class Group : public CkReductionMgr
{
	CkReductionMgr::contributorInfo reductionInfo;//My reduction information
 public:
	Group();
	Group(CkMigrateMessage *msg):CkReductionMgr(msg) {}
	virtual void pup(PUP::er &p);

	CK_REDUCTION_CONTRIBUTE_METHODS_DECL
};

#endif //_CKREDUCTION_H
