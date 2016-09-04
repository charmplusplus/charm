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

class CkGroupReadyCallback : public IrrGroup {
private:
  bool _isReady;
  CkQ<CkGroupCallbackMsg *> _msgs;
  void callBuffered(void);
public:
	CkGroupReadyCallback(void);
	CkGroupReadyCallback(CkMigrateMessage *m):IrrGroup(m) {}
	void callMeBack(CkGroupCallbackMsg *m);
	bool isReady(void) { return _isReady; }
protected:
	void setReady(void) {_isReady = true; callBuffered(); }
	void setNotReady(void) {_isReady = false; }
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



//A CkReductionMsg is sent up the reduction tree-- it
// carries a contribution, or several reduced contributions.
class CkReductionMsg : public CMessage_CkReductionMsg
{
	friend class CkReduction;
	friend class CkReductionMgr;
	friend class CkNodeReductionMgr;
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
	inline bool isFromUser(void) const {return sourceFlag==-1;}

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

	//Default constructor is private so you must use "buildNew", above
    CkReductionMsg();
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
