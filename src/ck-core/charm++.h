/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CHARMPP_H_
#define _CHARMPP_H_

#include <stdlib.h>
#include <memory.h>
#include "charm.h"

#include "middle.h"

class CMessage_CkArgMsg {
public: static int __idx;
};
#define CK_ALIGN(val,to) (((val)+(to)-1)&~((to)-1))

#include "cklists.h"
#include "init.h"
#include "pup.h"

//Implementation class (don't instantiate)
template <class T>
class implCkVecPup : public CkVec<T> {
 protected:
	int pupbase(PUP::er &p) {
		int l=length();
		p(l);
		if (p.isUnpacking()) {
			setSize(l);
			length()=l;
		}
		return l;
	}
};

///A vector of derived types, which must be pupped separately
template <class T>
class CkPupVec : public implCkVecPup<T> {
 public:
	void pup(PUP::er &p) {
		int l=pupbase(p);
		for (int i=0;i<l;i++)
			p|(*this)[i];
	}
	friend void operator|(PUP::er &p,CkPupVec<T> &v) {v.pup(p);}
};

///A vector of basic types, which can be pupped as an array
/// (more restricted, but more efficient version of above)
template <class T>
class CkPupBasicVec : public implCkVecPup<T> {
 public:
	void pup(PUP::er &p) {
		int l=pupbase(p);
		p(getVec(),l);
	}
	friend void operator|(PUP::er &p,CkPupBasicVec<T> &v) {v.pup(p);}
};

///A vector of heap-allocated objects of type T
template <class T>
class CkPupPtrVec : public implCkVecPup<T *> {
 public:
	~CkPupPtrVec() {
		for (int i=0;i<size();i++)
			delete (*this)[i];
	}
	void pup(PUP::er &p) {
		int l=pupbase(p);
		for (int i=0;i<l;i++) {
			if (p.isUnpacking()) (*this)[i]=new T;
			(*this)[i]->pup(p);
		}
	}
	friend void operator|(PUP::er &p,CkPupPtrVec<T> &v) {v.pup(p);}
};

///A vector of pointers-to-subclasses of a PUP::able parent
template <class T>
class CkPupAblePtrVec : public implCkVecPup<T *> {
 public:
	~CkPupAblePtrVec() {
		for (int i=0;i<size();i++)
			delete (*this)[i];
	}
	void pup(PUP::er &p) {
		int l=pupbase(p);
		for (int i=0;i<l;i++)
			p|(*this)[i]; //Pup framework will allocate appropriately
	}
	friend void operator|(PUP::er &p,CkPupAblePtrVec<T> &v) {v.pup(p);}
};


#include "debug-charm.h"

class CkMessage { //Superclass of all Charm++ messages
	//Don't use these: use CkCopyMsg
	CkMessage(const CkMessage &);
	void operator=(const CkMessage &);
public:
	CkMessage() {}
	void operator delete(void *ptr) { CkFreeMsg(ptr); }
	/* This pup routine only packs the message itself, *not* the
	message header.  Use CkPupMessage instead of calling this directly. */
	void pup(PUP::er &p);
        void setImmediate(CmiBool i);
};
class CMessage_CkMessage {
 public:
	static int __idx;
};

class CkArray;

void CkPupMessage(PUP::er &p,void **atMsg,int fast_and_dirty=1);

//This is for passing a single Charm++ message via parameter marshalling
class CkMarshalledMessage {
	void *msg;
	//Don't use these: only pass by reference
	void operator=(const CkMarshalledMessage &);
 public:
	CkMarshalledMessage(void) {msg=NULL;}
	CkMarshalledMessage(CkMessage *m) {msg=m;} //Takes ownership of message
	CkMarshalledMessage(const CkMarshalledMessage &);
	~CkMarshalledMessage() {if (msg) CkFreeMsg(msg);}
	CkMessage *getMessage(void) {void *ret=msg; msg=NULL; return (CkMessage *)ret;}
	void pup(PUP::er &p) {CkPupMessage(p,&msg,1);}
};
PUPmarshall(CkMarshalledMessage);

//A queue-of-messages, like CkMsgQ<CkReductionMsg>
template <class MSG>
class CkMsgQ : public CkQ<MSG *> {
public:
	~CkMsgQ() { //Delete the messages in the queue:
		MSG *m;
		while (NULL!=(m=deq())) delete m;
	}
	void pup(PUP::er &p) {
		int l=length();
		p(l);
		for (int i=0;i<l;i++) {
			MSG *m=NULL;
			if (!p.isUnpacking()) m=deq();
			CkPupMessage(p,(void **)&m);
			enq(m);
		}
	}
	friend void operator|(PUP::er &p,CkMsgQ<MSG> &v) {v.pup(p);}
};

/*******************************************************
Array Index class.  An array index is just a hash key-- 
a run of integers used to look up an object in a hash table.
*/

#include "ckhashtable.h"

#ifndef CK_ARRAYINDEX_MAXLEN 
#define CK_ARRAYINDEX_MAXLEN 3 /*Max. # of integers in an array index*/
#endif

class CkArrayIndex
{
public:
	//Length of index in *integers*
	int nInts;
	
	//Index data immediately follows...
	
	int *data(void) {return (&nInts)+1;}
	const int *data(void) const {return (&nInts)+1;}
	
	void pup(PUP::er &p);

    //These routines allow CkArrayIndex to be used in
    //  a CkHashtableT
	CkHashCode hash(void) const;
	static CkHashCode staticHash(const void *a,size_t);
	int compare(const CkArrayIndex &ind) const;
	static int staticCompare(const void *a,const void *b,size_t);
};

//This class is as large as any CkArrayIndex
class CkArrayIndexMax : public CkArrayIndex {
	struct {
		int data[CK_ARRAYINDEX_MAXLEN];
	} index;
	void copyFrom(const CkArrayIndex &that)
	{
		nInts=that.nInts;
		index=((const CkArrayIndexMax *)&that)->index;
		//for (int i=0;i<nInts;i++) index[i]=that.data()[i];
	}
public:
	CkArrayIndexMax(void) { }
	CkArrayIndexMax(int i) {}   // used for CkVec
	CkArrayIndexMax(const CkArrayIndex &that) 
		{copyFrom(that);}
	CkArrayIndexMax &operator=(const CkArrayIndex &that) 
		{copyFrom(that); return *this;}
        void print() { CmiPrintf("%d: %d %d %d\n", nInts,index.data[0], index.data[1], index.data[2]); }
};

//A layout-compatible version of a CkArrayIndexMax.
//  Needed, e.g., for use in unions where a constructor is forbidden.
class CkArrayIndexStruct {
public:
	int nInts;
	int index[CK_ARRAYINDEX_MAXLEN];
	CkArrayIndexMax &asMax(void) 
		{return *(CkArrayIndexMax *)this;}
	const CkArrayIndexMax &asMax(void) const
		{return *(const CkArrayIndexMax *)this;}
};

class CkArrayID {
	CkGroupID _gid;
public:
	CkArrayID() : _gid() { }
	CkArrayID(CkGroupID g) :_gid(g) {}
	inline void setZero(void) {_gid.setZero();}
	inline int isZero(void) const {return _gid.isZero();}
	operator CkGroupID() const {return _gid;}
	CkArray *ckLocalBranch(void) const
		{ return (CkArray *)CkLocalBranch(_gid); }
	static CkArray *CkLocalBranch(CkArrayID id) 
		{ return (CkArray *)::CkLocalBranch(id); }
	void pup(PUP::er &p) {p | _gid; }
};
PUPmarshall(CkArrayID)

class CkSectionCookie {
public:
  CkArrayID aid;
  int pe;
  void *val;    // point to mCastCookie
  int redNo;
public:
  CkSectionCookie(): pe(-1), val(NULL), redNo(0) {}
  CkSectionCookie(void *p): val(p), redNo(0) { pe = CkMyPe();};
  CkSectionCookie(int e, void *p, int r):  pe(e), val(p), redNo(r) {}
};

class CkSectionID {
public:
  CkSectionCookie _cookie;
  CkArrayIndexMax *_elems;
  int _nElems;
public:
  CkSectionID(): _elems(NULL), _nElems(0) {}
  CkSectionID(const CkSectionID &sid);
  CkSectionID(const CkArrayID &aid, const CkArrayIndexMax *elems, const int nElems);
  void operator=(const CkSectionID &);
  ~CkSectionID();
  void pup(PUP::er &p);
};

#include "ckcallback.h"

//Used by parameter marshalling:
#include "CkMarshall.decl.h"
//This is the message type marshalled parameters get packed into:
class CkMarshallMsg : public CMessage_CkMarshallMsg {
public: 
	char *msgBuf;
};


/********************* Superclass of all Chares ******************/
#if CMK_MULTIPLE_DELETE
#define CHARM_INPLACE_NEW \
    void *operator new(size_t, void *ptr) { return ptr; }; \
    void operator delete(void*, void*) {}; \
    void *operator new(size_t s) { return malloc(s); } \
    void operator delete(void *ptr) { free(ptr); }
#else
#define CHARM_INPLACE_NEW \
    void *operator new(size_t, void *ptr) { return ptr; }; \
    void *operator new(size_t s) { return malloc(s); } \
    void operator delete(void *ptr) { free(ptr); }
#endif

class Chare {
  protected:
    CkChareID thishandle;
  public:
    Chare(CkMigrateMessage *m) {}
    Chare();
    virtual ~Chare(); //<- needed for *any* child to have a virtual destructor
    virtual void pup(PUP::er &p);//<- pack/unpack routine
    inline const CkChareID &ckGetChareID(void) const {return thishandle;}
    inline void CkGetChareID(CkChareID *dest) const {*dest=thishandle;}
    CHARM_INPLACE_NEW
};

///A small piece of a Chare; one that cannot communicate.
class CkComponent {
 public:
	virtual ~CkComponent();
	virtual void pup(PUP::er &p);
};

///A cross-processor handle for a CkComponent.
//Components can currently only live inside groups:
class CkComponentID {
	CkGroupID gid;
	int index; //Index to pass to group's ckLookupComponent method
 public:
	CkComponentID() {index=-1;}
	CkComponentID(int ignored) {index=-1;} //<- stupid; but needed by CkVec
	CkComponentID(const CkGroupID &gid_,int index_)
		:gid(gid_), index(index_) {}
	CkComponent *ckLookup(void) const;
	void pup(PUP::er &p);
};

//Superclass of all Groups that cannot participate in reductions.
//  Undocumented: should only be used inside Charm++.
/*forward*/ class Group;
class IrrGroup : public Chare { 
  protected:
    CkGroupID thisgroup;
    CmiBool ckEnableTracing; //Normally true, except for array manager
  public:
    inline CmiBool ckTracingEnabled(void) {return ckEnableTracing;}

    IrrGroup(CkMigrateMessage *m) { }
    IrrGroup();
    virtual ~IrrGroup(); //<- needed for *any* child to have a virtual destructor

    virtual void pup(PUP::er &p);//<- pack/unpack routine
    inline const CkGroupID &ckGetGroupID(void) const {return thisgroup;}
    inline CkGroupID CkGetGroupID(void) const {return thisgroup;}

    ///Map an index (user-defined meaning) to a component:
    virtual CkComponent *ckLookupComponent(int userIndex);
};

class NodeGroup : public IrrGroup { //Superclass of all NodeGroups
  public:
    CmiNodeLock __nodelock;
    NodeGroup();
    ~NodeGroup();
    inline const CkGroupID &ckGetGroupID(void) const {return thisgroup;}
    inline CkGroupID CkGetNodeGroupID(void) const {return thisgroup;}
    
    virtual void pup(PUP::er &p);
};


/*Macro implmentation of CBase_* */
#define CBase_ProxyDecls(Derived) CProxy_##Derived thisProxy;
#define CBase_ProxyInits(this) thisProxy(this)

/*Templated implementation of CBase_* classes.*/
template <class Parent,class CProxy_Derived>
class CBaseT : public Parent {
public:
	CProxy_Derived thisProxy;
	CBaseT(void) :Parent(), CBase_ProxyInits(this) {}
	CBaseT(CkMigrateMessage *m) :Parent(m), CBase_ProxyInits(this) {}
	void pup(PUP::er &p) {
		Parent::pup(p);
	}
};

/*Templated version of above for multiple (at least duplicate) inheritance:*/
template <class Parent1,class Parent2,class CProxy_Derived>
class CBaseT2 : public Parent1, public Parent2 {
public:
	CProxy_Derived thisProxy;
	CBaseT2(void) :Parent1(), Parent2(), 
		CBase_ProxyInits((Parent1 *)this) {}
	CBaseT2(CkMigrateMessage *m) :Parent1(m), Parent2(m),
		CBase_ProxyInits((Parent1 *)this) {} 
	void pup(PUP::er &p) {
		Parent1::pup(p);
		Parent2::pup(p);
	}

//These overloads are needed to prevent ambiguity for multiple inheritance:
	inline const CkChareID &ckGetChareID(void) const 
		{return ((Parent1 *)this)->ckGetChareID();}
	CHARM_INPLACE_NEW
};


/**************************** Proxies **************************/

/*Message delegation support, where you send a message via
a proxy normally, but the message ends up routed via a 
special delegateMgr group.
*/
class CkDelegateMgr;

/*
class CProxy {
  private:
    CkGroupID delegatedTo;
  protected: //Never allocate CProxy's-- only subclass them.
    CProxy() { delegatedTo.pe = -1; }
    CProxy(CkGroupID dTo) { delegatedTo = dTo; }
  public:
    void ckDelegate(CkGroupID to) {delegatedTo=to;}
    void ckUndelegate(void) {delegatedTo.pe =-1;}
    int ckIsDelegated(void) const {return (delegatedTo.pe != -1);}
    CkGroupID ckDelegatedIdx(void) const {return delegatedTo;}
    CkDelegateMgr *ckDelegatedTo(void) const {
    	return (CkDelegateMgr *)CkLocalBranch(delegatedTo);
    }
    void pup(PUP::er &p) {
      p|delegatedTo;
    }
};
*/

// changed CProxy

class CProxy {
  private:
    CkGroupID delegatedTo;
  protected: //Never allocate CProxy's-- only subclass them.
    CProxy() { /*delegatedTo.idx = -1;*/ delegatedTo.setZero(); }
    CProxy(CkGroupID dTo) { delegatedTo = dTo; }
  public:
    void ckDelegate(CkGroupID to) {delegatedTo=to;}
    void ckUndelegate(void) {/*delegatedTo.idx = -1;*/ delegatedTo.setZero();}
    int ckIsDelegated(void) const {/*return (delegatedTo.idx != -1);*/ return(!delegatedTo.isZero());}
    CkGroupID ckDelegatedIdx(void) const {return delegatedTo;}
    CkDelegateMgr *ckDelegatedTo(void) const {
    	return (CkDelegateMgr *)CkLocalBranch(delegatedTo);
    }
    void pup(PUP::er &p) {
      p|delegatedTo;
    }
};

PUPmarshall(CProxy)

/*These disambiguation macros are needed to support
  multiple inheritance in Chares (Groups, Arrays).
  They resolve ambiguous accessor calls to the parent "super".
  Because mutator routines need to change *all* the base
  classes, mutators are generated in xi-symbol.C.
*/
#define CK_DISAMBIG_CPROXY(super) \
    int ckIsDelegated(void) const {return super::ckIsDelegated();}\
    CkGroupID ckDelegatedIdx(void) const {return super::ckDelegatedIdx();}\
    CkDelegateMgr *ckDelegatedTo(void) const {\
    	return super::ckDelegatedTo();\
    }\



/*The base classes of each proxy type
*/
class CProxy_Chare : public CProxy {
  private:
    CkChareID _ck_cid;
  public:
    CProxy_Chare() { 
#ifndef CMK_OPTIMIZE
	_ck_cid.onPE=0; _ck_cid.objPtr=0;
#endif
    }
#ifndef CMK_OPTIMIZE
    inline void ckCheck(void) const  {   //Make sure this proxy has a value
	if (_ck_cid.objPtr==0)
		CkAbort("Error! This chare proxy has not been initialized!");
    }
#else
    inline void ckCheck() const {}
#endif
    CProxy_Chare(const CkChareID &c) : _ck_cid(c) {}
    CProxy_Chare(const Chare *c) : _ck_cid(c->ckGetChareID()) {}
    const CkChareID &ckGetChareID(void) const {return _ck_cid;}
    operator const CkChareID &(void) const {return ckGetChareID();}
    void ckSetChareID(const CkChareID &c) {_ck_cid=c;}
    void pup(PUP::er &p) {
    	CProxy::pup(p);
    	p(_ck_cid.onPE);
    	//Copy the pointer as straight bytes
    	p((void *)&_ck_cid.objPtr,sizeof(_ck_cid.objPtr));
    }
};
PUPmarshall(CProxy_Chare)

#define CK_DISAMBIG_CHARE(super) \
	CK_DISAMBIG_CPROXY(super) \
	inline void ckCheck(void) const {super::ckCheck();} \
	const CkChareID &ckGetChareID(void) const\
    	   {return super::ckGetChareID();} \
        operator const CkChareID &(void) const {return ckGetChareID();}

/******************* Reduction Declarations ****************/
//Silly: need the type of a reduction client here so it can be used by proxies.
//A clientFn is called on PE 0 when all contributions
// have been received and reduced.
//  param can be ignored, or used to pass any client-specific data you $
//  dataSize gives the size (in bytes) of the data array
//  data gives the reduced contributions--
//       it will be disposed of after this procedure returns.
typedef void (*CkReductionClientFn)(void *param,int dataSize,void *data);

/// Tiny utility class used by CkReductionClientAdaptor--
/// lets us keep backward compatability with the old C-style interface.
class CkReductionClientBundle : public CkCallback {
	CkReductionClientFn fn;
	void *param;
	static void callbackCfn(void *thisPtr,void *reductionMsg);
 public:
	CkReductionClientBundle(CkReductionClientFn fn_,void *param_);
};

#define CK_REDUCTION_CLIENT_DECL \
	void setReductionClient(CkReductionClientFn fn,void *param=NULL) const\
		{ ckSetReductionClient(fn,param); } \
	void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const \
		{ ckSetReductionClient(new CkReductionClientBundle(fn,param)); } \
	void ckSetReductionClient(CkCallback *cb) const;\

#define CK_REDUCTION_CLIENT_DEF(className,mgr) \
 	void className::ckSetReductionClient(CkCallback *cb) const \
		{ (mgr)->ckSetReductionClient(cb); }\

#define CK_REDUCTION_CLIENT_DISAMBIG(super) \
	inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const \
		{ super::setReductionClient(fn,param); } \
	inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const \
		{ super::ckSetReductionClient(fn,param); } \
	inline void ckSetReductionClient(CkCallback *cb) const \
		{ super::ckSetReductionClient(cb); }\

class CProxy_Group : public CProxy {
  private:
    CkGroupID _ck_gid;
  public:
    CProxy_Group() { 
#ifndef CMK_OPTIMIZE
	_ck_gid.setZero();
#endif
    }
    CProxy_Group(CkGroupID g) 
    	:CProxy(),_ck_gid(g) {}
    CProxy_Group(CkGroupID g,CkGroupID dTo) 
    	:CProxy(dTo),_ck_gid(g) {}
    CProxy_Group(const IrrGroup *g) 
        :CProxy(), _ck_gid(g->ckGetGroupID()) {}
    CProxy_Group(const NodeGroup *g)  //<- for compatability with NodeGroups
        :CProxy(), _ck_gid(g->ckGetGroupID()) {}

#ifndef CMK_OPTIMIZE
    inline void ckCheck(void) const {   //Make sure this proxy has a value
	if (_ck_gid.isZero())
		CkAbort("Error! This group proxy has not been initialized!");
    }
#else
    inline void ckCheck() const {}
#endif

    CkChareID ckGetChareID(void) const { 
    	CkChareID ret;
    	ret.onPE=CkMyPe();
    	ret.objPtr=CkLocalBranch(_ck_gid);
    	return ret; 
    }
    CkGroupID ckGetGroupID(void) const {return _ck_gid;}
    operator CkGroupID () const {return ckGetGroupID();}
    void ckSetGroupID(CkGroupID g) {_ck_gid=g;}
    void pup(PUP::er &p) {
    	CProxy::pup(p);
	p | _ck_gid;
    }
    CK_REDUCTION_CLIENT_DECL
};
PUPmarshall(CProxy_Group)
#define CK_DISAMBIG_GROUP(super) \
	CK_DISAMBIG_CPROXY(super) \
	inline void ckCheck(void) const {super::ckCheck();} \
	CkChareID ckGetChareID(void) const \
	   {return super::ckGetChareID();} \
	CkGroupID ckGetGroupID(void) const \
	   {return super::ckGetGroupID();} \
	operator CkGroupID () const { return ckGetGroupID(); } \
	CK_REDUCTION_CLIENT_DISAMBIG(super)\


class CProxyElement_Group : public CProxy_Group {
  private:
    int _onPE;    
  public:
    CProxyElement_Group() { }
    CProxyElement_Group(CkGroupID g,int onPE)
	: CProxy_Group(g),_onPE(onPE) {}
    CProxyElement_Group(CkGroupID g,int onPE,CkGroupID dTo)
	: CProxy_Group(g,dTo),_onPE(onPE) {}
    CProxyElement_Group(const IrrGroup *g) 
        :CProxy_Group(g), _onPE(CkMyPe()) {}
    CProxyElement_Group(const NodeGroup *g)  //<- for compatability with NodeGroups
        :CProxy_Group(g), _onPE(CkMyPe()) {}
    
    int ckGetGroupPe(void) const {return _onPE;}
    void pup(PUP::er &p) {
    	CProxy_Group::pup(p);
    	p(_onPE);
    }
};
PUPmarshall(CProxyElement_Group)
#define CK_DISAMBIG_GROUP_ELEMENT(super) \
	CK_DISAMBIG_GROUP(super) \
	int ckGetGroupPe(void) const\
    	   {return super::ckGetGroupPe();} \


/* These classes exist to provide chare indices for the basic
 chare types.*/
class CkIndex_Chare { public:
    static int __idx;//Fake chare index for registration
};
class CkIndex_ArrayBase { public:
    static int __idx;//Fake chare index for registration
};
class CkIndex_Group { public:
    static int __idx;//Fake chare index for registration
};

typedef CkIndex_Group CkIndex_NodeGroup;
typedef CkIndex_Group CkIndex_IrrGroup;
typedef CProxy_Group CProxy_NodeGroup;
typedef CProxy_Group CProxy_IrrGroup;
typedef CProxyElement_Group CProxyElement_NodeGroup;
typedef CProxyElement_Group CProxyElement_IrrGroup;
typedef CkGroupID CkNodeGroupID;

//(CProxy_ArrayBase is defined in ckarray.h)

//an "interface" class-- all delegated messages are routed via a DelegateMgr.  
// The default action is to deliver the message directly.
class CkDelegateMgr : public IrrGroup {
  public:
    virtual ~CkDelegateMgr(); //<- so children can have virtual destructor
    virtual void ChareSend(int ep,void *m,const CkChareID *c,int onPE);

    virtual void GroupSend(int ep,void *m,int onPE,CkGroupID g);
    virtual void GroupBroadcast(int ep,void *m,CkGroupID g);

    virtual void NodeGroupSend(int ep,void *m,int onNode,CkNodeGroupID g);
    virtual void NodeGroupBroadcast(int ep,void *m,CkNodeGroupID g);

    virtual void ArrayCreate(int ep,void *m,const CkArrayIndexMax &idx,int onPE,CkArrayID a);
    virtual void ArraySend(int ep,void *m,const CkArrayIndexMax &idx,CkArrayID a);
    virtual void ArrayBroadcast(int ep,void *m,CkArrayID a);
    virtual void ArraySectionSend(int ep,void *m,CkArrayID a,CkSectionCookie &s);
};

//Defines the actual "Group"
#include "ckreduction.h"

class CkQdMsg {
  public:
    void *operator new(size_t s) { return CkAllocMsg(0,s,0); }
    void operator delete(void* ptr) { CkFreeMsg(ptr); }
    static void *alloc(int, size_t s, int*, int) {
      return CkAllocMsg(0,s,0);
    }
    static void *pack(CkQdMsg *m) { return (void*) m; }
    static CkQdMsg *unpack(void *buf) { return (CkQdMsg*) buf; }
};

class CkThrCallArg {
  public:
    void *msg;
    void *obj;
    CkThrCallArg(void *m, void *o) : msg(m), obj(o) {}
};

extern void _REGISTER_BASE(int didx, int bidx);
extern void _REGISTER_DONE(void);

static inline void _CHECK_CID(CkChareID, int){}

#include "readonly.h"
#include "ckarray.h"
#include "ckstream.h"
#include "CkFutures.decl.h"
#include "charisma.h"
#include "tempo.h"
#include "waitqd.h"
#include "sdag.h"

#endif



