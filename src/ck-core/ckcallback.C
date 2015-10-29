/*
A CkCallback is a simple way for a library to return data 
to a wide variety of user code, without the library having
to handle all 17 possible cases.

This object is implemented as a union, so the entire object
can be sent as bytes.  Another option would be to use a virtual 
"send" method.

Initial version by Orion Sky Lawlor, olawlor@acm.org, 2/8/2002
*/
#include "charm++.h"
#include "ckcallback-ccs.h"
#include "CkCallback.decl.h"
#include "envelope.h"
/*readonly*/ CProxy_ckcallback_group _ckcallbackgroup;

typedef CkHashtableT<CkHashtableAdaptorT<unsigned int>, CkCallback*> threadCB_t;
CpvStaticDeclare(threadCB_t, threadCBs);
CpvStaticDeclare(unsigned int, nextThreadCB);

//This main chare is only used to create the callback forwarding group
class ckcallback_main : public CBase_ckcallback_main {
public:
	ckcallback_main(CkArgMsg *m) {
		_ckcallbackgroup=CProxy_ckcallback_group::ckNew();
		delete m;
	}
};

//The callback group is used to forward a callback to the processor
// it originated from.
class ckcallback_group : public CBase_ckcallback_group {
public:
	ckcallback_group() { /*empty*/ }
	ckcallback_group(CkMigrateMessage *m) { /*empty*/ }
	void registerCcsCallback(const char *name,const CkCallback &cb);
	void call(CkCallback &c,CkMarshalledMessage &msg) {
		c.send(msg.getMessage());
	}
};

/*************** CkCallback implementation ***************/
//Initialize the callback's thread fields before sending it off:
void CkCallback::impl_thread_init(void)
{
    int exist;
    CkCallback **cb;
    d.thread.onPE=CkMyPe();
	do {
	  if (CpvAccess(nextThreadCB)==0) CpvAccess(nextThreadCB)=1;
	  d.thread.cb=CpvAccess(nextThreadCB)++;
	  cb = &CpvAccess(threadCBs).put(d.thread.cb, &exist);
	} while (exist==1);
	*cb = this; //<- so we can find this structure later
	d.thread.th=NULL; //<- thread isn't suspended yet
	d.thread.ret=(void*)-1;//<- no data to return yet
}

//Actually suspend this thread
void *CkCallback::impl_thread_delay(void) const
{
	if (type!=resumeThread) 
		CkAbort("Called impl_thread_delay on non-threaded callback");
	if (CkMyPe()!=d.thread.onPE)
		CkAbort("Called thread_delay on different processor than where callback was created");
	
	//Find the original callback object:
	CkCallback *dest=(CkCallback *)this;
	if (d.thread.cb!=0) dest=CpvAccess(threadCBs).get(d.thread.cb);
	if (dest==0)
	    CkAbort("Called thread_delay on an already deleted callback");
	if (dest->d.thread.ret==(void*)-1) 
	{  //We need to sleep for the result:
		dest->d.thread.th=CthSelf(); //<- so we know a thread is waiting
		CthSuspend();
		if (dest->d.thread.ret==(void*)-1) 
			CkAbort("thread resumed, but callback data is still empty");
	}
	return dest->d.thread.ret;
}


/*These can't be defined in the .h file like the other constructors
 * because we need CkCallback before CProxyElement* are defined.
 */
CkCallback::CkCallback(Chare *p, int ep, bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendChare:sendChare;
	d.chare.ep=ep; 
	d.chare.id=p->ckGetChareID();
        d.chare.hasRefnum= false;
        d.chare.refnum = 0;
}
CkCallback::CkCallback(Group *p, int ep, bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendGroup:sendGroup;
	d.group.ep=ep; d.group.id=p->ckGetGroupID(); d.group.onPE=CkMyPe();
        d.group.hasRefnum= false;
        d.group.refnum = 0;
}
CkCallback::CkCallback(NodeGroup *p, int ep, bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendNodeGroup:sendNodeGroup;
	d.group.ep=ep; d.group.id=p->ckGetGroupID(); d.group.onPE=CkMyNode();
        d.group.hasRefnum= false;
        d.group.refnum = 0;
}

CkCallback::CkCallback(int ep,const CProxy_NodeGroup &ngp) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=bcastNodeGroup;
	d.group.ep=ep; d.group.id=ngp.ckGetGroupID();
        d.group.hasRefnum= false;
        d.group.refnum = 0;
}

CkCallback::CkCallback(int ep,int onPE,const CProxy_NodeGroup &ngp,bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendNodeGroup:sendNodeGroup;
	d.group.ep=ep; d.group.id=ngp.ckGetGroupID(); d.group.onPE=onPE;
        d.group.hasRefnum= false;
        d.group.refnum = 0;
}

CkCallback::CkCallback(int ep,const CProxyElement_Group &grpElt,bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendGroup:sendGroup;
	d.group.ep=ep; 
	d.group.id=grpElt.ckGetGroupID(); 
	d.group.onPE=grpElt.ckGetGroupPe();
        d.group.hasRefnum= false;
        d.group.refnum = 0;
}
CkCallback::CkCallback(int ep,const CProxyElement_ArrayBase &arrElt,bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendArray:sendArray;
	d.array.ep=ep; 
	d.array.id=arrElt.ckGetArrayID(); 
	d.array.idx = arrElt.ckGetIndex();
        d.array.hasRefnum= false;
        d.array.refnum = 0;
}

CkCallback::CkCallback(int ep,CProxySection_ArrayBase &sectElt,bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=bcastSection;
      d.section.ep=ep; 
      CkSectionID secID=sectElt.ckGetSectionID(0); 
      d.section.sinfo = secID._cookie.info;
      d.section._elems = secID._elems;
      d.section._nElems = secID._nElems;
      d.section.pelist = secID.pelist;
      d.section.npes = secID.npes;
      secID._elems = NULL;
      secID.pelist = NULL;
      d.section.hasRefnum = false;
      d.section.refnum = 0;
}

CkCallback::CkCallback(int ep, CkSectionID &id) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=bcastSection;
      d.section.ep=ep;
      d.section.sinfo = id._cookie.info;
      d.section._elems = id._elems;
      d.section._nElems = id._nElems;
      d.section.pelist = id.pelist;
      d.section.npes = id.npes;
}

CkCallback::CkCallback(ArrayElement *p, int ep,bool doInline) {
#if CMK_ERROR_CHECKING
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendArray:sendArray;
    d.array.ep=ep; 
	d.array.id=p->ckGetArrayID(); 
	d.array.idx = p->ckGetArrayIndex();
        d.array.hasRefnum= false;
        d.array.refnum = 0;
}


void CkCallback::send(int length,const void *data) const
{
	send(CkDataMsg::buildNew(length,data));
}

/*Libraries should call this from their "done" entry points.
  It takes the given message and handles it appropriately.
  After the send(), this callback is finished and cannot be reused.
*/
void CkCallback::send(void *msg) const
{
	switch(type) {
	  //	CkPrintf("type:%d\n",type);
	case ignore: //Just ignore the callback
		if (msg) CkFreeMsg(msg);
		break;
	case ckExit: //Call ckExit
		if (msg) CkFreeMsg(msg);
		CkExit();
		break;
	case resumeThread: //Resume a waiting thread
		if (d.thread.onPE==CkMyPe()) {
			CkCallback *dest=CpvAccess(threadCBs).get(d.thread.cb);
			if (dest==0 || dest->d.thread.ret!=(void*)-1)
				CkAbort("Already sent a value to this callback!\n");
			dest->d.thread.ret=msg; //<- return data
			if (dest->d.thread.th!=NULL)
				CthAwaken(dest->d.thread.th);
		} 
		else //Forward message to processor where the thread actually lives
			_ckcallbackgroup[d.thread.onPE].call(*this,(CkMessage *)msg);
		break;
	case call1Fn: //Call a C function pointer on the current processor
		(d.c1fn.fn)(msg);
		break;
	case callCFn: //Call a C function pointer on the appropriate processor
		if (d.cfn.onPE==CkMyPe())
			(d.cfn.fn)(d.cfn.param,msg);
		else
			_ckcallbackgroup[d.cfn.onPE].call(*this,(CkMessage *)msg);
		break;
	case sendChare: //Send message to a chare
		if (!msg) msg=CkAllocSysMsg();
                if (d.chare.hasRefnum) CkSetRefNum(msg, d.chare.refnum);
		CkSendMsg(d.chare.ep,msg,&d.chare.id);
		break;
	case isendChare: //inline send-to-chare
		if (!msg) msg=CkAllocSysMsg();
                if (d.chare.hasRefnum) CkSetRefNum(msg, d.chare.refnum);
		CkSendMsgInline(d.chare.ep,msg,&d.chare.id);
		break;
	case sendGroup: //Send message to a group element
		if (!msg) msg=CkAllocSysMsg();
                if (d.group.hasRefnum) CkSetRefNum(msg, d.group.refnum);
		CkSendMsgBranch(d.group.ep,msg,d.group.onPE,d.group.id);
		break;
	case sendNodeGroup: //Send message to a group element
		if (!msg) msg=CkAllocSysMsg();
                if (d.group.hasRefnum) CkSetRefNum(msg, d.group.refnum);
		CkSendMsgNodeBranch(d.group.ep,msg,d.group.onPE,d.group.id);
		break;
	case isendGroup: //inline send-to-group element
		if (!msg) msg=CkAllocSysMsg();
                if (d.group.hasRefnum) CkSetRefNum(msg, d.group.refnum);
		CkSendMsgBranchInline(d.group.ep,msg,d.group.onPE,d.group.id);
		break;
	case isendNodeGroup: //inline send-to-group element
		if (!msg) msg=CkAllocSysMsg();
                if (d.group.hasRefnum) CkSetRefNum(msg, d.group.refnum);
		CkSendMsgNodeBranchInline(d.group.ep,msg,d.group.onPE,d.group.id);
		break;
	case sendArray: //Send message to an array element
		if (!msg) msg=CkAllocSysMsg();
                if (d.array.hasRefnum) CkSetRefNum(msg, d.array.refnum);

		CkSetMsgArrayIfNotThere(msg);
		CkSendMsgArray(d.array.ep,msg,d.array.id,d.array.idx.asChild());
		break;
	case isendArray: //inline send-to-array element
		if (!msg) msg=CkAllocSysMsg();
                if (d.array.hasRefnum) CkSetRefNum(msg, d.array.refnum);
		CkSendMsgArrayInline(d.array.ep,msg,d.array.id,d.array.idx.asChild());
		break;
	case bcastGroup:
		if (!msg) msg=CkAllocSysMsg();
                if (d.group.hasRefnum) CkSetRefNum(msg, d.group.refnum);
		CkBroadcastMsgBranch(d.group.ep,msg,d.group.id);
		break;
	case bcastNodeGroup:
		if (!msg) msg=CkAllocSysMsg();
                if (d.group.hasRefnum) CkSetRefNum(msg, d.group.refnum);
		CkBroadcastMsgNodeBranch(d.group.ep,msg,d.group.id);
		break;
	case bcastArray:
		if (!msg) msg=CkAllocSysMsg();
                if (d.array.hasRefnum) CkSetRefNum(msg, d.array.refnum);
		CkBroadcastMsgArray(d.array.ep,msg,d.array.id);
		break;
	case bcastSection: {
		if(!msg)msg=CkAllocSysMsg();
                if (d.section.hasRefnum) CkSetRefNum(msg, d.section.refnum);
                CkSectionInfo sinfo(d.section.sinfo);
                CkSectionID secID(sinfo, d.section._elems, d.section._nElems, d.section.pelist, d.section.npes);
		CkBroadcastMsgSection(d.section.ep,msg,secID);
                secID._elems = NULL;
                secID.pelist = NULL;
		break;
             }
	case replyCCS: { /* Send CkDataMsg as a CCS reply */
		void *data=NULL;
		int length=0;
		if (msg) {
			CkDataMsg *m=(CkDataMsg *)msg;
			m->check();
			data=m->getData();
			length=m->getLength();
		}
		CcsSendDelayedReply(d.ccsReply.reply,length,data);
		if (msg) CkFreeMsg(msg);
		} break;
	case invalid: //Uninitialized
		CmiAbort("Called send on uninitialized callback");
		break;
	default: //Out-of-bounds type code
		CmiAbort("Called send on corrupted callback");
		break;
	};
}

void CkCallback::pup(PUP::er &p) {
  //p((char*)this, sizeof(CkCallback));
  int t = (int)type;
  p|t;
  type = (callbackType)t;
  switch (type) {
  case resumeThread:
    p|d.thread.onPE;
    p|d.thread.cb;
    break;
  case isendChare:
  case sendChare:
    p|d.chare.ep;
    p|d.chare.id;
    p|d.chare.hasRefnum;
    p|d.chare.refnum;
    break;
  case isendGroup:
  case sendGroup:
  case isendNodeGroup:
  case sendNodeGroup:
    p|d.group.onPE;
    p|d.group.hasRefnum;
    p|d.group.refnum;
  case bcastNodeGroup:
  case bcastGroup:
    p|d.group.ep;
    p|d.group.id;
    p|d.group.hasRefnum;
    p|d.group.refnum;
    break;
  case isendArray:
  case sendArray:
    p|d.array.idx;
    p|d.array.hasRefnum;
    p|d.array.refnum;
  case bcastArray:
    p|d.array.ep;
    p|d.array.id;
    p|d.array.hasRefnum;
    p|d.array.refnum;
    break;
  case replyCCS:
    p((char*)&d.ccsReply.reply, sizeof(d.ccsReply.reply));
    break;
  case call1Fn:
    p((char*)&d.c1fn, sizeof(d.c1fn));
    break;
  case callCFn:
    p((char*)&d.cfn, sizeof(d.cfn));
    break;
  case ignore:
  case ckExit:
  case invalid:
    break;
  default:
    CkAbort("Inconsistent CkCallback type");
  }
}

bool CkCallback::containsPointer() const {
  switch(type) {
  case invalid:
  case ignore:
  case ckExit:
  case sendGroup:
  case sendNodeGroup:
  case sendArray:
  case isendGroup:
  case isendNodeGroup:
  case isendArray:
  case bcastGroup:
  case bcastNodeGroup:
  case bcastArray:
    return false;

  case resumeThread:
  case callCFn:
  case call1Fn:
  case replyCCS:
  case bcastSection:
    return true;

  case sendChare:
  case isendChare:
#if CMK_CHARE_USE_PTR
    return true;
#else
    return false;
#endif

  default:
    CkAbort("Asked about an unknown CkCallback type");
    return true;
  }
}

void CkCallback::thread_destroy() const {
  if (type==resumeThread && CpvAccess(threadCBs).get(d.thread.cb)==this) {
    CpvAccess(threadCBs).remove(d.thread.cb);
  }
}

CkCallbackResumeThread::~CkCallbackResumeThread() {
  void * res = thread_delay(); //<- block thread here if it hasn't already
  if (result != NULL) *result = res;
  else CkFreeMsg(res);
  thread_destroy();
}

/****** Callback-from-CCS ******/

// This function is called by CCS when a request comes in-- it maps the 
// request to a Charm++ message and passes the message to its callback.
extern "C" void ccsHandlerToCallback(void *cbPtr,int reqLen,const void *reqData) 
{
	CkCallback *cb=(CkCallback *)cbPtr;
	CkCcsRequestMsg *msg=new (reqLen,0) CkCcsRequestMsg;
	msg->reply=CcsDelayReply();
	msg->length=reqLen;
	memcpy(msg->data,reqData,reqLen);
	cb->send(msg);
}

// Register this callback with CCS.
void ckcallback_group::registerCcsCallback(const char *name,const CkCallback &cb)
{
	CcsRegisterHandlerFn(name,ccsHandlerToCallback,new CkCallback(cb));
}

// Broadcast this callback registration to all processors
void CcsRegisterHandler(const char *ccs_handlername,const CkCallback &cb) {
	_ckcallbackgroup.registerCcsCallback(ccs_handlername,cb);
}

enum {dataMsgTag=0x7ed2beef};
CkDataMsg *CkDataMsg::buildNew(int length,const void *data)
{
	CkDataMsg *msg=new (&length,0) CkDataMsg;
	msg->length=length;
	memcpy(msg->data,data,length);
	msg->checkTag=dataMsgTag;
	return msg;
}

void CkDataMsg::check(void)
{
	if (checkTag!=dataMsgTag)
		CkAbort("CkDataMsg corrupted-- bad tag.");
}

void CkCallbackInit() {
  CpvInitialize(threadCB_t, threadCBs);
  CpvInitialize(unsigned int, nextThreadCB);
  CpvAccess(nextThreadCB)=1;
}

#include "CkCallback.def.h"

