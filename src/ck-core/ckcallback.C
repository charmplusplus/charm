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
#include "CkCallback.decl.h"

/*readonly*/ CProxy_ckcallback_group ckcallbackgroup;

//This main chare is only used to create the callback forwarding group
class ckcallback_main : public CBase_ckcallback_main {
public:
	ckcallback_main(CkArgMsg *m) {
		ckcallbackgroup=CProxy_ckcallback_group::ckNew();
		delete m;
	}
};

//The callback group is used to forward a callback to the processor
// it originated from.
class ckcallback_group : public CBase_ckcallback_group {
public:
	ckcallback_group() { /*empty*/ }
	void call(CkCallback &c,CkMarshalledMessage &msg) {
		c.send(msg.getMessage());
	}
};

/*************** CkCallback implementation ***************/
//Initialize the callback's thread fields before sending it off:
void CkCallback::impl_thread_init(void)
{
	d.thread.onPE=CkMyPe();
	d.thread.cb=this; //<- so we can find this structure later
	d.thread.th=NULL; //<- thread isn't suspended yet
	d.thread.ret=NULL;//<- no data to return yet
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
	if (d.thread.cb!=NULL) dest=d.thread.cb;
	if (dest->d.thread.cb!=NULL) 
	{  //We need to sleep for the result:
		dest->d.thread.th=CthSelf(); //<- so we know a thread is waiting
		CthSuspend();
		if (dest->d.thread.cb!=NULL) 
			CkAbort("thread resumed, but callback data is still empty");
	}
	return dest->d.thread.ret;
}


/*These can't be defined in the .h file like the other constructors
 * because we need CkCallback before CProxyElement* are defined.
 */
CkCallback::CkCallback(int ep,const CProxyElement_Group &grpElt,bool doInline) 
	:type(doInline?isendGroup:sendGroup) 
{
	d.group.ep=ep; 
	d.group.id=grpElt.ckGetGroupID(); 
	d.group.onPE=grpElt.ckGetGroupPe();
}
CkCallback::CkCallback(int ep,const CProxyElement_ArrayBase &arrElt,bool doInline)
	:type(doInline?isendArray:sendArray) 
{
	d.array.ep=ep; 
	d.array.id=arrElt.ckGetArrayID(); 
	d.array.idx.asMax()=arrElt.ckGetIndex();
}


/*Libraries should call this from their "done" entry points.
  It takes the given message and handles it appropriately.
  After the send(), this callback is finished and cannot be reused.
*/
void CkCallback::send(void *msg) const
{
	switch(type) {
	case ignore: //Just ignore the callback
		if (msg) CkFreeMsg(msg);
		break; 
	case ckExit: //Call ckExit
		if (msg) CkFreeMsg(msg);
		CkExit();
		break;
	case resumeThread: //Resume a waiting thread
		if (d.thread.onPE==CkMyPe()) {
			CkCallback *dest=d.thread.cb;
			if (dest==NULL) 
				CkAbort("Already sent a value to this callback!\n");
			dest->d.thread.ret=msg; //<- return data
			dest->d.thread.cb=NULL; //<- mark callback as finished
			if (dest->d.thread.th!=NULL)
				CthAwaken(dest->d.thread.th);
		} 
		else //Forward message to processor where the thread actually lives
			ckcallbackgroup[d.thread.onPE].call(*this,(CkMessage *)msg);
		break;
	case call1Fn: //Call a C function pointer on the current processor
		(d.c1fn.fn)(msg);
		break;
	case callCFn: //Call a C function pointer on the appropriate processor
		if (d.cfn.onPE==CkMyPe())
			(d.cfn.fn)(d.cfn.param,msg);
		else
			ckcallbackgroup[d.cfn.onPE].call(*this,(CkMessage *)msg);
		break;
	case sendChare: //Send message to a chare
		if (!msg) msg=CkAllocSysMsg();
		CkSendMsg(d.chare.ep,msg,&d.chare.id);
		break;
	case isendChare: //inline send-to-chare
		if (!msg) msg=CkAllocSysMsg();
		CkSendMsgInline(d.chare.ep,msg,&d.chare.id);
		break;
	case sendGroup: //Send message to a group element
		if (!msg) msg=CkAllocSysMsg();
		CkSendMsgBranch(d.group.ep,msg,d.group.onPE,d.group.id);
		break;
	case isendGroup: //inline send-to-group element
		if (!msg) msg=CkAllocSysMsg();
		CkSendMsgBranchInline(d.group.ep,msg,d.group.onPE,d.group.id);
		break;
	case sendArray: //Send message to an array element
		if (!msg) msg=CkAllocSysMsg();
		CkSendMsgArray(d.array.ep,msg,d.array.id,d.array.idx.asMax());
		break;
	case isendArray: //inline send-to-array element
		if (!msg) msg=CkAllocSysMsg();
		CkSendMsgArrayInline(d.array.ep,msg,d.array.id,d.array.idx.asMax());
		break;
	case bcastGroup:
		if (!msg) msg=CkAllocSysMsg();
		CkBroadcastMsgBranch(d.group.ep,msg,d.group.id);
		break;
	case bcastArray:
		if (!msg) msg=CkAllocSysMsg();
		CkBroadcastMsgArray(d.array.ep,msg,d.array.id);
		break;
	case invalid: //Uninitialized
		CmiAbort("Called send on uninitialized callback");
		break;
	default: //Out-of-bounds type code
		CmiAbort("Called send on corrupted callback");
		break;
	};
}


#include "CkCallback.def.h"

