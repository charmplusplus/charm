/*
A CkCallback is a simple way for a library to return data 
to a wide variety of user code, without the library having
to handle all 17 possible cases.

This object is implemented as a union, so the entire object
can be sent as bytes.  Another option would be to use a virtual 
"send" method.

Initial version by Orion Sky Lawlor, olawlor@acm.org, 2/8/2002
*/
#ifndef _CKCALLBACK_H_
#define _CKCALLBACK_H_

typedef void (*CkCallbackFn)(void *param,void *message);
typedef void (*Ck1CallbackFn)(void *message);

class CkCallback {
public:
	typedef enum {
	invalid=0, //Invalid callback
	ignore, //Do nothing
	ckExit, //Call ckExit
	resumeThread, //Resume a waiting thread (d.thread)
	callCFn, //Call a C function pointer with a parameter (d.cfn)
	call1Fn, //Call a C function pointer on any processor (d.c1fn)
	sendChare, //Send to a chare (d.chare)
	sendGroup, //Send to a group (d.group)
	sendArray, //Send to an array (d.array)
	isendChare, //Inlined send to a chare (d.chare)
	isendGroup, //Inlined send to a group (d.group)
	isendArray, //Inlined send to an array (d.array)
	bcastGroup, //Broadcast to a group (d.group)
	bcastArray //Broadcast to an array (d.array)
	} callbackType;
private:
	union callbackData {
	struct s_thread { //resumeThread
		int onPE; //Thread is waiting on this PE
		CkCallback *cb; //The suspending callback (NULL if already done)
		CthThread th; //Thread to resume (NULL if none waiting)
		void *ret; //Place to put the returned message
	} thread;
	struct s_cfn { //callCFn
		int onPE; //Call on this PE
		CkCallbackFn fn; //Function to call
		void *param; //User parameter
	} cfn;
	struct s_c1fn { //call1Fn
		Ck1CallbackFn fn; //Function to call on whatever processor
	} c1fn;
	struct s_chare { //sendChare
		int ep; //Entry point to call
		CkChareID id; //Chare to call it on
	} chare;
	struct s_group { //(sendGroup, bcastGroup)
		int ep; //Entry point to call
		CkGroupID id; //Group to call it on
		int onPE; //Processor to send to (if any)
	} group;
	struct s_array { //(sendArray, bcastArray)
		int ep; //Entry point to call
		CkGroupID id; //Array ID to call it on
		CkArrayIndexStruct idx; //Index to send to (if any)
	} array;
	};
	
	callbackType type; 
	callbackData d;
	
	void impl_thread_init(void);
	void *impl_thread_delay(void) const;
public:
	CkCallback(void) :type(invalid) {}
	//This is how you create ignore, ckExit, and resumeThreads:
	CkCallback(callbackType t) 
		:type(t) { if (t==resumeThread) impl_thread_init(); }

	CkCallback(Ck1CallbackFn fn)
		:type(call1Fn)
		{d.c1fn.fn=fn;}

	CkCallback(CkCallbackFn fn,void *param)
		:type(callCFn) 
		{d.cfn.onPE=CkMyPe(); d.cfn.fn=fn; d.cfn.param=param;}

	CkCallback(int ep,const CkChareID &id,bool doInline=false)
		:type(doInline?isendChare:sendChare) 
		{d.chare.ep=ep; d.chare.id=id;}

	CkCallback(int ep,const CkGroupID &id)
		:type(bcastGroup) 
		{d.group.ep=ep; d.group.id=id;}
	CkCallback(int ep,int onPE,const CkGroupID &id,bool doInline=false)
		:type(doInline?isendGroup:sendGroup) 
		{d.group.ep=ep; d.group.id=id; d.group.onPE=onPE;}
	
	CkCallback(int ep,const CkArrayID &id)
		:type(bcastArray) 
		{d.array.ep=ep; d.array.id=id;}
	CkCallback(int ep,const CkArrayIndex &idx,const CkArrayID &id,bool doInline=false)
		:type(doInline?isendArray:sendArray) 
		{d.array.ep=ep; d.array.id=id; d.array.idx.asMax()=idx;}

	int isInvalid(void) const {return type==invalid;}

/**
 * Interface used by threaded callbacks:
 * Libraries should call these from their "start" entry points.
 * Use "return cb.thread_delay()" to suspend the thread before
 * the return.
 * It's a no-op for everything but threads.
 */
	void *thread_delay(void) const {
		if (type==resumeThread) return impl_thread_delay();
		return NULL;
	}

/**
 * Libraries should call this from their "done" entry points.
 * It takes the given message and handles it appropriately.
 * After the send(), this callback is finished and cannot be reused.
 */
	void send(void *msg=NULL) const;
};


/**
 * Convenience class: a thread-suspending callback.  
 * Makes sure the thread actually gets delayed, even if the 
 *   library can't or won't call "thread_delay".
 * The return value is lost, so your library needs to call
 *   thread_delay itself if you want a return value.
 */
class CkCallbackResumeThread : public CkCallback {
 public:
	CkCallbackResumeThread(void)
		:CkCallback(resumeThread) {}
	~CkCallbackResumeThread(void) {
		thread_delay(); //<- block thread here if it hasn't already
	}
};


void _registerCkCallback(void); //used by init

#endif



