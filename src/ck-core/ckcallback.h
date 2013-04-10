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

#include "cksection.h"
#include "conv-ccs.h" /*for CcsDelayedReply struct*/
#include "charm.h"
#include "ckarrayindex.h"

typedef void (*CkCallbackFn)(void *param,void *message);
typedef void (*Ck1CallbackFn)(void *message);

class CProxyElement_ArrayBase; /*forward declaration*/
class CProxySection_ArrayBase;/*forward declaration*/
class CProxyElement_Group; /*forward declaration*/
class CProxy_NodeGroup;
class Chare;
class Group;
class NodeGroup;
class ArrayElement;
#define CkSelfCallback(ep)  CkCallback(this, ep)

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
	sendNodeGroup, //Send to a nodegroup (d.group)
	sendArray, //Send to an array (d.array)
	isendChare, //Inlined send to a chare (d.chare)
	isendGroup, //Inlined send to a group (d.group)
	isendNodeGroup, //Inlined send to a nodegroup (d.group)
	isendArray, //Inlined send to an array (d.array)
	bcastGroup, //Broadcast to a group (d.group)
	bcastNodeGroup, //Broadcast to a nodegroup (d.group)
	bcastArray, //Broadcast to an array (d.array)
	bcastSection,//Broadcast to a section(d.section)
	replyCCS // Reply to a CCS message (d.ccsReply)
	} callbackType;
private:
	union callbackData {
	struct s_thread { //resumeThread
		int onPE; //Thread is waiting on this PE
		int cb; //The suspending callback (0 if already done)
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
		CkArrayIndexBase idx; //Index to send to (if any)
	} array;
	struct s_section{
		CkSectionInfoStruct sinfo;
                CkArrayIndex *_elems;
                int _nElems;
                int *pelist;
                int npes;
		int ep;
	} section;

	struct s_ccsReply {
		CcsDelayedReply reply;
	} ccsReply;
	//callbackData(){memset(this,0,sizeof(callbackData));}
	//callbackData()=default;
	//constructor()=default;
	};

public:	
	callbackType type;
	callbackData d;
	void impl_thread_init(void);
	void *impl_thread_delay(void) const;

	CkCallback(void) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=invalid;
	}
	//This is how you create ignore, ckExit, and resumeThreads:
	CkCallback(callbackType t) {
#if CMK_REPLAYSYSTEM
	  bzero(this, sizeof(CkCallback));
#endif
	  if (t==resumeThread) impl_thread_init();
	  type=t;
	}

    // Call a C function on the current PE
	CkCallback(Ck1CallbackFn fn) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=call1Fn;
	  d.c1fn.fn=fn;
	}

    // Call a C function on the current PE
	CkCallback(CkCallbackFn fn,void *param) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=callCFn;
	  d.cfn.onPE=CkMyPe(); d.cfn.fn=fn; d.cfn.param=param;
	}

    // Call a chare entry method
	CkCallback(int ep,const CkChareID &id,CmiBool doInline=CmiFalse) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=doInline?isendChare:sendChare;
	  d.chare.ep=ep; d.chare.id=id;
	}

    // Bcast to nodegroup
	CkCallback(int ep,const CProxy_NodeGroup &ngp);

    // Bcast to a group or nodegroup
	CkCallback(int ep,const CkGroupID &id, int isNodeGroup=0) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=isNodeGroup?bcastNodeGroup:bcastGroup;
	  d.group.ep=ep; d.group.id=id;
	}

    // Send to nodegroup element
	CkCallback(int ep,int onPE,const CProxy_NodeGroup &ngp,CmiBool doInline=CmiFalse);

    // Send to group/nodegroup element
	CkCallback(int ep,int onPE,const CkGroupID &id,CmiBool doInline=CmiFalse, int isNodeGroup=0) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=doInline?(isNodeGroup?isendNodeGroup:isendGroup):(isNodeGroup?sendNodeGroup:sendGroup); 
      d.group.ep=ep; d.group.id=id; d.group.onPE=onPE;
	}

    // Send to specified group element
	CkCallback(int ep,const CProxyElement_Group &grpElt,CmiBool doInline=CmiFalse);
	
    // Bcast to array
	CkCallback(int ep,const CkArrayID &id) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=bcastArray;
	  d.array.ep=ep; d.array.id=id;
	}

    // Send to array element
	CkCallback(int ep,const CkArrayIndex &idx,const CkArrayID &id,CmiBool doInline=CmiFalse) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=doInline?isendArray:sendArray;
	  d.array.ep=ep; d.array.id=id; d.array.idx = idx;
	}

    // Bcast to array
	CkCallback(int ep,const CProxyElement_ArrayBase &arrElt,CmiBool doInline=CmiFalse);
	
	//Bcast to section
	CkCallback(int ep,CProxySection_ArrayBase &sectElt,CmiBool doInline=CmiFalse);
	CkCallback(int ep, CkSectionID &sid);
	
	// Send to chare
	CkCallback(Chare *p, int ep, CmiBool doInline=CmiFalse);

    // Send to group element on current PE
	CkCallback(Group *p, int ep, CmiBool doInline=CmiFalse);

    // Send to nodegroup element on current node
	CkCallback(NodeGroup *p, int ep, CmiBool doInline=CmiFalse);

    // Send to specified array element 
 	CkCallback(ArrayElement *p, int ep,CmiBool doInline=CmiFalse);

	CkCallback(const CcsDelayedReply &reply) {
#if CMK_REPLAYSYSTEM
      bzero(this, sizeof(CkCallback));
#endif
      type=replyCCS;
	  d.ccsReply.reply=reply;
	}

	~CkCallback() {
	  thread_destroy();
          if (bcastSection == type) {
            if (d.section._elems != NULL) delete [] d.section._elems;
            if (d.section.pelist != NULL) delete [] d.section.pelist;
          }
	}
	
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

	void thread_destroy() const;
	
/**
 * Send this message back to the caller.
 *
 * Libraries should call this from their "done" entry points.
 * It takes the given message and handles it appropriately.
 * After the send(), this callback is finished and cannot be reused.
 */
	void send(void *msg=NULL) const;
	
/**
 * Send this data, formatted as a CkDataMsg, back to the caller.
 */
	void send(int length,const void *data) const;
	
	void pup(PUP::er &p);
};
//PUPbytes(CkCallback) //FIXME: write a real pup routine

/**
 * Convenience class: a thread-suspending callback.  
 * Makes sure the thread actually gets delayed, even if the 
 *   library can't or won't call "thread_delay".
 * The return value is lost, so your library needs to call
 *   thread_delay itself if you want a return value.
 * Modification Filippo: Passing in an pointer argument, the return
 *   value will be stored in that pointer 
 */
class CkCallbackResumeThread : public CkCallback {
 protected: void ** result;
 public:
	CkCallbackResumeThread(void)
		:CkCallback(resumeThread) { result = NULL; }
	CkCallbackResumeThread(void * &ptr)
	    :CkCallback(resumeThread) { result = &ptr; }
        ~CkCallbackResumeThread(void);
};

void _registerCkCallback(void); //used by init

void CkCallbackInit();

#endif



