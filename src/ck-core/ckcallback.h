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
	enum callbackType {
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
	};
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
		CMK_REFNUM_TYPE refnum; // Reference number to set on the message
		bool hasRefnum;
	} chare;
	struct s_group { //(sendGroup, bcastGroup)
		int ep; //Entry point to call
		CkGroupID id; //Group to call it on
		int onPE; //Processor to send to (if any)
		CMK_REFNUM_TYPE refnum; // Reference number to set on the message
		bool hasRefnum;
	} group;
	struct s_array { //(sendArray, bcastArray)
		int ep; //Entry point to call
		CkGroupID id; //Array ID to call it on
		CkArrayIndexBase idx; //Index to send to (if any)
		CMK_REFNUM_TYPE refnum; // Reference number to set on the message
		bool hasRefnum;
	} array;
	struct s_section{
		CkSectionInfoStruct sinfo;
                CkArrayIndex *_elems;
                int _nElems;
                int *pelist;
                int npes;
		int ep;
		CMK_REFNUM_TYPE refnum; // Reference number to set on the message
		bool hasRefnum;
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

	bool operator==(CkCallback & other){
	  if(type != other.type)
	    return false;
	  switch (type) {
	    case resumeThread:
	      return (d.thread.onPE == other.d.thread.onPE &&
		  d.thread.cb == other.d.thread.cb);
	    case isendChare:
	    case sendChare:
	      return (d.chare.ep == other.d.chare.ep &&
		  d.chare.id.onPE == other.d.chare.id.onPE &&
		  d.chare.hasRefnum == other.d.chare.hasRefnum &&
		  d.chare.refnum == other.d.chare.refnum);
	    case isendGroup:
	    case sendGroup:
	    case isendNodeGroup:
	    case sendNodeGroup:
	      return (d.group.ep == other.d.group.ep &&
		  d.group.id == other.d.group.id &&
		  d.group.onPE == other.d.group.onPE &&
		  d.group.hasRefnum == other.d.group.hasRefnum &&
		  d.group.refnum == other.d.group.refnum);
	    case bcastNodeGroup:
	    case bcastGroup:
	      return (d.group.ep == other.d.group.ep &&
		  d.group.id == other.d.group.id &&
		  d.group.hasRefnum == other.d.group.hasRefnum &&
		  d.group.refnum == other.d.group.refnum);
	    case isendArray:
	    case sendArray:
	      return (d.array.ep == other.d.array.ep &&
		  d.array.id == other.d.array.id &&
		  d.array.idx == other.d.array.idx &&
		  d.array.hasRefnum == other.d.array.hasRefnum &&
		  d.array.refnum == other.d.array.refnum);
	    case bcastArray:
	      return (d.array.ep == other.d.array.ep &&
		  d.array.id == other.d.array.id &&
		  d.array.hasRefnum == other.d.array.hasRefnum &&
		  d.array.refnum == other.d.array.refnum);
	    case replyCCS:
	      return true;
	    case call1Fn:
	      return (d.c1fn.fn == other.d.c1fn.fn);
	    case callCFn:
	      return (d.cfn.fn == other.d.cfn.fn &&
		  d.cfn.onPE == other.d.cfn.onPE &&
		  d.cfn.param == other.d.cfn.param);
	    case ignore:
	    case ckExit:
	    case invalid:
	    case bcastSection:
	      return true;
	    default:
	      CkAbort("Inconsistent CkCallback type");
	      return false;
	  }
	}


	void impl_thread_init(void);
	void *impl_thread_delay(void) const;

	CkCallback(void) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=invalid;
	}
	//This is how you create ignore, ckExit, and resumeThreads:
	CkCallback(callbackType t) {
#if CMK_REPLAYSYSTEM
	  memset(this, 0, sizeof(CkCallback));
#endif
	  if (t==resumeThread) impl_thread_init();
	  type=t;
	}

    // Call a C function on the current PE
	CkCallback(Ck1CallbackFn fn) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=call1Fn;
	  d.c1fn.fn=fn;
	}

    // Call a C function on the current PE
	CkCallback(CkCallbackFn fn,void *param) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=callCFn;
	  d.cfn.onPE=CkMyPe(); d.cfn.fn=fn; d.cfn.param=param;
	}

    // Call a chare entry method
	CkCallback(int ep,const CkChareID &id,bool doInline=false) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendChare:sendChare;
	  d.chare.ep=ep; d.chare.id=id;
          d.chare.hasRefnum = false;
          d.chare.refnum = 0;
	}

    // Bcast to nodegroup
	CkCallback(int ep,const CProxy_NodeGroup &ngp);

    // Bcast to a group or nodegroup
	CkCallback(int ep,const CkGroupID &id, int isNodeGroup=0) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=isNodeGroup?bcastNodeGroup:bcastGroup;
	  d.group.ep=ep; d.group.id=id;
          d.group.hasRefnum = false;
          d.group.refnum = 0;
	}

    // Send to nodegroup element
	CkCallback(int ep,int onPE,const CProxy_NodeGroup &ngp,bool doInline=false);

    // Send to group/nodegroup element
	CkCallback(int ep,int onPE,const CkGroupID &id,bool doInline=false, int isNodeGroup=0) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?(isNodeGroup?isendNodeGroup:isendGroup):(isNodeGroup?sendNodeGroup:sendGroup); 
      d.group.ep=ep; d.group.id=id; d.group.onPE=onPE;
	  d.group.hasRefnum = false;
          d.group.refnum = 0;
        }

    // Send to specified group element
	CkCallback(int ep,const CProxyElement_Group &grpElt,bool doInline=false);
	
    // Bcast to array
	CkCallback(int ep,const CkArrayID &id) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=bcastArray;
	  d.array.ep=ep; d.array.id=id;
	  d.array.hasRefnum = false;
          d.array.refnum = 0;
        }

    // Send to array element
	CkCallback(int ep,const CkArrayIndex &idx,const CkArrayID &id,bool doInline=false) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
#endif
      type=doInline?isendArray:sendArray;
	  d.array.ep=ep; d.array.id=id; d.array.idx = idx;
	  d.array.hasRefnum = false;
          d.array.refnum = 0;
        }

    // Bcast to array
	CkCallback(int ep,const CProxyElement_ArrayBase &arrElt,bool doInline=false);
	
	//Bcast to section
	CkCallback(int ep,CProxySection_ArrayBase &sectElt,bool doInline=false);
	CkCallback(int ep, CkSectionID &sid);
	
	// Send to chare
	CkCallback(Chare *p, int ep, bool doInline=false);

    // Send to group element on current PE
	CkCallback(Group *p, int ep, bool doInline=false);

    // Send to nodegroup element on current node
	CkCallback(NodeGroup *p, int ep, bool doInline=false);

    // Send to specified array element 
 	CkCallback(ArrayElement *p, int ep,bool doInline=false);

	CkCallback(const CcsDelayedReply &reply) {
#if CMK_REPLAYSYSTEM
      memset(this, 0, sizeof(CkCallback));
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

        /// Does this callback point at something that may not be at the same
        /// address after a checkpoint/restart cycle?
        bool containsPointer() const;

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

        void setRefnum(CMK_REFNUM_TYPE refnum) {
		switch(type) {
                case sendChare:
                case isendChare:
                  d.chare.hasRefnum = true;
                  d.chare.refnum = refnum;
                  break;

                case sendGroup:
                case sendNodeGroup:
                case isendGroup:
                case isendNodeGroup:
                case bcastGroup:
                case bcastNodeGroup:
                  d.group.hasRefnum = true;
                  d.group.refnum = refnum;
                  break;

                case sendArray:
                case isendArray:
                case bcastArray:
                  d.array.hasRefnum = true;
                  d.array.refnum = refnum;
                  break;

                case bcastSection:
                  d.section.hasRefnum = true;
                  d.section.refnum = refnum;
                  break;

                default:
                  CkAbort("Tried to set a refnum on a callback not directed at an entry method");
                }
        }
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



