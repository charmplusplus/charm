#include "ckfutures.h"

#ifndef _IGET_FLOWCONTROL_H
#define _IGET_FLOWCONTROL_H

#define IGET_FLOWCONTROL 0 
#define IGET_TOKENNUM 16 
#define IGET_MINTOKENNUM 1

#define CKFUTURE_IGET 	1
#define DAGGER_IGET   	0
#if CKFUTURE_IGET
typedef   CkFutureID     CkIGetID;
#define   CkIGet 	 CkCreateAttachedFutureSend
#define   CkIGetWait 	 CkWaitReleaseFuture
#elif DAGGER_IGET

#endif

#ifndef IGET_FLOWCONTROL
#define IGET_FLOWCONTROL 0 
#endif

#if IGET_FLOWCONTROL
#ifndef IGET_TOKENNUM
#define IGET_TOKENNUM 6 
#endif
#endif

extern "C" void TokenUpdatePeriodic();
extern "C" int getRSS();

#if IGET_FLOWCONTROL==0

class IGetControlClass {
public:
  int iget_request(CkIGetID fut, void *msg, int ep, CkArrayID, CkArrayIndex, void(*fptr)(CkArrayID,CkArrayIndex,void*,int,int))
    {return 1;}
  void iget_free(CthThread tid, int size) {}
  void iget_resend(CkIGetID) {}
  void iget_updateTokenNum() {}
};
#elif CKFUTURE_IGET


// BUG: if queue length is longer than 128, trouble 
//      new TheHashTable entry won't be inited to '-1'
#define MAXQUEUELENGTH 1024

template <class KEY, class OBJ>
class HashQueueT {
  CkQ<OBJ> *TheHashQueue;
  CkQ<int> *TheHashTable;
public:
  HashQueueT() 
  { 
    TheHashQueue = new CkQ<OBJ>(); 
    TheHashTable = new CkQ<int>(MAXQUEUELENGTH);
    // Init table to be all "-1"
    // VERY IMPORTANT: -1 indicates no data with this 'key' index
    for(int i=0;i<MAXQUEUELENGTH;i++)
      TheHashTable->insert(i,-1);
  }
  ~HashQueueT() {}
  OBJ deq()
  {
    if(TheHashTable->length()<=0) return NULL;
    OBJ e=TheHashQueue->deq();
//    TheHashTable->removeFrom(e->futNum);
    return e;
  }
  void key_enq(OBJ entry, KEY key) 
  {
    TheHashQueue->enq(entry);
//    updatetable(TheHashTable, key, TheHashQueue->length()-1);
  }
  OBJ  key_deq(KEY key) 
  {
/*  int pos=getpostable(TheHashTable,key);
    CkAssert(pos>=0);
*/
    int i;
    for(i=0;i<TheHashQueue->length();i++)
    {
      if((*TheHashQueue)[i]->futNum==key)     
	break;
    }
    if(i>=TheHashQueue->length()) return NULL;
    return TheHashQueue->remove(i); 
  }
  void key_promote(KEY key) 
  {
    OBJ entry=key_deq(key);
    TheHashQueue->insert(0,entry);
  }
  bool key_find(KEY key)
  {
/*
    CkAssert(key<MAXQUEUELENGTH);
    return (getpostable(TheHashTable,key)>=0);
*/
    int i;
    for(i=0;i<TheHashQueue->length();i++)
    {
      if((*TheHashQueue)[i]->futNum==key)
        break;
    }
    if(i>=TheHashQueue->length()) return false;
    else return true;

  }
private:
  void updatetable(CkQ<int> *table, KEY key, int pos) 
  {
    table->insert(key, pos);
  }  
  int getpostable(CkQ<int> *table, KEY key)
  {
    return (*table)[(int)key];
  }
};

typedef struct iget_token_struct {
  CkIGetID futNum;
  int status;
  void *m;
  int ep;
//  void *obj;     
//  void(*fptr1)(void*,void*,int,int);
  CkArrayID aid;
  CkArrayIndex idx;
  void(*fptr)(CkArrayID,CkArrayIndex,void*,int,int);   
} *iget_tokenqueue_entry;

typedef HashQueueT<CkIGetID, iget_tokenqueue_entry> HashTokenQueue;

class IGetControlClass {
public:
/*  int iget_request(CkIGetID fut, void *msg, int ep, void *obj,void(*fptr)(void*,void*,int,int))
  {
    int ret_status=1, size=1;
    if(iget_token>=size){
      iget_token-=size;
      //(fptr)(obj,msg,ep,0);  // Send the msg here
    }
    else //(iget_request(CthSelf(),1)==false)
    { 
      //iget_tokenqueue_enqueue(fut,msg,ep,obj,fptr);
      ret_status = 0; // No send will be done this case
    }
    return ret_status;
  }
*/
	int iget_request(CkIGetID fut, void *msg, int ep, CkArrayID id,
CkArrayIndex idx, void(*fptr)(CkArrayID,CkArrayIndex,void*,int,int),
int);

	void iget_free(int size);

  void iget_updateTokenNum();
/*
 *  Called when wait is posted, but no iget is really sent yet
 *  
 */
  void iget_resend(CkIGetID fut);

/* First-come First-serve queue */
/*   a) request for resource, 
 *      if no available, push requester into queue, return status=0 ; 
 *      if yes, return status=1, user should go ahead and do iget */


/* Thread First-come First-serve queue: 
 *   Guarantees at least the first incoming thread would 
 *   get all its requests and progress 
 *     
 */
  int  CTH_FCFS_request(CthThread tid, int size){ return 0;}  
  void CTH_FCFS_free(){ }

/* Paired First-come First-serve queue:
 *   Requests are served in pairs if there is any that exists
 *
 */
  int  PAIR_FCFS_request() {return 0;}
  void PAIR_FCFS_free() {}

/* 
 *   Drop sent requests, to ensure progress of at least one thread
 */
  int DROP_FCFS_request() {return 0;}

  IGetControlClass() {iget_token = IGET_TOKENNUM;iget_token_history =
IGET_TOKENNUM;lastupdatetime = 0;iget_outstanding=0;IGET_UNITMESSAGE=1024;}
private:
  HashTokenQueue queue;
  int iget_token;
  int iget_outstanding;
  int iget_token_history;
  double lastupdatetime;
  int IGET_UNITMESSAGE;
/*
  inline void iget_tokenqueue_enqueue(CkIGetID gid,void* m,int ep,void *obj,void(*fptr)(void*,void*,int,int))
  {
    iget_tokenqueue_entry e=new iget_token_struct(); 
    e->futNum=gid; e->m=m; e->ep=ep; e->obj=obj; e->fptr=fptr; e->status=0; 
    queue.key_enq(e,gid);
  } 
*/
  inline void iget_tokenqueue_enqueue(CkIGetID gid,void* m,int ep, CkArrayID aid, CkArrayIndex
				      idx, void(*fptr)(CkArrayID,CkArrayIndex,void*,int,int))
  {
    iget_tokenqueue_entry e=new iget_token_struct();
    e->futNum=gid; e->m=m; e->ep=ep; e->aid=aid; e->idx=idx; e->fptr=fptr; e->status=0;
    queue.key_enq(e,gid);
  }

  inline iget_tokenqueue_entry iget_tokenqueue_dequeue() {
    return queue.deq();
  }

  inline iget_tokenqueue_entry iget_tokenqueue_remove(CkIGetID gid)
  {
    return queue.key_deq(gid);
  }
  inline bool iget_tokenqueue_find(CkIGetID gid)
  {
    return queue.key_find(gid);
  }
  inline void iget_tokenqueue_promote(CkIGetID gid)
  {
    queue.key_promote(gid);
  }
};

#endif

extern IGetControlClass TheIGetControlClass;

#endif
