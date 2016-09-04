#ifndef _CHARE_H_
#define _CHARE_H_

#define CHARE_MAGIC    0x201201

#include "ckobjQ.h"
#include "pup.h"
#include "charm.h"


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


#if CMK_SMP && CMK_TASKQUEUE
#include "cktaskQ.h"
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
class ChareMlogData;
#endif

#define CHARE_MAGIC    0x201201

/**
  The base class of all parallel objects in Charm++,
  including Array Elements, Groups, and NodeGroups.
*/
class Chare {
  protected:
    CkChareID thishandle;
#if CMK_OBJECT_QUEUE_AVAILABLE
    CkObjectMsgQ objQ;                // object message queue
#endif
  public:
#if CMK_ERROR_CHECKING
    int magic;
#endif
#ifndef CMK_CHARE_USE_PTR
    int chareIdx;                  // index in the chare obj table (chare_objs)
#endif
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    ChareMlogData *mlogData;
#endif
    Chare(CkMigrateMessage *m);
    Chare();
    virtual ~Chare(); //<- needed for *any* child to have a virtual destructor

    /// Pack/UnPack - tell the runtime how to serialize this class's
    /// data for migration, checkpoint, etc.
    virtual void pup(PUP::er &p);
    /// Routine that runtime the runtime actually calls, to enable
    /// more intelligence in generated code that overrides this. The
    /// actual pup() method must remain virtual, so that this
    /// continues to work for older code.
    virtual void virtual_pup(PUP::er &p) { pup(p); }
    void parent_pup(PUP::er &p) {
      (void)p;
      CkAbort("Should never get here - only called in generated CBase code");
    }

    inline const CkChareID &ckGetChareID(void) const {return thishandle;}
    inline void CkGetChareID(CkChareID *dest) const {*dest=thishandle;}
    // object message queue
    void  CkEnableObjQ();
#if CMK_OBJECT_QUEUE_AVAILABLE
    inline CkObjectMsgQ &CkGetObjQueue() { return objQ; }
#endif
    CHARM_INPLACE_NEW
    /// Return the type of this chare, as present in _chareTable
    virtual int ckGetChareType() const;
    /// Return a strdup'd array containing this object's string name.
    virtual char *ckDebugChareName(void);
    /// Place into str a copy of the id of this object up to limit bytes, return
    /// the number of bytes used for the id
    virtual int ckDebugChareID(char *str, int limit);
    virtual void ckDebugPup(PUP::er &p);
    /// Called when a [threaded] charm entry method is created:
    virtual void CkAddThreadListeners(CthThread tid, void *msg);
#if CMK_ERROR_CHECKING
    inline void sanitycheck() { 
        if (magic != CHARE_MAGIC)
          CmiAbort("Charm++ Fatal Error> Chare magic number does not agree, possibly due to pup functions not calling parent class.");
    }
#endif
};

#endif
