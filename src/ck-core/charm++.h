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

#if CMK_DEBUG_MODE
#include <string.h>
class Chare;
extern void putObject(Chare *);
extern void removeObject(Chare *);
#endif

#include "cklists.h"
#include "init.h"
#include "pup.h"

//We need CkMigrateMessage only to distinguish the migration
// constructor from all other constructors-- the type
// itself has no meaningful fields.
typedef struct {int is_only_a_name;} CkMigrateMessage;

class Chare {
  protected:
    CkChareID thishandle;
  public:
    /*Chare(CkMigrateMessage *m) {}*/
    void *operator new(size_t, void *ptr) { return ptr; };
#if CMK_COMPILEMODE_ANSI
    void operator delete(void*, void*) {};
#endif
    void *operator new(size_t s) { return malloc(s); }
    void operator delete(void *ptr) { free(ptr); }
#if CMK_DEBUG_MODE
    Chare() { CkGetChareID(&thishandle); putObject(this); }
    virtual ~Chare();
    virtual char *showHeader(void);
    virtual char *showContents(void);
#else
    Chare() { CkGetChareID(&thishandle); }
    virtual ~Chare(); //<- this is needed if *any* child is to have a virtual destructor
#endif
    virtual void pup(PUP::er &p);//<- pack/unpack routine
};

class Group : public Chare {
  protected:
    CkGroupID thisgroup;
  public:
    /*Group(CkMigrateMessage *m) {}*/
    Group() { thisgroup = CkGetGroupID(); }
    virtual void pup(PUP::er &p);//<- pack/unpack routine
};

class NodeGroup : public Chare {
  protected:
    CkGroupID thisgroup;
  public:
    CmiNodeLock __nodelock;
    NodeGroup() { thisgroup=CkGetNodeGroupID(); __nodelock=CmiCreateLock();}
    ~NodeGroup() { CmiDestroyLock(__nodelock); }
};

class _CK_CID {
  protected:
    CkChareID _ck_cid;
};

class _CK_GID : public _CK_CID {
  private:
    int _chare;
  protected:
    CkGroupID _ck_gid;
    int _isChare(void) { return _chare; }
    void _setChare(int c) { _chare = c; }
};

class _CK_NGID : public _CK_GID {
};

class CkArray;

class CkArrayID {
  public:
    CkGroupID _aid;
    CkArrayID() {}
    CkArrayID(CkGroupID aid) {
      _aid=aid;
    }
    CkArray *ckLocalBranch(void) const
	{ return (CkArray *)CkLocalBranch(_aid); }
    CkArray *getArrayManager(void) const {return ckLocalBranch();}
    static CkArray *CkLocalBranch(CkGroupID id) 
	{ return (CkArray *)::CkLocalBranch(id); }

    void pup(PUP::er &p) { 
      p(_aid); 
    }
};

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

extern unsigned int _primesTable[];

extern int _GETIDX(int cidx);
extern void _REGISTER_BASE(int didx, int bidx);
extern void _REGISTER_DONE(void);

#ifndef CMK_OPTIMIZE
static inline void _CHECK_CID(CkChareID cid, int idx)
{
  if(cid.magic%_GETIDX(idx))
    CkAbort("Illegal ChareID assignment to proxy.\n");
}
#else
static inline void _CHECK_CID(CkChareID, int){}
#endif

#include "ckarray.h"
#include "ckstream.h"
#include "CkFutures.decl.h"
#include "tempo.h"
#include "waitqd.h"

#endif
