#ifndef _CHARMPP_H_
#define _CHARMPP_H_

#include <stdlib.h>
#include "charm.h"
#if CMK_DEBUG_MODE
#include <string.h>
#endif

#if CMK_BOOL_UNDEFINED
enum Bool {False=0, True=1};
#else
typedef bool Bool;
#define False false
#define True true
#endif

#if CMK_DEBUG_MODE
class Chare;
extern void putObject(Chare *);
extern void removeObject(Chare *);
#endif

class Chare {
  protected:
    CkChareID thishandle;
  public:
    void *operator new(size_t, void *ptr) { return ptr; };
#if CMK_COMPILEMODE_ANSI
    void operator delete(void*, void*) {};
#endif
    void *operator new(size_t s) { return malloc(s); }
    void operator delete(void *ptr) { free(ptr); }
#if CMK_DEBUG_MODE
    Chare() { CkGetChareID(&thishandle); putObject(this); }
    // Making the destructor virtual gets rid of some egcs warnings
    virtual ~Chare() { removeObject(this); }
    virtual char *showHeader(void) {
      char *ret = (char *)malloc(strlen("Default Header")+1);
      _MEMCHECK(ret);
      strcpy(ret,"Default Header");
      return ret;
    }
    virtual char *showContents(void) {
      char *ret = (char *)malloc(strlen("Default Content")+1);
      _MEMCHECK(ret);
      strcpy(ret,"Default Content");
      return ret;
    }
#else
    Chare() { CkGetChareID(&thishandle); }
#endif
};

class Group : public Chare {
  protected:
    CkGroupID thisgroup;
  public:
    Group() { thisgroup = CkGetGroupID(); }
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

class _CK_NGID : public _CK_CID {
  private:
    int _chare;
  protected:
    CkGroupID _ck_ngid;
    int _isChare(void) { return _chare; }
    void _setChare(int c) { _chare = c; }
};

class Array1D;

class _CK_AID {
  private:
    int _chare;
  public:
    CkChareID _cid;
    CkGroupID _ck_aid;
    Array1D *_array;
    int _elem;
    void _setAid(CkGroupID aid) {
      _ck_aid = aid;
      _array = (Array1D*) CkLocalBranch(aid);
    }
    _CK_AID(CkGroupID aid) {
      _setAid(aid);
      _elem = -1;
    }
    _CK_AID(CkGroupID aid, int elem) {
      _setAid(aid);
      _elem = elem;
    }
    _CK_AID() {}
    void _setCid(CkChareID cid) { _cid = cid; }
    int _isChare(void) { return _chare; }
    void _setChare(int c) { _chare = c; }
};

typedef _CK_AID CkArrayID;
typedef CkArrayID CkAID;  /* Depricated usage */

class CkQdMsg {
  public:
    void *operator new(size_t s) { return CkAllocMsg(0,s,0); }
    void operator delete(void* ptr) { CkFreeMsg(ptr); }
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
  if(_GETIDX(idx)% cid.magic)
    CkAbort("Illegal ChareID assignment to proxy.\n");
}
#else
static inline void _CHECK_CID(CkChareID, int){}
#endif

#include "ckstream.h"
#include "CkFutures.decl.h"
#include "ckarray.h"
#include "tempo.h"
#include "waitqd.h"

#endif
