#ifndef _CHARMPP_H_
#define _CHARMPP_H_

#include <stdlib.h>

extern "C" {
#include "charm.h"
}

#if CMK_DEBUG_MODE
class Chare;
void putObject(Chare *);
void removeObject(Chare *);
#endif

class Chare {
  protected:
    CkChareID thishandle;
  public:
    void *operator new(size_t, void *ptr) { return ptr; }
    void operator delete(void *, void *) { }
    void *operator new(size_t s) { return malloc(s); }
    void operator delete(void *ptr) { free(ptr); }
#if CMK_DEBUG_MODE
    Chare() { CkGetChareID(&thishandle); putObject(this); }
    ~Chare() { removeObject(this); }
    virtual char *showHeader(void) {
      char *ret = (char *)malloc(strlen("Default Header")+1);
      strcpy(ret,"Default Header");
      return ret;
    }
    virtual char *showContents(void) {
      char *ret = (char *)malloc(strlen("Default Content")+1);
      strcpy(ret,"Default Content");
      return ret;
    }
#else
    Chare() { CkGetChareID(&thishandle); }
#endif
};

class Group : public Chare {
  protected:
    int thisgroup;
  public:
    Group() { thisgroup = CkGetGroupID(); }
};

class _CK_CID {
  protected:
    CkChareID _ck_cid;
};

class _CK_GID : public _CK_CID {
  protected:
    int _ck_gid;
};

class CkQdMsg {
  public:
    void *operator new(size_t s) { return CkAllocMsg(0,s,0); }
    void operator delete(void* ptr) { CkFreeMsg(ptr); }
};

#include "ckstream.h"
#include "CkFutures.decl.h"

#endif
