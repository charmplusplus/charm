/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CHARMPP_H_
#define _CHARMPP_H_

#include <stdlib.h>
#include "charm.h"

#include "pup.h"

#if CMK_DEBUG_MODE
#include <string.h>
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

class _CK_NGID : public _CK_GID {
};

class CkArray;

class CkArrayID {
  public:
    CkGroupID _aid;
    CkArrayID(CkGroupID aid) {
      _aid=aid;
    }
    CkArrayID() {}
};

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
  if(cid.magic%_GETIDX(idx))
    CkAbort("Illegal ChareID assignment to proxy.\n");
}
#else
static inline void _CHECK_CID(CkChareID, int){}
#endif

/* These came from init.h-- OSL, 3/20/2000 */

template <class T>
class CkQ {
    T *block;
    int blklen;
    int first;
    int len;
  public:
    CkQ() :first(0),len(0) {
      block = new T[blklen=16];
    }
    ~CkQ() { delete[] block; }
    int length(void) { return len; }
    T deq(void) {
      if(len>0) {
        T &ret = block[first];
        first = (first+1)%blklen;
        len--;
      	return ret;
      } else return T(0);
    }
    void enq(const T &elt) {
      if(len==blklen) {
      	int newlen=len*2;
        T *newblk = new T[newlen];
        memcpy(newblk, block+first, sizeof(T)*(blklen-first));
        memcpy(newblk+blklen-first, block, sizeof(T)*first);
        delete[] block; block = newblk;
        blklen = newlen; first = 0;
      }
      block[(first+len)%blklen] = elt;
      len++;
    }
    //Peek at the n'th item from the queue
    T& operator[](size_t n) 
    {
    	n=(n+first)%blklen;
    	return block[n];
    }
};

template <class T>
class CkVec {
    T *block;
    int blklen,len;
  public:
    CkVec() {block=NULL;blklen=len=0;}
    ~CkVec() { delete[] block; }
    int &length(void) { return len; }
    T *getVec(void) { return block; }
    T& operator[](size_t n) { return block[n]; }
    const T& operator[](size_t n) const { return block[n]; }
    void insert(int pos, const T &elt) {
      if(pos>=blklen) {
      	int newlen=pos*2+16;//Double length at each step
        T *newblk = new T[newlen];
        if (block!=NULL)
        	memcpy(newblk, block, sizeof(T)*blklen);
        for(int i=blklen; i<newlen; i++) newblk[i] = T(0);
        delete[] block; block = newblk;
        blklen = newlen;
      }
      if (pos>=len) len=pos+1;
      block[pos] = elt;
    }
    void insertAtEnd(const T &elt) {insert(length(),elt);}
};


#include "ckarray.h"
#include "ckstream.h"
#include "CkFutures.decl.h"
#include "tempo.h"
#include "waitqd.h"

#endif
