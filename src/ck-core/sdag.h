#ifndef _sdag_H_
#define _sdag_H_

#include "pup.h"

namespace SDAG {
  struct Closure : public PUP::able {
    virtual void pup(PUP::er& p) = 0;
    PUPable_abstract(Closure);
    int continuations;
    bool hasRefnum;
    CMK_REFNUM_TYPE refnum;
    // reference count and self-destruct when no continuations have a reference
    void ref() { continuations++; }
    void deref() { if (--continuations <= 0) delete this; }
    // done this way to keep Closure abstract for PUP reasons
    // these must be called by descendents of Closure
    void packClosure(PUP::er& p) { p | continuations; p | hasRefnum; p | refnum; }
    void init() { continuations = 1; hasRefnum = false; refnum = 0; }
    virtual ~Closure() { }

    // Handles refnum setting for generated code by allowing calls to setRefnum
    // regardless of the type of the first parameter to an entry method.
    // Refnums are arithmetic types only.
    template< typename T, typename std::enable_if< std::is_arithmetic< T >::value, bool >::type = true >
    void setRefnum(T t) { hasRefnum = true; refnum = t; }

    template< typename T, typename std::enable_if< !std::is_arithmetic< T >::value, bool >::type = true >
    void setRefnum(T t) {}

    void unsetRefnum() { hasRefnum = false; refnum = 0; }
  };
}

#include <vector>
#include <list>
#include <unordered_set>
#include <memory>

#include <pup_stl.h>
#include <envelope.h>
#include <debug-charm.h>

class CkMessage;
#if USE_CRITICAL_PATH_HEADER_ARRAY
class MergeablePathHistory;
#endif
namespace SDAG {
  struct ForallClosure : public Closure {
    int val;
    ForallClosure() : val(0) { init(); }
    ForallClosure(CkMigrateMessage*) : val(0) { init(); }
    ForallClosure(int val) : val(val) { init(); }

    void pup(PUP::er& p) {
      p | val;
      packClosure(p);
    }
    PUPable_decl(ForallClosure);
    int& getP0() { return val; }
  };

  struct MsgClosure : public Closure {
    void* msg;

    MsgClosure() : msg(0) { init(); continuations = 0; }
    MsgClosure(CkMigrateMessage*) : msg(0) { init(); continuations = 0; }

    MsgClosure(void* msg)
      : msg(msg) {
      init();
      setRefnum(CkGetRefNum(msg));
      continuations = 0;
      CkReferenceMsg(msg);
    }

    void pup(PUP::er& p) {
      bool hasMsg = msg;
      p | hasMsg;
      if (hasMsg) CkPupMessage(p, (void**)&msg);
      if (hasMsg && p.isUnpacking())
        CkReferenceMsg(msg);
      packClosure(p);
    }

    virtual ~MsgClosure() {
      if (msg) CmiFree(UsrToEnv(msg));
    }

    PUPable_decl(MsgClosure);
  };

  class CCounter : public Closure {
  private:
    unsigned int count;
  public:
    CCounter() { init(); }
    CCounter(CkMigrateMessage*) { init(); }
    CCounter(int c) : count(c) { init(); }
    CCounter(int first, int last, int stride) {
      init();
      count = ((last - first) / stride) + 1;
    }
    void decrement(void) { count--; }
    int isDone(void) { return (count == 0); }

    void pup(PUP::er& p) {
      p | count;
      packClosure(p);
    }
    PUPable_decl(CCounter);
  };

  struct CSpeculator : public Closure {
    int speculationIndex;

    CSpeculator() : speculationIndex(0) { init(); }
    CSpeculator(CkMigrateMessage*) : speculationIndex(0) { init(); }

    CSpeculator(int speculationIndex_)
      : speculationIndex(speculationIndex_) { init(); }

    void pup(PUP::er& p) {
      p | speculationIndex;
      packClosure(p);
    }
    PUPable_decl(CSpeculator);
  };

  struct Continuation : public PUP::able {
    int whenID;
    std::vector<Closure*> closure;
    std::vector<CMK_REFNUM_TYPE> entries, refnums;
    std::vector<int> anyEntries;
    int speculationIndex;
    Continuation() : speculationIndex(-1) { }
    Continuation(CkMigrateMessage*) : speculationIndex(-1) { }

    Continuation(int whenID)
      : whenID(whenID)
      , speculationIndex(-1) { }

    void pup(PUP::er& p) {
      p | whenID;
      p | closure;
      p | entries;
      p | refnums;
      p | anyEntries;
      p | speculationIndex;
    }

    void addClosure(Closure* cl) {
      if (cl) cl->ref();
      closure.push_back(cl);
    }

#if USE_CRITICAL_PATH_HEADER_ARRAY
    MergeablePathHistory *saved;
    void setPath(MergeablePathHistory *tmp) {saved = tmp;}

    MergeablePathHistory *getPath() { return saved;}
#endif

    virtual ~Continuation() {
      for (size_t i = 0; i < closure.size(); i++)
        if (closure[i])
	  closure[i]->deref();
    }

    PUPable_decl(Continuation);
  };

  struct Buffer : public PUP::able {
    int entry;
    Closure* cl;
#if USE_CRITICAL_PATH_HEADER_ARRAY
    MergeablePathHistory *savedPath;
#endif

    Buffer(CkMigrateMessage*) { }

    Buffer(int entry, Closure* cl)
      : entry(entry)
      , cl(cl)
#if USE_CRITICAL_PATH_HEADER_ARRAY
      ,savedPath(NULL)
#endif
    {
      cl->ref();
    }

#if USE_CRITICAL_PATH_HEADER_ARRAY
    void  setPath(MergeablePathHistory* p)
    {
        savedPath = p;
    }

    MergeablePathHistory* getPath() 
    {
        return savedPath;
    }
#endif
    void pup(PUP::er& p) {
      p | entry;
      p | cl;
    }

    virtual ~Buffer() {
      cl->deref();
    }

    PUPable_decl(Buffer);
  };

  struct Dependency {
    std::vector<std::list<int> > entryToWhen;
    std::vector<std::list<Continuation*> > whenToContinuation;

    // entry -> lst of buffers
    // @todo this will have sequential lookup time for specific reference
    // numbers
    std::vector<std::list<Buffer*> > buffer;

    int curSpeculationIndex;

    void pup(PUP::er& p) {
      p | curSpeculationIndex;
      p | entryToWhen;
      p | buffer;
      p | whenToContinuation;
    }

    Dependency(int numEntries, int numWhens)
      : entryToWhen(numEntries)
      , whenToContinuation(numWhens)
      , buffer(numEntries)
      , curSpeculationIndex(0)
      { }

    // Default constructor for migration
    Dependency() { }

    // after a migration free the structures
    ~Dependency() {
      for (std::vector<std::list<Buffer*> >::iterator iter = buffer.begin();
           iter != buffer.end(); ++iter) {
        std::list<Buffer*> lst = *iter;
        for (std::list<Buffer*>::iterator iter2 = lst.begin();
             iter2 != lst.end(); ++iter2) {
          delete *iter2;
        }
      }

      for (size_t i = 0; i < whenToContinuation.size(); i++) {
        for (std::list<Continuation*>::iterator iter2 = whenToContinuation[i].begin();
             iter2 != whenToContinuation[i].end(); ++iter2) {
          delete *iter2;
        }
      }
    }

    void addDepends(int whenID, int entry) {
      entryToWhen[entry].push_back(whenID);
    }

    void reg(Continuation *c) {
      //printf("registering new continuation %p, whenID = %d\n", c, c->whenID);
      whenToContinuation[c->whenID].push_back(c);
    }

    void dereg(Continuation *c) {
      CkAssert(c->whenID < (int)whenToContinuation.size());
      std::list<Continuation*>& lst = whenToContinuation[c->whenID];
      lst.remove(c);
    }

    Buffer* pushBuffer(int entry, Closure *cl) {
      Buffer* buf = new Buffer(entry, cl);
      buffer[entry].push_back(buf);
      return buf;
    }

    Continuation *tryFindContinuation(int entry) {
      for (std::list<int>::iterator iter = entryToWhen[entry].begin();
           iter != entryToWhen[entry].end();
           ++iter) {
        int whenID = *iter;

        for (std::list<Continuation*>::iterator iter2 = whenToContinuation[whenID].begin();
             iter2 != whenToContinuation[whenID].end();
             iter2++) {
          Continuation* c = *iter2;
          if (searchBufferedMatching(c)) {
            dereg(c);
            return c;
          }
        }
      }
      //printf("no continuation found\n");
      return 0;
    }

    bool searchBufferedMatching(Continuation* t) {
      CkAssert(t->entries.size() == t->refnums.size());
      for (size_t i = 0; i < t->entries.size(); i++) {
        if (!tryFindMessage(t->entries[i], true, t->refnums[i], 0)) {
          return false;
        }
      }
      for (size_t i = 0; i < t->anyEntries.size(); i++) {
        if (!tryFindMessage(t->anyEntries[i], false, 0, 0)) {
          return false;
        }
      }
      return true;
    }

    Buffer* tryFindMessage(int entry, bool hasRef, CMK_REFNUM_TYPE refnum, std::unordered_set<Buffer*>* ignore) {
      // @todo sequential lookup for buffer with reference number or ignore set
      for (std::list<Buffer*>::iterator iter = buffer[entry].begin();
           iter != buffer[entry].end();
           ++iter) {
        if ((!hasRef || ((*iter)->cl->hasRefnum && (*iter)->cl->refnum == refnum)) &&
            (!ignore || ignore->find(*iter) == ignore->end()))
          return *iter;
      }
      return 0;
    }

    Buffer* tryFindMessage(int entry) {
      if (buffer[entry].size() == 0)
        return 0;
      else
        return buffer[entry].front();
    }

    void removeMessage(Buffer *buf) {
      buffer[buf->entry].remove(buf);
    }

    int getAndIncrementSpeculationIndex() {
      return curSpeculationIndex++;
    }

    void removeAllSpeculationIndex(int speculationIndex) {
      for (std::vector<std::list<Continuation*> >::iterator iter = whenToContinuation.begin();
           iter != whenToContinuation.end();
           ++iter) {
        std::list<Continuation*>& lst = *iter;

        for (std::list<Continuation*>::iterator iter2 = lst.begin();
             iter2 != lst.end();
	     //cppcheck-suppress StlMissingComparison
             ) {
          if ((*iter2)->speculationIndex == speculationIndex) {
            Continuation *cancelled = *iter2;
	    //cppcheck-suppress StlMissingComparison
            iter2 = lst.erase(iter2);
            delete cancelled;
          } else {
            iter2++;
          }
        }
      }
    }
  };

  typedef std::unique_ptr<Dependency> dep_ptr;

  void registerPUPables();
}

#endif
