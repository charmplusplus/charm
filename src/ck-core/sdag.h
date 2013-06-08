#ifndef _sdag_H_
#define _sdag_H_

#include "charm++.h"
#include <vector>
#include <list>
#include <map>
#include <set>

struct PackableParams {
  virtual void pup(PUP::er& p) = 0;
  virtual int getType() = 0;
};

struct TransportableEntity : public PackableParams {
  enum PType { TransportableEntityType,
               TransportableMsgType,
               TransportableCounterType,
               TransportableSpeculatorType,
               TransportableBigSimLogType };

  PType type;

  TransportableEntity() : type(TransportableEntityType) { };
  TransportableEntity(PType type) : type(type) { };

  void pupType(PUP::er& p) {
    int t = (int)type;
    p | t;
    type = (PType)t;
  }

  virtual int getType() { return 0; }

  virtual void pup(PUP::er& p) {
    pupType(p);
  }
};

struct TransportableBigSimLog : public TransportableEntity {
  void* log;
  TransportableBigSimLog() : log(0) { }

  TransportableBigSimLog(void* log)
    : TransportableEntity(TransportableBigSimLogType)
    , log(log) { }

  void pup(PUP::er& p) {
    TransportableEntity::pupType(p);
    if (p.isUnpacking()) log = 0;
  }
};

struct MsgClosure : public TransportableEntity {
  void* msg;

  MsgClosure() : msg(0) { }

  MsgClosure(CkMessage* msg)
    : TransportableEntity(TransportableMsgType)
    , msg(msg) { }

  void pup(PUP::er& p) {
    TransportableEntity::pupType(p);
    bool isNull = !msg;
    p | isNull;
    if (!isNull) CkPupMessage(p, &msg);
  }
};

class CCounter : public TransportableEntity {
private:
  unsigned int count;
public:
  CCounter() : TransportableEntity(TransportableCounterType) { }
  CCounter(int c) : TransportableEntity(TransportableCounterType), count(c) { }
  CCounter(int first, int last, int stride)
    : TransportableEntity(TransportableCounterType) {
    count = ((last - first) / stride) + 1;
  }
  void decrement(void) { count--; }
  int isDone(void) { return (count == 0); }

  void pup(PUP::er& p) {
    TransportableEntity::pupType(p);
    p | count;
  }
};

struct CSpeculator : public TransportableEntity {
  int speculationIndex;

  CSpeculator() : TransportableEntity(TransportableSpeculatorType) { }

  CSpeculator(int speculationIndex_)
    : TransportableEntity(TransportableSpeculatorType)
    , speculationIndex(speculationIndex_) { }

  void pup(PUP::er& p) {
    TransportableEntity::pupType(p);
    p | speculationIndex;
  }
};

namespace SDAG {
  struct Continuation {
    int whenID;
    std::vector<PackableParams*> closure;
    std::vector<int> entries, refnums;
    std::vector<int> anyEntries;
    int speculationIndex;

    Continuation() { }

    Continuation(int whenID)
      : whenID(whenID) { }

    void pup(PUP::er& p) {
      //p | whenID;
      //p | args;
      //p | entries;
      //p | refnums;
      //p | anyEntries;
      //p | speculationIndex;
    }
  };

  struct Buffer {
    int entry, refnum;
    PackableParams* packable;

    Buffer(int entry, PackableParams* packable, int refnum)
      : entry(entry)
      , packable(packable)
      , refnum(refnum) { }
  };

  struct Dependency {
    std::vector<std::list<int> > entryToWhen;
    std::map<int, std::list<Continuation*> > whenToContinuation;

    // entry -> lst of buffers
    std::vector<std::list<Buffer*> > buffer;

    int curSpeculationIndex;

    void pup(PUP::er& p) {
      p | curSpeculationIndex;
    }

    Dependency(int numEntries, int numWhens)
      : curSpeculationIndex(0)
      , buffer(numEntries)
      , entryToWhen(numEntries) { }

    void addDepends(int whenID, int entry) {
      entryToWhen[entry].push_back(whenID);
    }

    void reg(Continuation *c) {
      //printf("registering new continuation %p, whenID = %d\n", c, c->whenID);
      whenToContinuation[c->whenID].push_back(c);
    }

    void dereg(Continuation *c) {
      if (whenToContinuation.find(c->whenID) != whenToContinuation.end()) {
        std::list<Continuation*>& lst = whenToContinuation[c->whenID];
        lst.remove(c);
      } else {
        CkAbort("trying to deregister: continuation not found");
      }
    }

    Buffer* pushBuffer(int entry, PackableParams *packable, int refnum) {
      Buffer* buf = new Buffer(entry, packable, refnum);
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
            //printf("found matching continuation %p\n", t);
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
      for (int i = 0; i < t->entries.size(); i++) {
        if (!tryFindMessage(t->entries[i], true, t->refnums[i], false)) {
          return false;
        }
      }
      for (int i = 0; i < t->anyEntries.size(); i++) {
        if (!tryFindMessage(t->anyEntries[i], false, 0, false)) {
          return false;
        }
      }
      return true;
    }

    Buffer* tryFindMessage(int entry, bool hasRef, int refnum, bool hasIgnore,
                           std::set<Buffer*> ignore = std::set<Buffer*>()) {
      if (buffer[entry].size() == 0) return 0;
      else {
        for (std::list<Buffer*>::iterator iter = buffer[entry].begin();
             iter != buffer[entry].end();
             ++iter) {
          if ((!hasRef || (*iter)->refnum == refnum) &&
              (!hasIgnore || ignore.find(*iter) == ignore.end()))
            return *iter;
        }
        return 0;
      }
    }

    Buffer* tryFindMessage(int entry) {
      if (buffer[entry].size() == 0) return 0;
      else {
        Buffer* buf = buffer[entry].front();
        //printf("found buffered message %p\n", buf);
        return buffer[entry].front();
      }
    }

    void removeMessage(Buffer *buf) {
      buffer[buf->entry].remove(buf);
    }

    int getAndIncrementSpeculationIndex() {
      return curSpeculationIndex++;
    }

    void removeAllSpeculationIndex(int speculationIndex) {
      for (std::map<int, std::list<Continuation*> >::iterator iter = whenToContinuation.begin();
           iter != whenToContinuation.end();
           ++iter) {
        std::list<Continuation*>& lst = iter->second;

        for (std::list<Continuation*>::iterator iter2 = lst.begin();
             iter2 != lst.end();
             ++iter2) {
          if ((*iter2)->speculationIndex == speculationIndex) {
            Continuation *cancelled = *iter2;
            iter2 = lst.erase(iter2);
            delete cancelled;
          }
        }
      }
    }
  };
}

class CMsgBuffer {
public:
  int entry;
  void *msg;
  void *bgLog1;
  void *bgLog2;
  int refnum;
  CMsgBuffer *next;

  CMsgBuffer(int e, void *m, void* l1, int r) : entry(e), msg(m), bgLog1(l1), bgLog2(NULL),refnum(r), next(NULL) {}
  CMsgBuffer(int e, void *m, int r) : entry(e), msg(m), bgLog1(NULL), bgLog2(NULL),refnum(r), next(NULL) {}
  CMsgBuffer(): bgLog1(NULL), bgLog2(NULL), next(NULL) {}
  void pup(PUP::er& p) {
    p|entry;
    CkPupMessage(p, &msg);
    p|refnum;
    if (p.isUnpacking()) {
      bgLog1 = bgLog2 = NULL;
    }
  }
};


#define MAXARG 8
#define MAXANY 8
#define MAXREF 8

class CWhenTrigger {
  public:
    int whenID, nArgs;
    TransportableEntity* args[MAXARG];
    int nAnyEntries;
    int anyEntries[MAXANY];
    int nEntries;
    int entries[MAXREF];
    int refnums[MAXREF];
    int speculationIndex;

    CWhenTrigger *next;
    CWhenTrigger(int id, int na, int ne, int nae) :
      whenID(id), nArgs(na), nAnyEntries(nae), nEntries(ne), speculationIndex(-1), next(NULL) { init(); }
    CWhenTrigger(): next(NULL) { init(); }
    void init() {
      for (int i = 0; i < MAXARG; i++) args[i] = NULL;
    }
    void pup(PUP::er& p) {
      p|whenID;
      p|nArgs;
      p|nAnyEntries;
      p(anyEntries, MAXANY);
      p|nEntries;
      p(entries, MAXREF);
      p(refnums, MAXREF);

      for (int i = 0; i < MAXARG; i++) {
        bool isNull = !args[i];
        p | isNull;

        if (!isNull) {
          int t = p.isUnpacking() ? -1 : args[i]->type;
          p | t;

          if (p.isUnpacking()) {
            switch ((TransportableEntity::PType)t) {
            case TransportableEntity::TransportableEntityType: args[i] = new TransportableEntity(); break;
            case TransportableEntity::TransportableMsgType: args[i] = new MsgClosure(); break;
            case TransportableEntity::TransportableCounterType: args[i] = new CCounter(); break;
            case TransportableEntity::TransportableSpeculatorType: args[i] = new CSpeculator(); break;
            case TransportableEntity::TransportableBigSimLogType: args[i] = new TransportableBigSimLog(); break;
            }
          }

          // use virtual dispatch
          args[i]->pup(p);
        } else {
          args[i] = NULL;
        }
      }
      p|speculationIndex;
    }
};

// Quick and dirty List for small numbers of items.
// It should ideally be a template, but in order to have portability,
// we would make it two lists

class TListCWhenTrigger
{
  private:

    CWhenTrigger *first, *last;
    CWhenTrigger *current;

  public:

    TListCWhenTrigger(void) : first(0), last(0), current(0) {;}

    void pup(PUP::er& p) {
      int nEntries=0;
      int cur=-1;
      if (!p.isUnpacking()) { 
        for (CWhenTrigger *tmp = first; tmp; tmp=tmp->next, nEntries++)
          if (current == tmp) cur = nEntries;
      }
      p|nEntries;
      p|cur;
      if (p.isUnpacking()) { 
        first = last = current = NULL;
        if (nEntries) {
	  CWhenTrigger** unpackArray = new CWhenTrigger*[nEntries]; 
          for (int i=0; i<nEntries; i++)  {
            unpackArray[i] = new CWhenTrigger;
            if (i!=0) unpackArray[i-1]->next=unpackArray[i];
          }
          first = unpackArray[0];
          last = unpackArray[nEntries-1];
          last->next = NULL;
          current = cur==-1?NULL:unpackArray[cur];
          delete [] unpackArray;
        }
      }
      for (CWhenTrigger *tmp = first; tmp; tmp=tmp->next) tmp->pup(p);
    }

    int empty(void) { return ! first; }
    
    CWhenTrigger *begin(void) {
      return (current = first);
    }

    int end(void) {
      return (current == 0);
    }

    CWhenTrigger *next (void) {
      return (current = current->next);
    }

    CWhenTrigger *front(void)
    {
      return first;
    }

    void remove(CWhenTrigger *data)
    {
      // case 1: empty list
      if (first == 0)
        return;
      // case 2: first element to be removed
      if(first == data) {
        first = first->next;
	if(first==0) last=0;
        return;
      }
      // case 3: middle or last element to be removed
      CWhenTrigger *nn;
      CWhenTrigger *prev = first;
      for(nn=first->next; nn; nn = nn->next) {
        if (nn == data) {
          prev->next = nn->next;
	  if(nn==last)
	    last=prev;
          return;
        }
        prev = nn;
      }
    }

    void append(CWhenTrigger *data)
    {
      data->next = 0;
      if(first == 0) {
        last = first = data;
      } else {
        last->next = data;
	last = last->next;
      }
    }
};

class TListCMsgBuffer
{
  private:

    CMsgBuffer *first, *last;
    CMsgBuffer *current;

  public:

    TListCMsgBuffer(void) : first(0), last(0), current(0) {}

    void pup(PUP::er& p) {
      int nEntries=0;
      int cur=0;
      if (!p.isUnpacking()) { 
        for (CMsgBuffer *tmp = first; tmp; tmp=tmp->next, nEntries++) {
          if (current == tmp) cur = nEntries;
        }
      }
      p|nEntries;
      p|cur;
      if (p.isUnpacking()) { 
        first = last = current = NULL;
        if (nEntries) {
	  CMsgBuffer** unpackArray = new CMsgBuffer*[nEntries]; 
          for (int i=0; i<nEntries; i++)  {
            unpackArray[i] = new CMsgBuffer;
            if (i!=0) unpackArray[i-1]->next=unpackArray[i];
          }
          first = unpackArray[0];
          last = unpackArray[nEntries-1];
          current = unpackArray[cur];
	  delete [] unpackArray;
        }
      }
      for (CMsgBuffer *tmp = first; tmp; tmp=tmp->next) tmp->pup(p);
    }

    int empty(void) { return ! first; }
    
    CMsgBuffer *begin(void) {
      return (current = first);
    }

    int end(void) {
      return (current == 0);
    }

    CMsgBuffer *next (void) {
      return (current = current->next);
    }

    CMsgBuffer *front(void)
    {
      return first;
    }

    void remove(CMsgBuffer *data)
    {
      // case 1: empty list
      if (first == 0)
        return;
      // case 2: first element to be removed
      if(first == data) {
        first = first->next;
	if(first==0) last=0;
        return;
      }
      // case 3: middle or last element to be removed
      CMsgBuffer *nn;
      CMsgBuffer *prev = first;
      for(nn=first->next; nn; nn = nn->next) {
        if (nn == data) {
          prev->next = nn->next;
	  if(nn==last)
	    last=prev;
          return;
        }
        prev = nn;
      }
    }

    void append(CMsgBuffer *data)
    {
      data->next = 0;
      if(first == 0) {
        last = first = data;
      } else {
        last->next = data;
	last = last->next;
      }
    }
};


/**
 This class hides all the details of dependencies between
 when blocks and entries. It also contains the entry buffers
 and when triggers.
*/

class CDep {
   int numEntries, numWhens;
   TListCWhenTrigger **whens;
   TListCMsgBuffer **buffers;
   int *numWhenDepends;
   int *numEntryDepends;
   TListCMsgBuffer ***whenDepends;
   TListCWhenTrigger ***entryDepends;
   int curSpeculationIndex;

 public:
   void pup(PUP::er& p) {
     /* 
        no need for initMem() because __sdag_pup() will take care of 
        allocating of CDep and call addDepends(), so we don't pup whenDepends
        and entryDepends here.
     */ 
     int i;

     for (i=0; i<numWhens; i++)    whens[i]->pup(p);
     for (i=0; i<numEntries; i++)  buffers[i]->pup(p);

     p(numWhenDepends, numWhens);
     p(numEntryDepends, numEntries);

     p | curSpeculationIndex;
/*
     // don't actually pack this info because it gets created once 
     // the addDepends() in the initialization scheme are called for this class
     for (i=0; i<numWhens; i++)
       for (j=0; j<numWhenDepends[i]; j++) {
         int which;
         if (p.isPacking())  which = whenDepends[i][j] - buffers[0];
         p|which;
         if (p.isUnpacking()) whenDepends[i][j] = buffers[which];
       }

     for (i=0; i<numEntries; i++)
       for (j=0; j<numEntryDepends[i]; j++) {
         int which;
         if (p.isPacking())  which = entryDepends[i][j] - whens[0];
         p|which;
         if (p.isUnpacking()) entryDepends[i][j] = whens[which];
     }
*/
   }

   CDep(int ne, int nw) : numEntries(ne), numWhens(nw), curSpeculationIndex(0) { initMem(); }

   ~CDep() {
     int i;
     delete [] numWhenDepends;
     delete [] numEntryDepends;
     for(i=0;i<numWhens;i++) {
       delete whens[i];
       delete [] whenDepends[i];
     }
     for(i=0;i<numEntries;i++) {
       delete buffers[i];
       delete [] entryDepends[i];
     }
     delete [] whens;
     delete [] buffers;
     delete [] whenDepends;
     delete [] entryDepends;
   }

 private:
   void initMem() {
     // initialize the internal data structures here
     whens = new TListCWhenTrigger *[numWhens];
     buffers = new TListCMsgBuffer *[numEntries];
     numWhenDepends = new int[numWhens];
     numEntryDepends = new int[numEntries];
     whenDepends = new TListCMsgBuffer **[numWhens];
     entryDepends = new TListCWhenTrigger **[numEntries];
     int i;
     for(i=0;i<numWhens;i++) {
       whens[i] = new TListCWhenTrigger();
       whenDepends[i] = new TListCMsgBuffer *[numEntries];
       numWhenDepends[i] = 0;
     }
     for(i=0;i<numEntries;i++) {
       buffers[i] = new TListCMsgBuffer();
       entryDepends[i] = new TListCWhenTrigger *[numWhens];
       numEntryDepends[i] = 0;
     }
   }

 public:
   // adds a dependency of whenID upon Entry
   // done only at initialization.
   void addDepends(int whenID, int entry) {
     whenDepends[whenID][numWhenDepends[whenID]++] = buffers[entry];
     entryDepends[entry][numEntryDepends[entry]++] = whens[whenID];
   }

   // register a trigger to be called with
   // with <nEntries> specified
   // in <entries> with corresponding <refnums>
   void Register(CWhenTrigger *trigger)
   {
     whens[trigger->whenID]->append(trigger);
   }

   // deregister trigger from all
   // the entries it is registered for
   void deRegister(CWhenTrigger *trigger) {
     whens[trigger->whenID]->remove(trigger);
   }

   // buffer a message for a specific entry point with a specified
   // reference number
   CMsgBuffer* bufferMessage(int entry, void *msg, void* log , int refnum)
   {
     CMsgBuffer *buf = new CMsgBuffer(entry, msg, log,refnum);
     buffers[entry]->append(buf);
     return buf;
   }

   // For a specified entry number and reference number,
   // get the registered trigger which satisfies dependency. 
   // If no trigger exists
   // for the given reference number, get the trigger registered for
   // ANY ref num. If that also doesnt exist, Return NULL
   CWhenTrigger *getTrigger(int entry, int refnum)
   {
     for(int i=0;i<numEntryDepends[entry];i++) {
       TListCWhenTrigger *wlist = entryDepends[entry][i];
       for(CWhenTrigger *elem=wlist->begin(); 
           !wlist->end(); 
           elem=wlist->next()) {
         if(elem==0)
           break;
         if(depSatisfied(elem)){
            deRegister(elem);
            return elem;
         }
       }
     }
     return 0;
   }


   // given the entry number and reference number,
   // get the buffered message, without removing it from
   // the list, NULL if no such message exists
   CMsgBuffer *getMessage(int entry, int refnum, std::set<CMsgBuffer*> ignore)
   {
     TListCMsgBuffer *list = buffers[entry];
     for(CMsgBuffer *elem=list->begin(); !list->end(); elem=list->next()) {
       if(elem == 0)
         return 0;
       if(elem->refnum == refnum && ignore.find(elem) == ignore.end())
         return elem;
     }
     return 0;
   }

   // given the entry number,
   // get the buffered message, without removing it from
   // the list, NULL if no such message exists
   // note that this is the ANY case
   CMsgBuffer *getMessage(int entry, std::set<CMsgBuffer*> ignore)
   {
     TListCMsgBuffer *list = buffers[entry];
     for(CMsgBuffer *elem=list->begin(); !list->end(); elem=list->next()) {
       if(elem == 0)
         return 0;
       if(ignore.find(elem) == ignore.end())
         return elem;
     }
     return 0;
   }

   CMsgBuffer *getMessageSingle(int entry, int refnum) {
     TListCMsgBuffer *list = buffers[entry];
     for(CMsgBuffer *elem=list->begin(); !list->end(); elem=list->next()) {
       if(elem == 0)
         return 0;
       if(elem->refnum == refnum)
         return elem;
     }
     return 0;
   }

   CMsgBuffer *getMessageSingle(int entry) {
     return buffers[entry]->front();
   }

   // remove the given message from buffer
   void removeMessage(CMsgBuffer *msg)
   {
     TListCMsgBuffer *list = buffers[msg->entry];
     list->remove(msg);
   }

   // return 1 if all the dependeces for trigger are satisfied
   // return 0 otherwise
   int depSatisfied(CWhenTrigger *trigger)
   {
     int i;
     for(i=0;i<trigger->nEntries;i++) {
       if(!getMessageSingle(trigger->entries[i], trigger->refnums[i]))
         return 0;
     }
     for(i=0;i<trigger->nAnyEntries;i++) {
       if(!getMessageSingle(trigger->anyEntries[i]))
         return 0;
     }
     return 1;
   }

   int getAndIncrementSpeculationIndex() {
     return curSpeculationIndex++;
   }

   void removeAllSpeculationIndex(int speculationIndex) {
     for (int i = 0; i < numWhens; i++) {
       TListCWhenTrigger *wlist = whens[i];
       CWhenTrigger *elem = wlist->begin();
       while (elem && !wlist->end()) {
         if (elem->speculationIndex == speculationIndex) {
           CWhenTrigger *cancelled = elem;
           deRegister(elem);
           elem = wlist->next();
           delete cancelled;
         } else {
           elem = wlist->next();
         }
       }
     }
   }

};


/** 
 This class hides all of the details of dependencies between
 overlap blocks and when blocks. 
 */

class COverDep {

   int numOverlaps, numWhens;
   TListCWhenTrigger **whens;
   int *numOverlapDepends;
   TListCWhenTrigger ***overlapDepends;
   
  public:
     void pup(PUP::er& p) {
        /*
          no need for initMem() because __sdag_pup() will take care of
          allocating of COverDep and call addOverlapDepends(), so we don't pup overlapsDepends here
        */
        int i; // , j;
        for (i=0; i<numWhens; i++)    whens[i]->pup(p);

        p(numOverlapDepends, numOverlaps);
     }

     COverDep(int no, int nw) : numOverlaps(no), numWhens(nw) { initMem(); }

     ~COverDep() {
        int i;
        delete [] numOverlapDepends;
        for(i=0;i<numWhens;i++) {
            delete whens[i];
        }
        for(i=0;i<numOverlaps;i++) {
            delete [] overlapDepends[i];

        }
	delete [] whens;
	delete [] overlapDepends;
     }
     
   private:
     void initMem() {
       // initialize the internal data structures here
       whens = new TListCWhenTrigger *[numWhens];
       numOverlapDepends = new int[numOverlaps];
       overlapDepends = new TListCWhenTrigger **[numOverlaps];
       int i;
       for(i=0;i<numWhens;i++) {
         whens[i] = new TListCWhenTrigger();
       }
       for(i=0;i<numOverlaps;i++) {
         overlapDepends[i] = new TListCWhenTrigger *[numWhens];
         numOverlapDepends[i] = 0;
       }
     }

   public:
     //adds a dependency of the whenID for each Overlap
     // done only at initialization
     void addOverlapDepends(int whenID, int overlap) {
       overlapDepends[overlap][whenID] = whens[whenID];
       //overlapDepends[overlap][numOverlapDepends[overlap]++] = whens[whenID];
     }
     
     // register a trigger to be called with
     // with <nEntries> specified
     // in <entries> with corresponding <refnums>
     void Register(CWhenTrigger *trigger)
     {
       whens[trigger->whenID]->append(trigger);
     }
     
     // deregister trigger from all
     // the entries it is registered for
     void deRegister(CWhenTrigger *trigger)
     {
        whens[trigger->whenID]->remove(trigger);
     }
     
     
     // For a specified entry number and reference number,
     // get the registered trigger which satisfies dependency.
     // If no trigger exists
     // for the given reference number, get the trigger registered for
     // ANY ref num. If that also doesnt exist, Return NULL
     CWhenTrigger *getTrigger(int overlapID, int whenID)
     {
        TListCWhenTrigger *wlist = overlapDepends[overlapID][whenID];
        CWhenTrigger *elem=wlist->begin(); 
        if (elem == 0)
          return 0;	  
        else {
          deRegister(elem);
	  return elem;
        }
     }
};

#endif
