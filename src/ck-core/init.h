#ifndef _INIT_H
#define _INIT_H


typedef CkQ<void *> PtrQ;
typedef CkVec<void *> PtrVec;

class GroupTable {
  enum {MAXBINS=32};
  class TableEntry {
    public:
      CkGroupID num;
      void *obj;
      PtrQ *pending;
      TableEntry *next;
      TableEntry(CkGroupID _num, void *_obj, TableEntry* _next=0) :
        num(_num), obj(_obj), next(_next) { 
        pending = ((obj)?0:(new PtrQ()));
      }
  };
  TableEntry *bins[MAXBINS];
  public:
    GroupTable();
    void add(CkGroupID n, void *obj) {
      TableEntry *ptr = locate(n);
      if(ptr) {
        ptr->obj = obj;
        return;
      }
      int slot = n%MAXBINS;
      bins[slot] = new TableEntry(n, obj, bins[slot]);
      _MEMCHECK(bins[slot]);
    }
    void enqmsg(CkGroupID n, void *msg) {
      TableEntry *ptr = locate(n);
      if(!ptr) {
        int slot = n%MAXBINS;
        ptr = bins[slot] = new TableEntry(n, 0, bins[slot]);
        _MEMCHECK(bins[slot]);
      }
      ptr->pending->enq(msg);
    }
    PtrQ *getPending(CkGroupID n) {
      TableEntry *ptr = locate(n);
      return ((ptr)?(ptr->pending):0);
    }
    TableEntry *locate(CkGroupID n) {
      TableEntry *next = bins[n%MAXBINS];
      while(next!=0) {
        if(next->num == n)
          return next;
        next = next->next;
      }
      return 0;
    }
    void *find(CkGroupID n) {
      TableEntry *ptr = locate(n);
      return ((ptr)? ptr->obj : 0);
    }
};

extern UInt    _printCS;
extern UInt    _printSS;

extern UInt    _numGroups;
extern UInt    _numNodeGroups;
extern int     _infoIdx;
extern UInt    _numInitMsgs;
extern UInt    _numInitNodeMsgs;
extern int     _charmHandlerIdx;
extern int     _initHandlerIdx;
extern int     _bocHandlerIdx;
extern int     _nodeBocHandlerIdx;
extern int     _qdHandlerIdx;
extern CmiNodeLock _nodeLock;

CpvExtern(void*,       _currentChare);
CpvExtern(int,       _currentChareType);
CpvExtern(CkGroupID,         _currentGroup);
CpvExtern(CkGroupID,         _currentNodeGroup);
CpvExtern(GroupTable*, _groupTable);

extern GroupTable*    _nodeGroupTable;

extern void _initCharm(int argc, char **argv);
extern void _processBocInitMsg(envelope *);
extern void _processNodeBocInitMsg(envelope *);
#endif
