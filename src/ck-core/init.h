#ifndef _INIT_H
#define _INIT_H

#define BLKSZ 32
class PtrQ {
    void **block;
    int blklen;
    int first;
    int len;
  public:
    PtrQ() :len(0), first(0) {
      block = new void*[blklen=BLKSZ];
    }
    ~PtrQ() { delete[] block; }
    int length(void) { return len; }
    void *deq(void) {
      void *ret=0;
      if(len>0) {
        len--;
        ret = block[first];
        first = (first+1)%blklen;
      }
      return ret;
    }
    void enq(void *elt) {
      if(len==blklen) {
        void **newblk = new void*[blklen+BLKSZ];
        memcpy(newblk, block+first, sizeof(void*)*(blklen-first));
        memcpy(newblk+blklen-first, block, sizeof(void*)*first);
        delete[] block; block = newblk;
        blklen += BLKSZ; first = 0;
      }
      block[(first+len)%blklen] = elt;
      len++;
      return;
    }
};

// const inst MAXBINS = 32; in GroupTable is not allowed on solaris CC

#define MAXBINS 32

class GroupTable {
  class TableEntry {
    public:
      CkGroupID num;
      void *obj;
      TableEntry *next;
      TableEntry(CkGroupID _num, void *_obj, TableEntry* _next=0) :
        num(_num), obj(_obj), next(_next) {}
  };
  TableEntry *bins[MAXBINS];
  public:
    GroupTable();
    void add(CkGroupID n, void *obj) {
      int slot = n%MAXBINS;
      bins[slot] = new TableEntry(n, obj, bins[slot]);
    }
    void *find(CkGroupID n) {
      TableEntry *next = bins[n%MAXBINS];
      while(next!=0) {
        if(next->num == n)
          return next->obj;
        next = next->next;
      }
      return 0;
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
CpvExtern(CkGroupID,         _currentGroup);
CpvExtern(CkGroupID,         _currentNodeGroup);
CpvExtern(GroupTable*, _groupTable);

extern GroupTable*    _nodeGroupTable;

extern void _initCharm(int argc, char **argv);
extern void _processBocInitMsg(envelope *);
extern void _processNodeBocInitMsg(envelope *);
#endif
