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
      int num;
      void *obj;
      TableEntry *next;
      TableEntry(int _num, void *_obj, TableEntry* _next=0) :
        num(_num), obj(_obj), next(_next) {}
  };
  TableEntry *bins[MAXBINS];
  public:
    GroupTable();
    void add(int n, void *obj) {
      int slot = n%MAXBINS;
      bins[slot] = new TableEntry(n, obj, bins[slot]);
    }
    void *find(int n) {
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
extern int     _infoIdx;
extern UInt    _numInitMsgs;
extern int     _charmHandlerIdx;
extern int     _initHandlerIdx;
extern int     _bocHandlerIdx;
extern int     _qdHandlerIdx;

CpvExtern(void*,       _currentChare);
CpvExtern(int,         _currentGroup);
CpvExtern(GroupTable*, _groupTable);

extern void _initCharm(int argc, char **argv);
extern void _processBocInitMsg(envelope *);

#endif
