#ifndef _INIT_H
#define _INIT_H


typedef CkQ<void *> PtrQ;
typedef CkVec<void *> PtrVec;

class GroupTable {
  enum {MAXBINS=256};
  class TableEntry {
    public:
      void *obj;
      PtrQ *pending; //Buffers msgs recv'd before group is created
      TableEntry() { 
        obj=0; pending=0;
      }
  };
  TableEntry tab[MAXBINS];
  public:
    GroupTable();
    void add(CkGroupID n, void *obj);
    void enqmsg(CkGroupID n, void *msg);
    PtrQ *getPending(CkGroupID n) {
      PtrQ *ret=tab[n].pending;
      tab[n].pending=0;
      return ret;
    }
    void *find(CkGroupID n) {return tab[n].obj;}
};

extern unsigned int    _printCS;
extern unsigned int    _printSS;

extern unsigned int    _numGroups;
extern unsigned int    _numNodeGroups;
extern int     _infoIdx;
extern unsigned int    _numInitMsgs;
extern unsigned int    _numInitNodeMsgs;
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
CpvExtern(GroupTable, _groupTable);

static inline void *_localBranch(int gID)
{
  return CpvAccess(_groupTable).find(gID);
}

extern GroupTable*    _nodeGroupTable;

extern void _initCharm(int argc, char **argv);

#endif


