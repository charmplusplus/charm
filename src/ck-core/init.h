#ifndef _INIT_H
#define _INIT_H

#include <charm.h> // For CkNumPes

typedef CkQ<void *> PtrQ;
typedef CkVec<void *> PtrVec;

class IrrGroup;
class TableEntry {
    IrrGroup *obj;
    PtrQ *pending; //Buffers msgs recv'd before group is created
  public:
    void init(void) { obj=0; pending=0; }
    IrrGroup* getObj(void) { return obj; }
    void setObj(void *_obj) { obj=(IrrGroup *)_obj; }
    PtrQ* getPending(void) { return pending; }
    void enqMsg(void *msg) {
      if (pending==0)
        pending=new PtrQ();
      pending->enq(msg);
    }
};

template <class dtype>
class GroupIdxArrayEntry {
  dtype *tab;
  int max;
  public:
    GroupIdxArrayEntry() { tab=NULL; max=0; }
    void init(int _max) {
      max = _max;
      tab = new dtype[max];
      for(int i=0;i<max;i++)
       tab[i].init(); 
    }
    dtype &find(int n) {
#ifndef CMK_OPTIMIZE
       if (n<0) CkAbort("Group ID is negative-- invalid!\n");
       if (n==0) CkAbort("Group ID is zero-- invalid!\n");
       if (n>=max) CkAbort("Group ID is too large!\n");
#endif
      return tab[n];
    }
};

template <class dtype> 
class GroupIdxArray {
  enum {MAXBINSPE0=256,MAXBINSOTHER=16};
  GroupIdxArrayEntry<dtype> *tab;
  public:
     GroupIdxArray() {}
     void init(void) {
       tab = new GroupIdxArrayEntry<dtype>[CkNumPes()];
       tab[0].init(MAXBINSPE0); 
       if (MAXBINSOTHER)
         for(int i=1;i<CkNumPes();i++) {
           tab[i].init(MAXBINSOTHER); 
         }
     }
    dtype& find(CkGroupID n) {return tab[n.pe].find(n.idx);}
};

typedef GroupIdxArray<TableEntry> GroupTable;

extern unsigned int    _printCS;
extern unsigned int    _printSS;

extern int     _infoIdx;
extern unsigned int    _numInitMsgs;
extern unsigned int    _numInitNodeMsgs;
extern int     _charmHandlerIdx;
extern int     _initHandlerIdx;
extern int     _bocHandlerIdx;
extern int     _nodeBocHandlerIdx;
extern int     _qdHandlerIdx;
extern CmiNodeLock _nodeLock;

CkpvExtern(void*,       _currentChare);
CkpvExtern(int,       _currentChareType);
CkpvExtern(CkGroupID,         _currentGroup);
CkpvExtern(CkGroupID,         _currentNodeGroup);
CkpvExtern(GroupTable, _groupTable);
CkpvExtern(unsigned int, _numGroups);
extern unsigned int _numNodeGroups;

static inline IrrGroup *_localBranch(CkGroupID gID)
{
  return CkpvAccess(_groupTable).find(gID).getObj();
}

extern GroupTable*    _nodeGroupTable;

extern void _initCharm(int argc, char **argv);

#endif


