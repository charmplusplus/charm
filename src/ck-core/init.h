#ifndef _INIT_H
#define _INIT_H

#include <charm.h> // For CkNumPes
#include <new.h>   // for in-place new operator
#include "ckhashtable.h"

typedef CkQ<void *> PtrQ;
typedef CkVec<void *> PtrVec;

class IrrGroup;
class TableEntry {
    IrrGroup *obj;
    PtrQ *pending; //Buffers msgs recv'd before group is created
  public:
    TableEntry(void) {init();}
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

// new GroupIdxArray
template <class dtype>
class GroupIdxArray {
#if CMK_BLUEGENE_CHARM
  enum {MAXBINSPE0=256,MAXBINSOTHER=0};			// MAXBINSOTHER is not used now
#else
  enum {MAXBINSPE0=256,MAXBINSOTHER=16};
#endif

  dtype *tab;                           // direct entry table for processor 0
  CkHashtable_c hashTab;
  int max;

  public:
     GroupIdxArray() {tab=NULL;max=0;hashTab=NULL;}
     void init(void) {
      max = MAXBINSPE0;
      tab = new dtype[max];
      for(int i=0;i<max;i++)
       tab[i].init();
      hashTab=NULL;
     }

    dtype& find(CkGroupID n) {
#ifndef CMK_OPTIMIZE
       if (n.idx==0) CkAbort("Group ID is zero-- invalid!\n");
       if (n.idx>=max) { CkAbort("Group ID is too large!\n");}
#endif

// TODO: make the table extensible. i.e. if (unsigned)n.idx<max then return tab[n.idx]
// else if (n.idx<0)    then hashtable things
// else extend the table

      if(n.idx>0)
        return tab[n.idx];
      else
      {
        if(hashTab == NULL)
                hashTab = CkCreateHashtable_int(sizeof(dtype),17);

        dtype *ret = (dtype *)CkHashtableGet(hashTab,&(n.idx));

        if(ret == NULL)                                 // insert data into the table
        {
                ret = (dtype *)CkHashtablePut(hashTab,&(n.idx));
                new (ret) dtype; //Call dtype's constructor (ICK!)
        }
        return *ret;
      }
    }
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

CkpvExtern(CkGroupID,_currentGroup);
CkpvExtern(GroupTable, _groupTable);
CkpvExtern(unsigned int, _numGroups);
extern unsigned int _numNodeGroups;

static inline IrrGroup *_localBranch(CkGroupID gID)
{
  return CkpvAccess(_groupTable).find(gID).getObj();
}

extern GroupTable*  _nodeGroupTable;

extern void _initCharm(int argc, char **argv);

#endif


