#ifndef _INIT_H
#define _INIT_H

#include <charm.h> // For CkNumPes
#include <new.h>   // for in-place new operator
#include "ckhashtable.h"

typedef CkQ<void *> PtrQ;
class envelope;
typedef CkVec<CkZeroPtr<envelope> > PtrVec;

class IrrGroup;
class TableEntry {
    IrrGroup *obj;
    PtrQ *pending; //Buffers msgs recv'd before group is created
  public:
    TableEntry(void) {init();}
    void init(void) { obj=0; pending=0; }
    inline IrrGroup* getObj(void) { return obj; }
    void setObj(void *_obj) { obj=(IrrGroup *)_obj; }
    PtrQ* getPending(void) { return pending; }
    void enqMsg(void *msg) {
      if (pending==0)
        pending=new PtrQ();
      pending->enq(msg);
    }
};

template <class dtype>
class GroupIdxArray {
  enum {MAXBINSPE0=256};

  dtype *tab;                           // direct entry table for processor 0
  CkHashtable_c hashTab;
  int max;
  
  //This non-inline version of "find", below, allows the (much simpler)
  // common case to be inlined.
  dtype& nonInlineFind(CkGroupID n) {
      dtype *ret;
#ifndef CMK_OPTIMIZE
      if (n.idx==0) CkAbort("Group ID is zero-- invalid!\n");
      else if (n.idx>=max) { CkAbort("Group ID is too large!\n");}
      else 
#endif
      /*n.idx < 0*/
      { /*Groups created on processors other than 0 go into a hashtable:*/
        if(hashTab == NULL)
          hashTab = CkCreateHashtable_int(sizeof(dtype),17);

        ret = (dtype *)CkHashtableGet(hashTab,&(n.idx));

        if(ret == NULL)               // insert data into the table
        {
          ret = (dtype *)CkHashtablePut(hashTab,&(n.idx));
          new (ret) dtype; //Call dtype's constructor (ICK!)
        }
      }
      return *ret;
   }

  public:
     GroupIdxArray() {tab=NULL;max=0;hashTab=NULL;}
     ~GroupIdxArray() {delete[] tab; if (hashTab!=NULL) CkDeleteHashtable(hashTab);}
     void init(void) {
      max = MAXBINSPE0;
      tab = new dtype[max];
      for(int i=0;i<max;i++)
       tab[i].init();
      hashTab=NULL;
     }

     inline dtype& find(CkGroupID n) {

// TODO: make the table extensible. i.e. if (unsigned)n.idx<max then return tab[n.idx]
// else if (n.idx<0)    then hashtable things
// else extend the table

      if(n.idx>0 && n.idx<MAXBINSPE0)
        return tab[n.idx];
      else
        return nonInlineFind(n);
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

CsvExtern(GroupTable*,  _nodeGroupTable);
CsvExtern(CmiNodeLock, _nodeLock);
CsvExtern(unsigned int, _numNodeGroups);

CkpvExtern(CkGroupID,_currentGroup);

CkpvExtern(CkGroupID, _currentGroupRednMgr);


CkpvExtern(GroupTable*, _groupTable);

CkpvExtern(unsigned int, _numGroups);
CpvExtern(char **,Ck_argv);

static inline IrrGroup *_localBranch(CkGroupID gID)
{
  return CkpvAccess(_groupTable)->find(gID).getObj();
}

extern void _initCharm(int argc, char **argv);

typedef  void  (*CkExitFn) (void);

extern CkQ<CkExitFn> CkExitFnVec;
extern void registerExitFn(CkExitFn);

#endif


