#ifndef _DECL_HeapCentLB_H_
#define _DECL_HeapCentLB_H_
#include "charm++.h"
#include "CentralLB.decl.h"

/* DECLS: group HeapCentLB: CentralLB{
void HeapCentLB(void);
};
 */
class HeapCentLB;
class CProxy_HeapCentLB:  public virtual _CK_GID, public CProxy_CentralLB{
  public:
    static int __idx;
static void __register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  _REGISTER_BASE(__idx, CProxy_CentralLB::__idx);
/* REG: void HeapCentLB(void);
 */
  __idx_HeapCentLB_void = CkRegisterEp("HeapCentLB", (CkCallFnPtr)_call_HeapCentLB_void, 0, __idx);
}
    CProxy_HeapCentLB(CkGroupID _gid) :CProxy_CentralLB(_gid){ _ck_gid = _gid; _setChare(0); }
    CProxy_HeapCentLB(CkChareID __cid) :CProxy_CentralLB(__cid){ ckSetChareId(__cid); }
    CkChareID ckGetChareId(void) { return _ck_cid; }
    void ckSetChareId(CkChareID __cid){_CHECK_CID(__cid,__idx);_ck_cid=__cid;_setChare(1);}
    CkGroupID ckGetGroupId(void) { return _ck_gid; }
   void ckSetGroupId(CkGroupID _gid){_ck_gid=_gid;_setChare(0);}
    HeapCentLB* ckLocalBranch(void) {
      return (HeapCentLB *) CkLocalBranch(_ck_gid);
    }
    static HeapCentLB* ckLocalBranch(CkGroupID gID) {
      return (HeapCentLB *) CkLocalBranch(gID);
    }
/* DECLS: void HeapCentLB(void);
 */
    static int __idx_HeapCentLB_void;
    static CkGroupID ckNew(void);
    static CkGroupID ckNewSync(void);
    CProxy_HeapCentLB(int retEP, CkChareID *cid);
    CProxy_HeapCentLB(void);
    static int ckIdx_HeapCentLB(void) { return __idx_HeapCentLB_void; }
    static void  _call_HeapCentLB_void(void* msg, HeapCentLB* obj);
};

extern void _registerHeapCentLB(void);
#endif
