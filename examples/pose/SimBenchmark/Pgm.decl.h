#ifndef _DECL_Pgm_H_
#define _DECL_Pgm_H_
#include "charm++.h"
#include "Worker.decl.h"

/* DECLS: readonly CkChareID mainhandle;
 */

/* DECLS: mainchare main: Chare{
main(CkArgMsg* impl_msg);
};
 */
 class main;
 class CkIndex_main;
 class CProxy_main;
/* --------------- index object ------------------ */
class CkIndex_main:public CProxy_Chare{
  public:
    typedef main local_t;
    typedef CkIndex_main index_t;
    typedef CProxy_main proxy_t;
    typedef CProxy_main element_t;

    static int __idx;
    static void __register(const char *s, size_t size);
/* DECLS: main(CkArgMsg* impl_msg);
 */
    static int __idx_main_CkArgMsg;
    static int ckNew(CkArgMsg* impl_msg) { return __idx_main_CkArgMsg; }
    static void _call_main_CkArgMsg(void* impl_msg,main* impl_obj);
};
/* --------------- element proxy ------------------ */
class CProxy_main:public CProxy_Chare{
  public:
    typedef main local_t;
    typedef CkIndex_main index_t;
    typedef CProxy_main proxy_t;
    typedef CProxy_main element_t;

    CProxy_main(void) {};
    CProxy_main(CkChareID __cid) : CProxy_Chare(__cid){  }
    CProxy_main(const Chare *c) : CProxy_Chare(c){  }
    CK_DISAMBIG_CHARE(CProxy_Chare)
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxy_Chare::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxy_Chare::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxy_Chare::pup(p);
    }
    void ckSetChareID(const CkChareID &c) {
      CProxy_Chare::ckSetChareID(c);
    }
    main *ckLocal(void) const
     { return (main *)CkLocalChare(&ckGetChareID()); }
/* DECLS: main(CkArgMsg* impl_msg);
 */
    static CkChareID ckNew(CkArgMsg* impl_msg, int onPE=CK_PE_ANY);
    static void ckNew(CkArgMsg* impl_msg, CkChareID* pcid, int onPE=CK_PE_ANY);
    CProxy_main(CkArgMsg* impl_msg, int onPE=CK_PE_ANY);
};
PUPmarshall(CProxy_main);
typedef CBaseT<Chare,CProxy_main>  CBase_main;

extern void _registerPgm(void);
extern "C" void CkRegisterMainModule(void);
#endif
