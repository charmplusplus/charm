#ifndef _DECL_X10_test_H_
#define _DECL_X10_test_H_
#include "charm++.h"
/* DECLS: message asyncMsg;
 */
class asyncMsg;
class CMessage_asyncMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t, void*p) { return p; }
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p, void*){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p, int*, const int){CkFreeMsg(p);}
    void operator delete(void*p, int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p, size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t, int*, int);
    CMessage_asyncMsg() {};
    static void *pack(asyncMsg *p);
    static asyncMsg* unpack(void* p);
    void *operator new(size_t, const int);
#if CMK_MULTIPLE_DELETE
    void operator delete(void *p, const int){CkFreeMsg(p);}
#endif
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: readonly CProxy_Main mainProxy;
 */

/* DECLS: readonly int nPlaces;
 */

/* DECLS: readonly CProxy_Places placesProxy;
 */

/* DECLS: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
threaded void libThread(void);
};
 */
 class Main;
 class CkIndex_Main;
 class CProxy_Main;
/* --------------- index object ------------------ */
class CkIndex_Main:public CProxy_Chare{
  public:
    typedef Main local_t;
    typedef CkIndex_Main index_t;
    typedef CProxy_Main proxy_t;
    typedef CProxy_Main element_t;

    static int __idx;
    static void __register(const char *s, size_t size);
/* DECLS: Main(CkArgMsg* impl_msg);
 */
    static int __idx_Main_CkArgMsg;
    static int ckNew(CkArgMsg* impl_msg) { return __idx_Main_CkArgMsg; }
    static void _call_Main_CkArgMsg(void* impl_msg,Main* impl_obj);

/* DECLS: threaded void libThread(void);
 */
    static int __idx_libThread_void;
    static int libThread(void) { return __idx_libThread_void; }
    static void _call_libThread_void(void* impl_msg,Main* impl_obj);
    static void _callthr_libThread_void(CkThrCallArg *);
};
/* --------------- element proxy ------------------ */
class CProxy_Main:public CProxy_Chare{
  public:
    typedef Main local_t;
    typedef CkIndex_Main index_t;
    typedef CProxy_Main proxy_t;
    typedef CProxy_Main element_t;

    CProxy_Main(void) {};
    CProxy_Main(CkChareID __cid) : CProxy_Chare(__cid){  }
    CProxy_Main(const Chare *c) : CProxy_Chare(c){  }
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
    Main *ckLocal(void) const
     { return (Main *)CkLocalChare(&ckGetChareID()); }
/* DECLS: Main(CkArgMsg* impl_msg);
 */
    static CkChareID ckNew(CkArgMsg* impl_msg, int onPE=CK_PE_ANY);
    static void ckNew(CkArgMsg* impl_msg, CkChareID* pcid, int onPE=CK_PE_ANY);
    CProxy_Main(CkArgMsg* impl_msg, int onPE=CK_PE_ANY);

/* DECLS: threaded void libThread(void);
 */
    void libThread(void);
};
PUPmarshall(CProxy_Main);
typedef CBaseT<Chare,CProxy_Main>  CBase_Main;

/* DECLS: array Places: ArrayElement{
Places(CkMigrateMessage* impl_msg);
void Places(void);
threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
};
 */
 class Places;
 class CkIndex_Places;
 class CProxy_Places;
 class CProxyElement_Places;
 class CProxySection_Places;
/* --------------- index object ------------------ */
class CkIndex_Places:public CProxyElement_ArrayElement{
  public:
    typedef Places local_t;
    typedef CkIndex_Places index_t;
    typedef CProxy_Places proxy_t;
    typedef CProxyElement_Places element_t;
    typedef CProxySection_Places section_t;

    static int __idx;
    static void __register(const char *s, size_t size);
/* DECLS: Places(CkMigrateMessage* impl_msg);
 */
    static int __idx_Places_CkMigrateMessage;
    static int ckNew(CkMigrateMessage* impl_msg) { return __idx_Places_CkMigrateMessage; }
    static void _call_Places_CkMigrateMessage(void* impl_msg,Places* impl_obj);

/* DECLS: void Places(void);
 */
    static int __idx_Places_void;
    static int ckNew(void) { return __idx_Places_void; }
    static void _call_Places_void(void* impl_msg,Places* impl_obj);

/* DECLS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
    static int __idx_startAsync_marshall2;
    static int startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src) { return __idx_startAsync_marshall2; }
    static void _call_startAsync_marshall2(void* impl_msg,Places* impl_obj);
    static void _callthr_startAsync_marshall2(CkThrCallArg *);
    static void _marshallmessagepup_startAsync_marshall2(PUP::er &p,void *msg);
};
/* --------------- element proxy ------------------ */
 class CProxyElement_Places : public CProxyElement_ArrayElement{
  public:
    typedef Places local_t;
    typedef CkIndex_Places index_t;
    typedef CProxy_Places proxy_t;
    typedef CProxyElement_Places element_t;
    typedef CProxySection_Places section_t;

    CProxyElement_Places(void) {}
    CProxyElement_Places(const ArrayElement *e) : CProxyElement_ArrayElement(e){  }
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxyElement_ArrayElement::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxyElement_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxyElement_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY_ELEMENT(CProxyElement_ArrayElement)
    Places *ckLocal(void) const
      { return (Places *)CProxyElement_ArrayElement::ckLocal(); }
    CProxyElement_Places(const CkArrayID &aid,const CkArrayIndex1D &idx,CK_DELCTOR_PARAM)
        :CProxyElement_ArrayElement(aid,idx,CK_DELCTOR_ARGS) {}
    CProxyElement_Places(const CkArrayID &aid,const CkArrayIndex1D &idx)
        :CProxyElement_ArrayElement(aid,idx) {}
/* DECLS: Places(CkMigrateMessage* impl_msg);
 */

/* DECLS: void Places(void);
 */
    void insert(int onPE=-1);
/* DECLS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
    void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src, const CkEntryOptions *impl_e_opts=NULL) ;
};
PUPmarshall(CProxyElement_Places);
/* ---------------- collective proxy -------------- */
 class CProxy_Places : public CProxy_ArrayElement{
  public:
    typedef Places local_t;
    typedef CkIndex_Places index_t;
    typedef CProxy_Places proxy_t;
    typedef CProxyElement_Places element_t;
    typedef CProxySection_Places section_t;

    CProxy_Places(void) {}
    CProxy_Places(const ArrayElement *e) : CProxy_ArrayElement(e){  }
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxy_ArrayElement::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxy_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxy_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY(CProxy_ArrayElement)
    static CkArrayID ckNew(void) {return ckCreateEmptyArray();}
//Generalized array indexing:
    CProxyElement_Places operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_Places(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Places operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_Places(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Places operator [] (int idx) const 
        {return CProxyElement_Places(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxyElement_Places operator () (int idx) const 
        {return CProxyElement_Places(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxy_Places(const CkArrayID &aid,CK_DELCTOR_PARAM) 
        :CProxy_ArrayElement(aid,CK_DELCTOR_ARGS) {}
    CProxy_Places(const CkArrayID &aid) 
        :CProxy_ArrayElement(aid) {}
/* DECLS: Places(CkMigrateMessage* impl_msg);
 */

/* DECLS: void Places(void);
 */
    static CkArrayID ckNew(const CkArrayOptions &opts);

/* DECLS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
    void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src, const CkEntryOptions *impl_e_opts=NULL) ;
};
PUPmarshall(CProxy_Places);
/* ---------------- section proxy -------------- */
 class CProxySection_Places : public CProxySection_ArrayElement{
  public:
    typedef Places local_t;
    typedef CkIndex_Places index_t;
    typedef CProxy_Places proxy_t;
    typedef CProxyElement_Places element_t;
    typedef CProxySection_Places section_t;

    CProxySection_Places(void) {}
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxySection_ArrayElement::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxySection_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxySection_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY_SECTION(CProxySection_ArrayElement)
//Generalized array indexing:
    CProxyElement_Places operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_Places(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Places operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_Places(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Places operator [] (int idx) const 
        {return CProxyElement_Places(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    CProxyElement_Places operator () (int idx) const 
        {return CProxyElement_Places(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex1D *elems, int nElems) {
      return CkSectionID(aid, elems, nElems);
    } 
    static CkSectionID ckNew(const CkArrayID &aid, int l, int u, int s) {
      CkVec<CkArrayIndex1D> al;
      for (int i=l; i<=u; i+=s) al.push_back(CkArrayIndex1D(i));
      return CkSectionID(aid, al.getVec(), al.size());
    } 
    CProxySection_Places(const CkArrayID &aid, CkArrayIndex *elems, int nElems, CK_DELCTOR_PARAM) 
        :CProxySection_ArrayElement(aid,elems,nElems,CK_DELCTOR_ARGS) {}
    CProxySection_Places(const CkArrayID &aid, CkArrayIndex *elems, int nElems) 
        :CProxySection_ArrayElement(aid,elems,nElems) {}
    CProxySection_Places(const CkSectionID &sid)       :CProxySection_ArrayElement(sid) {}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex *elems, int nElems) {
      return CkSectionID(aid, elems, nElems);
    } 
/* DECLS: Places(CkMigrateMessage* impl_msg);
 */

/* DECLS: void Places(void);
 */

/* DECLS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
    void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src, const CkEntryOptions *impl_e_opts=NULL) ;
};
PUPmarshall(CProxySection_Places);
typedef CBaseT<ArrayElementT<CkIndex1D>,CProxy_Places>  CBase_Places;

extern void _registerX10_test(void);
extern "C" void CkRegisterMainModule(void);
#endif
