#ifndef _DECL_idxl_H_
#define _DECL_idxl_H_
#include "charm++.h"
/* DECLS: message IDXL_DataMsg;
 */
class IDXL_DataMsg;
class CMessage_IDXL_DataMsg:public CkMessage{
  public:
    static int __idx;
    void* operator new(size_t,void*p) { return p; }
    void* operator new(size_t,const int);
    void* operator new(size_t);
    void* operator new(size_t, int*, const int);
    void* operator new(size_t, int*);
#if CMK_MULTIPLE_DELETE
    void operator delete(void*p,void*){CkFreeMsg(p);}
    void operator delete(void*p,const int){CkFreeMsg(p);}
    void operator delete(void*p){ CkFreeMsg(p);}
    void operator delete(void*p,int*,const int){CkFreeMsg(p);}
    void operator delete(void*p,int*){CkFreeMsg(p);}
#endif
    void operator delete(void*p,size_t){CkFreeMsg(p);}
    static void* alloc(int,size_t,int*,int);
    CMessage_IDXL_DataMsg() {};
    static void *pack(IDXL_DataMsg *p);
    static IDXL_DataMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, 0, size);
    }
};


/* DECLS: array IDXL_Chunk: ArrayElement{
IDXL_Chunk(CkMigrateMessage* impl_msg);
void IDXL_Chunk(const CkArrayID &threadArrayID);
void idxl_recv(IDXL_DataMsg* impl_msg);
};
 */
 class IDXL_Chunk;
/* --------------- index object ------------------ */
class CkIndex_IDXL_Chunk{
  public:
    static int __idx;
    static void __register(const char *s, size_t size);
/* DECLS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */
    static int __idx_IDXL_Chunk_CkMigrateMessage;
    static int ckNew(CkMigrateMessage* impl_msg) { return __idx_IDXL_Chunk_CkMigrateMessage; }
    static void _call_IDXL_Chunk_CkMigrateMessage(void* impl_msg,IDXL_Chunk* impl_obj);

/* DECLS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */
    static int __idx_IDXL_Chunk_marshall1;
    static int ckNew(const CkArrayID &threadArrayID) { return __idx_IDXL_Chunk_marshall1; }
    static void _call_IDXL_Chunk_marshall1(void* impl_msg,IDXL_Chunk* impl_obj);
    static int _callmarshall_IDXL_Chunk_marshall1(char* impl_buf,IDXL_Chunk* impl_obj);

/* DECLS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
    static int __idx_idxl_recv_IDXL_DataMsg;
    static int idxl_recv(IDXL_DataMsg* impl_msg) { return __idx_idxl_recv_IDXL_DataMsg; }
    static void _call_idxl_recv_IDXL_DataMsg(void* impl_msg,IDXL_Chunk* impl_obj);
};
/* --------------- element proxy ------------------ */
 class CProxyElement_IDXL_Chunk : public CProxyElement_ArrayElement{
  public:
    CProxyElement_IDXL_Chunk(void) {}
    CProxyElement_IDXL_Chunk(const ArrayElement *e) : CProxyElement_ArrayElement(e){  }
    void ckDelegate(CkGroupID to) {
      CProxyElement_ArrayElement::ckDelegate(to);
    }
    void ckUndelegate(void) {
      CProxyElement_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxyElement_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY_ELEMENT(CProxyElement_ArrayElement)
    IDXL_Chunk *ckLocal(void) const
      { return (IDXL_Chunk *)CProxyElement_ArrayElement::ckLocal(); }
    CProxyElement_IDXL_Chunk(const CkArrayID &aid,const CkArrayIndex1D &idx,CkGroupID dTo)
        :CProxyElement_ArrayElement(aid,idx,dTo) {}
    CProxyElement_IDXL_Chunk(const CkArrayID &aid,const CkArrayIndex1D &idx)
        :CProxyElement_ArrayElement(aid,idx) {}
/* DECLS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */

/* DECLS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */
    void insert(const CkArrayID &threadArrayID, int onPE=-1, const CkEntryOptions *impl_e_opts=NULL);
/* DECLS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
    void idxl_recv(IDXL_DataMsg* impl_msg) ;
};
PUPmarshall(CProxyElement_IDXL_Chunk);
/* ---------------- collective proxy -------------- */
 class CProxy_IDXL_Chunk : public CProxy_ArrayElement{
  public:
    CProxy_IDXL_Chunk(void) {}
    CProxy_IDXL_Chunk(const ArrayElement *e) : CProxy_ArrayElement(e){  }
    void ckDelegate(CkGroupID to) {
      CProxy_ArrayElement::ckDelegate(to);
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
    CProxyElement_IDXL_Chunk operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_IDXL_Chunk operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_IDXL_Chunk operator [] (int idx) const 
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), CkArrayIndex1D(idx), ckDelegatedIdx());}
    CProxyElement_IDXL_Chunk operator () (int idx) const 
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), CkArrayIndex1D(idx), ckDelegatedIdx());}
    CProxy_IDXL_Chunk(const CkArrayID &aid,CkGroupID dTo) 
        :CProxy_ArrayElement(aid,dTo) {}
    CProxy_IDXL_Chunk(const CkArrayID &aid) 
        :CProxy_ArrayElement(aid) {}
/* DECLS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */

/* DECLS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */
    static CkArrayID ckNew(const CkArrayID &threadArrayID, const CkArrayOptions &opts, const CkEntryOptions *impl_e_opts=NULL);

/* DECLS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
    void idxl_recv(IDXL_DataMsg* impl_msg) ;
};
PUPmarshall(CProxy_IDXL_Chunk);
/* ---------------- section proxy -------------- */
 class CProxySection_IDXL_Chunk : public CProxySection_ArrayElement{
  public:
    CProxySection_IDXL_Chunk(void) {}
    void ckDelegate(CkGroupID to) {
      CProxySection_ArrayElement::ckDelegate(to);
    }
    void ckUndelegate(void) {
      CProxySection_ArrayElement::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxySection_ArrayElement::pup(p);
    }
    CK_DISAMBIG_ARRAY_SECTION(CProxySection_ArrayElement)
//Generalized array indexing:
    CProxyElement_IDXL_Chunk operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_IDXL_Chunk operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), idx, ckDelegatedIdx());}
    CProxyElement_IDXL_Chunk operator [] (int idx) const 
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}
    CProxyElement_IDXL_Chunk operator () (int idx) const 
        {return CProxyElement_IDXL_Chunk(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], ckDelegatedIdx());}
    CProxySection_IDXL_Chunk(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems, CkGroupID dTo) 
        :CProxySection_ArrayElement(aid,elems,nElems,dTo) {}
    CProxySection_IDXL_Chunk(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) 
        :CProxySection_ArrayElement(aid,elems,nElems) {}
    CProxySection_IDXL_Chunk(const CkSectionID &sid)       :CProxySection_ArrayElement(sid) {}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) {
      return CkSectionID(aid, elems, nElems);
    } 
/* DECLS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */

/* DECLS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */

/* DECLS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
    void idxl_recv(IDXL_DataMsg* impl_msg) ;
};
PUPmarshall(CProxySection_IDXL_Chunk);
typedef CBaseT<ArrayElementT<CkIndex1D>,CProxy_IDXL_Chunk>  CBase_IDXL_Chunk;

extern void _registeridxl(void);
#endif
