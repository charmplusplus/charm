#ifndef _DECL_Worker_H_
#define _DECL_Worker_H_
#include "charm++.h"
#include "sim.decl.h"

#include "pose.decl.h"

/* DECLS: message SmallWorkMsg;
 */
class SmallWorkMsg;
class CMessage_SmallWorkMsg:public CkMessage{
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
    CMessage_SmallWorkMsg() {};
    static void *pack(SmallWorkMsg *p);
    static SmallWorkMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message WorkerData;
 */
class WorkerData;
class CMessage_WorkerData:public CkMessage{
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
    CMessage_WorkerData() {};
    static void *pack(WorkerData *p);
    static WorkerData* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message LargeWorkMsg;
 */
class LargeWorkMsg;
class CMessage_LargeWorkMsg:public CkMessage{
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
    CMessage_LargeWorkMsg() {};
    static void *pack(LargeWorkMsg *p);
    static LargeWorkMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: message MediumWorkMsg;
 */
class MediumWorkMsg;
class CMessage_MediumWorkMsg:public CkMessage{
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
    CMessage_MediumWorkMsg() {};
    static void *pack(MediumWorkMsg *p);
    static MediumWorkMsg* unpack(void* p);
    static void __register(const char *s, size_t size, CkPackFnPtr pack, CkUnpackFnPtr unpack) {
      __idx = CkRegisterMsg(s, pack, unpack, size);
    }
};

/* DECLS: array worker: sim{
worker(CkMigrateMessage* impl_msg);
worker(WorkerData* impl_msg);
void workSmall(SmallWorkMsg* impl_msg);
void workMedium(MediumWorkMsg* impl_msg);
void workLarge(LargeWorkMsg* impl_msg);
};
 */
 class worker;
 class CkIndex_worker;
 class CProxy_worker;
 class CProxyElement_worker;
 class CProxySection_worker;
/* --------------- index object ------------------ */
class CkIndex_worker:public CProxyElement_sim{
  public:
    typedef worker local_t;
    typedef CkIndex_worker index_t;
    typedef CProxy_worker proxy_t;
    typedef CProxyElement_worker element_t;
    typedef CProxySection_worker section_t;

    static int __idx;
    static void __register(const char *s, size_t size);
/* DECLS: worker(CkMigrateMessage* impl_msg);
 */
    static int __idx_worker_CkMigrateMessage;
    static int ckNew(CkMigrateMessage* impl_msg) { return __idx_worker_CkMigrateMessage; }
    static void _call_worker_CkMigrateMessage(void* impl_msg,worker* impl_obj);

/* DECLS: worker(WorkerData* impl_msg);
 */
    static int __idx_worker_WorkerData;
    static int ckNew(WorkerData* impl_msg) { return __idx_worker_WorkerData; }
    static void _call_worker_WorkerData(void* impl_msg,worker* impl_obj);

/* DECLS: void workSmall(SmallWorkMsg* impl_msg);
 */
    static int __idx_workSmall_SmallWorkMsg;
    static int workSmall(SmallWorkMsg* impl_msg) { return __idx_workSmall_SmallWorkMsg; }
    static void _call_workSmall_SmallWorkMsg(void* impl_msg,worker* impl_obj);

/* DECLS: void workMedium(MediumWorkMsg* impl_msg);
 */
    static int __idx_workMedium_MediumWorkMsg;
    static int workMedium(MediumWorkMsg* impl_msg) { return __idx_workMedium_MediumWorkMsg; }
    static void _call_workMedium_MediumWorkMsg(void* impl_msg,worker* impl_obj);

/* DECLS: void workLarge(LargeWorkMsg* impl_msg);
 */
    static int __idx_workLarge_LargeWorkMsg;
    static int workLarge(LargeWorkMsg* impl_msg) { return __idx_workLarge_LargeWorkMsg; }
    static void _call_workLarge_LargeWorkMsg(void* impl_msg,worker* impl_obj);
};
/* --------------- element proxy ------------------ */
 class CProxyElement_worker : public CProxyElement_sim{
  public:
    typedef worker local_t;
    typedef CkIndex_worker index_t;
    typedef CProxy_worker proxy_t;
    typedef CProxyElement_worker element_t;
    typedef CProxySection_worker section_t;

    CProxyElement_worker(void) {}
    CProxyElement_worker(const ArrayElement *e) : CProxyElement_sim(e){  }
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxyElement_sim::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxyElement_sim::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxyElement_sim::pup(p);
    }
    CK_DISAMBIG_ARRAY_ELEMENT(CProxyElement_sim)
    worker *ckLocal(void) const
      { return (worker *)CProxyElement_sim::ckLocal(); }
    CProxyElement_worker(const CkArrayID &aid,const CkArrayIndex1D &idx,CK_DELCTOR_PARAM)
        :CProxyElement_sim(aid,idx,CK_DELCTOR_ARGS) {}
    CProxyElement_worker(const CkArrayID &aid,const CkArrayIndex1D &idx)
        :CProxyElement_sim(aid,idx) {}
/* DECLS: worker(CkMigrateMessage* impl_msg);
 */

/* DECLS: worker(WorkerData* impl_msg);
 */
    void insert(WorkerData* impl_msg, int onPE=-1);
/* DECLS: void workSmall(SmallWorkMsg* impl_msg);
 */
    void workSmall(SmallWorkMsg* impl_msg) ;

/* DECLS: void workMedium(MediumWorkMsg* impl_msg);
 */
    void workMedium(MediumWorkMsg* impl_msg) ;

/* DECLS: void workLarge(LargeWorkMsg* impl_msg);
 */
    void workLarge(LargeWorkMsg* impl_msg) ;
};
PUPmarshall(CProxyElement_worker);
/* ---------------- collective proxy -------------- */
 class CProxy_worker : public CProxy_sim{
  public:
    typedef worker local_t;
    typedef CkIndex_worker index_t;
    typedef CProxy_worker proxy_t;
    typedef CProxyElement_worker element_t;
    typedef CProxySection_worker section_t;

    CProxy_worker(void) {}
    CProxy_worker(const ArrayElement *e) : CProxy_sim(e){  }
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxy_sim::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxy_sim::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxy_sim::pup(p);
    }
    CK_DISAMBIG_ARRAY(CProxy_sim)
    static CkArrayID ckNew(void) {return ckCreateEmptyArray();}
//Generalized array indexing:
    CProxyElement_worker operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_worker(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_worker operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_worker(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_worker operator [] (int idx) const 
        {return CProxyElement_worker(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxyElement_worker operator () (int idx) const 
        {return CProxyElement_worker(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxy_worker(const CkArrayID &aid,CK_DELCTOR_PARAM) 
        :CProxy_sim(aid,CK_DELCTOR_ARGS) {}
    CProxy_worker(const CkArrayID &aid) 
        :CProxy_sim(aid) {}
/* DECLS: worker(CkMigrateMessage* impl_msg);
 */

/* DECLS: worker(WorkerData* impl_msg);
 */
    static CkArrayID ckNew(WorkerData* impl_msg, const CkArrayOptions &opts);

/* DECLS: void workSmall(SmallWorkMsg* impl_msg);
 */
    void workSmall(SmallWorkMsg* impl_msg) ;

/* DECLS: void workMedium(MediumWorkMsg* impl_msg);
 */
    void workMedium(MediumWorkMsg* impl_msg) ;

/* DECLS: void workLarge(LargeWorkMsg* impl_msg);
 */
    void workLarge(LargeWorkMsg* impl_msg) ;
};
PUPmarshall(CProxy_worker);
/* ---------------- section proxy -------------- */
 class CProxySection_worker : public CProxySection_sim{
  public:
    typedef worker local_t;
    typedef CkIndex_worker index_t;
    typedef CProxy_worker proxy_t;
    typedef CProxyElement_worker element_t;
    typedef CProxySection_worker section_t;

    CProxySection_worker(void) {}
    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL) {
      CProxySection_sim::ckDelegate(dTo,dPtr);
    }
    void ckUndelegate(void) {
      CProxySection_sim::ckUndelegate();
    }
    void pup(PUP::er &p) {
      CProxySection_sim::pup(p);
    }
    CK_DISAMBIG_ARRAY_SECTION(CProxySection_sim)
//Generalized array indexing:
    CProxyElement_worker operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_worker(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_worker operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_worker(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_worker operator [] (int idx) const 
        {return CProxyElement_worker(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    CProxyElement_worker operator () (int idx) const 
        {return CProxyElement_worker(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    CProxySection_worker(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems, CK_DELCTOR_PARAM) 
        :CProxySection_sim(aid,elems,nElems,CK_DELCTOR_ARGS) {}
    CProxySection_worker(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) 
        :CProxySection_sim(aid,elems,nElems) {}
    CProxySection_worker(const CkSectionID &sid)       :CProxySection_sim(sid) {}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndexMax *elems, int nElems) {
      return CkSectionID(aid, elems, nElems);
    } 
/* DECLS: worker(CkMigrateMessage* impl_msg);
 */

/* DECLS: worker(WorkerData* impl_msg);
 */

/* DECLS: void workSmall(SmallWorkMsg* impl_msg);
 */
    void workSmall(SmallWorkMsg* impl_msg) ;

/* DECLS: void workMedium(MediumWorkMsg* impl_msg);
 */
    void workMedium(MediumWorkMsg* impl_msg) ;

/* DECLS: void workLarge(LargeWorkMsg* impl_msg);
 */
    void workLarge(LargeWorkMsg* impl_msg) ;
};
PUPmarshall(CProxySection_worker);
typedef CBaseT<sim,CProxy_worker>  CBase_worker;

extern void _registerWorker(void);
#endif
