/**
 * @file collideClient.h
 * @brief charm declarations (Chare group) required for defining collision detection libraries client,
 * the library needs a cleaner fix
 * @author Ehsan Totoni
 * @version 
 * @date 2012-09-05
 */
#ifndef _CLIENT_DECL_H_
#define _CLIENT_DECL_H_

/* DECLS: group collideClient: IrrGroup;
 */
 class collideClient;
 class CkIndex_collideClient;
 class CProxy_collideClient;
 class CProxyElement_collideClient;
 class CProxySection_collideClient;
/* --------------- index object ------------------ */
class CkIndex_collideClient:public CProxyElement_IrrGroup{
  public:
    typedef collideClient local_t;
    typedef CkIndex_collideClient index_t;
    typedef CProxy_collideClient proxy_t;
    typedef CProxyElement_collideClient element_t;
    typedef CProxySection_collideClient section_t;

    static int __idx;
    static void __register(const char *s, size_t size);
};
/* --------------- element proxy ------------------ */
class CProxyElement_collideClient: public CProxyElement_IrrGroup{
  public:
    typedef collideClient local_t;
    typedef CkIndex_collideClient index_t;
    typedef CProxy_collideClient proxy_t;
    typedef CProxyElement_collideClient element_t;
    typedef CProxySection_collideClient section_t;

    CProxyElement_collideClient(void) {}
    CProxyElement_collideClient(const IrrGroup *g) : CProxyElement_IrrGroup(g){  }
    CProxyElement_collideClient(CkGroupID _gid,int _onPE,CK_DELCTOR_PARAM) : CProxyElement_IrrGroup(_gid,_onPE,CK_DELCTOR_ARGS){  }
    CProxyElement_collideClient(CkGroupID _gid,int _onPE) : CProxyElement_IrrGroup(_gid,_onPE){  }

    int ckIsDelegated(void) const
    { return CProxyElement_IrrGroup::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxyElement_IrrGroup::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxyElement_IrrGroup::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxyElement_IrrGroup::ckDelegatedIdx(); }
inline void ckCheck(void) const {CProxyElement_IrrGroup::ckCheck();}
CkChareID ckGetChareID(void) const
   {return CProxyElement_IrrGroup::ckGetChareID();}
CkGroupID ckGetGroupID(void) const
   {return CProxyElement_IrrGroup::ckGetGroupID();}
operator CkGroupID () const { return ckGetGroupID(); }

    inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxyElement_IrrGroup::setReductionClient(fn,param); }
    inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxyElement_IrrGroup::ckSetReductionClient(fn,param); }
    inline void ckSetReductionClient(CkCallback *cb) const
    { CProxyElement_IrrGroup::ckSetReductionClient(cb); }
int ckGetGroupPe(void) const
{return CProxyElement_IrrGroup::ckGetGroupPe();}

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxyElement_IrrGroup::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxyElement_IrrGroup::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxyElement_IrrGroup::pup(p); }
    void ckSetGroupID(CkGroupID g) {
      CProxyElement_IrrGroup::ckSetGroupID(g);
    }
    collideClient* ckLocalBranch(void) const {
      return ckLocalBranch(ckGetGroupID());
    }
    static collideClient* ckLocalBranch(CkGroupID gID) {
      return (collideClient*)CkLocalBranch(gID);
    }
};
PUPmarshall(CProxyElement_collideClient)
/* ---------------- collective proxy -------------- */
class CProxy_collideClient: public CProxy_IrrGroup{
  public:
    typedef collideClient local_t;
    typedef CkIndex_collideClient index_t;
    typedef CProxy_collideClient proxy_t;
    typedef CProxyElement_collideClient element_t;
    typedef CProxySection_collideClient section_t;

    CProxy_collideClient(void) {}
    CProxy_collideClient(const IrrGroup *g) : CProxy_IrrGroup(g){  }
    CProxy_collideClient(CkGroupID _gid,CK_DELCTOR_PARAM) : CProxy_IrrGroup(_gid,CK_DELCTOR_ARGS){  }
    CProxy_collideClient(CkGroupID _gid) : CProxy_IrrGroup(_gid){  }
    CProxyElement_collideClient operator[](int onPE) const
      {return CProxyElement_collideClient(ckGetGroupID(),onPE,CK_DELCTOR_CALL);}

    int ckIsDelegated(void) const
    { return CProxy_IrrGroup::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxy_IrrGroup::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxy_IrrGroup::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxy_IrrGroup::ckDelegatedIdx(); }
inline void ckCheck(void) const {CProxy_IrrGroup::ckCheck();}
CkChareID ckGetChareID(void) const
   {return CProxy_IrrGroup::ckGetChareID();}
CkGroupID ckGetGroupID(void) const
   {return CProxy_IrrGroup::ckGetGroupID();}
operator CkGroupID () const { return ckGetGroupID(); }

    inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxy_IrrGroup::setReductionClient(fn,param); }
    inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxy_IrrGroup::ckSetReductionClient(fn,param); }
    inline void ckSetReductionClient(CkCallback *cb) const
    { CProxy_IrrGroup::ckSetReductionClient(cb); }

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxy_IrrGroup::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxy_IrrGroup::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxy_IrrGroup::pup(p); }
    void ckSetGroupID(CkGroupID g) {
      CProxy_IrrGroup::ckSetGroupID(g);
    }
    collideClient* ckLocalBranch(void) const {
      return ckLocalBranch(ckGetGroupID());
    }
    static collideClient* ckLocalBranch(CkGroupID gID) {
      return (collideClient*)CkLocalBranch(gID);
    }
};
PUPmarshall(CProxy_collideClient)
/* ---------------- section proxy -------------- */
class CProxySection_collideClient: public CProxySection_IrrGroup{
  public:
    typedef collideClient local_t;
    typedef CkIndex_collideClient index_t;
    typedef CProxy_collideClient proxy_t;
    typedef CProxyElement_collideClient element_t;
    typedef CProxySection_collideClient section_t;

    CProxySection_collideClient(void) {}
    CProxySection_collideClient(const IrrGroup *g) : CProxySection_IrrGroup(g){  }
    CProxySection_collideClient(const CkGroupID &_gid,const int *_pelist,int _npes,CK_DELCTOR_PARAM) : CProxySection_IrrGroup(_gid,_pelist,_npes,CK_DELCTOR_ARGS){  }
    CProxySection_collideClient(const CkGroupID &_gid,const int *_pelist,int _npes) : CProxySection_IrrGroup(_gid,_pelist,_npes){  }
    CProxySection_collideClient(int n,const CkGroupID *_gid, int const * const *_pelist,const int *_npes) : CProxySection_IrrGroup(n,_gid,_pelist,_npes){  }
    CProxySection_collideClient(int n,const CkGroupID *_gid, int const * const *_pelist,const int *_npes,CK_DELCTOR_PARAM) : CProxySection_IrrGroup(n,_gid,_pelist,_npes,CK_DELCTOR_ARGS){  }

    int ckIsDelegated(void) const
    { return CProxySection_IrrGroup::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxySection_IrrGroup::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxySection_IrrGroup::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxySection_IrrGroup::ckDelegatedIdx(); }
inline void ckCheck(void) const {CProxySection_IrrGroup::ckCheck();}
CkChareID ckGetChareID(void) const
   {return CProxySection_IrrGroup::ckGetChareID();}
CkGroupID ckGetGroupID(void) const
   {return CProxySection_IrrGroup::ckGetGroupID();}
operator CkGroupID () const { return ckGetGroupID(); }

    inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxySection_IrrGroup::setReductionClient(fn,param); }
    inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxySection_IrrGroup::ckSetReductionClient(fn,param); }
    inline void ckSetReductionClient(CkCallback *cb) const
    { CProxySection_IrrGroup::ckSetReductionClient(cb); }
inline int ckGetNumSections() const
{ return CProxySection_IrrGroup::ckGetNumSections(); }
inline CkSectionInfo &ckGetSectionInfo()
{ return CProxySection_IrrGroup::ckGetSectionInfo(); }
inline CkSectionID *ckGetSectionIDs()
{ return CProxySection_IrrGroup::ckGetSectionIDs(); }
inline CkSectionID &ckGetSectionID()
{ return CProxySection_IrrGroup::ckGetSectionID(); }
inline CkSectionID &ckGetSectionID(int i)
{ return CProxySection_IrrGroup::ckGetSectionID(i); }
inline CkGroupID ckGetGroupIDn(int i) const
{ return CProxySection_IrrGroup::ckGetGroupIDn(i); }
inline int *ckGetElements() const
{ return CProxySection_IrrGroup::ckGetElements(); }
inline int *ckGetElements(int i) const
{ return CProxySection_IrrGroup::ckGetElements(i); }
inline int ckGetNumElements() const
{ return CProxySection_IrrGroup::ckGetNumElements(); } 
inline int ckGetNumElements(int i) const
{ return CProxySection_IrrGroup::ckGetNumElements(i); }

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxySection_IrrGroup::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxySection_IrrGroup::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxySection_IrrGroup::pup(p); }
    void ckSetGroupID(CkGroupID g) {
      CProxySection_IrrGroup::ckSetGroupID(g);
    }
    collideClient* ckLocalBranch(void) const {
      return ckLocalBranch(ckGetGroupID());
    }
    static collideClient* ckLocalBranch(CkGroupID gID) {
      return (collideClient*)CkLocalBranch(gID);
    }
};
PUPmarshall(CProxySection_collideClient)
typedef CBaseT1<Group, CProxy_collideClient> CBase_collideClient;

#endif
