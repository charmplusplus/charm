
/* DEFS: group HeapCentLB: CentralLB{
void HeapCentLB(void);
};
 */
#ifndef CK_TEMPLATES_ONLY
int CProxy_HeapCentLB::__idx=0;
#endif
/* DEFS: void HeapCentLB(void);
 */
#ifndef CK_TEMPLATES_ONLY
int CProxy_HeapCentLB::__idx_HeapCentLB_void=0;
CkGroupID CProxy_HeapCentLB::ckNew(void)
{
  void *msg = CkAllocSysMsg();
  return CkCreateGroup(__idx, __idx_HeapCentLB_void, msg, 0, 0);
}
CkGroupID CProxy_HeapCentLB::ckNewSync(void)
{
  void *msg = CkAllocSysMsg();
  return CkCreateGroupSync(__idx, __idx_HeapCentLB_void, msg);
}
 CProxy_HeapCentLB::CProxy_HeapCentLB(int retEP, CkChareID *cid)
{
  void *msg = CkAllocSysMsg();
  _ck_gid = CkCreateGroup(__idx, __idx_HeapCentLB_void, msg, retEP, cid);
  _setChare(0);
}
 CProxy_HeapCentLB::CProxy_HeapCentLB(void)
{
  void *msg = CkAllocSysMsg();
  _ck_gid = CkCreateGroup(__idx, __idx_HeapCentLB_void, msg, 0, 0);
  _setChare(0);
}
 void CProxy_HeapCentLB::_call_HeapCentLB_void(void* msg, HeapCentLB* obj)
{
  new (obj) HeapCentLB();
  CkFreeSysMsg(msg);
}
#endif

#ifndef CK_TEMPLATES_ONLY
void _registerHeapCentLB(void)
{
  static int _done = 0; if (_done) return; _done = 1;
  _registerCentralLB();

/* REG: group HeapCentLB: CentralLB{
void HeapCentLB(void);
};
 */
  CProxy_HeapCentLB::__register("HeapCentLB", sizeof(HeapCentLB));

}
#endif
