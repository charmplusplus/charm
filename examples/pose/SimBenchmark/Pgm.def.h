
/* DEFS: readonly CkChareID mainhandle;
 */
extern CkChareID mainhandle;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_mainhandle(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|mainhandle;
}
#endif

/* DEFS: mainchare main: Chare{
main(CkArgMsg* impl_msg);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_main::__idx=0;
#endif
#ifndef CK_TEMPLATES_ONLY
/* DEFS: main(CkArgMsg* impl_msg);
 */
CkChareID CProxy_main::ckNew(CkArgMsg* impl_msg, int impl_onPE)
{
  CkChareID impl_ret;
  CkCreateChare(CkIndex_main::__idx, CkIndex_main::__idx_main_CkArgMsg, impl_msg, &impl_ret, impl_onPE);
  return impl_ret;
}
void CProxy_main::ckNew(CkArgMsg* impl_msg, CkChareID* pcid, int impl_onPE)
{
  CkCreateChare(CkIndex_main::__idx, CkIndex_main::__idx_main_CkArgMsg, impl_msg, pcid, impl_onPE);
}
  CProxy_main::CProxy_main(CkArgMsg* impl_msg, int impl_onPE)
{
  CkChareID impl_ret;
  CkCreateChare(CkIndex_main::__idx, CkIndex_main::__idx_main_CkArgMsg, impl_msg, &impl_ret, impl_onPE);
  ckSetChareID(impl_ret);
}
 int CkIndex_main::__idx_main_CkArgMsg=0;
void CkIndex_main::_call_main_CkArgMsg(void* impl_msg,main * impl_obj)
{
  new (impl_obj) main((CkArgMsg*)impl_msg);
}
#endif /*CK_TEMPLATES_ONLY*/
#ifndef CK_TEMPLATES_ONLY
void CkIndex_main::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  CkRegisterBase(__idx, CkIndex_Chare::__idx);
// REG: main(CkArgMsg* impl_msg);
  __idx_main_CkArgMsg = CkRegisterEp("main(CkArgMsg* impl_msg)",
     (CkCallFnPtr)_call_main_CkArgMsg, CMessage_CkArgMsg::__idx, __idx, 0);
  CkRegisterMainChare(__idx, __idx_main_CkArgMsg);
}
#endif

#ifndef CK_TEMPLATES_ONLY
void _registerPgm(void)
{
  static int _done = 0; if(_done) return; _done = 1;
      _registerWorker();

  CkRegisterReadonly("mainhandle","CkChareID",sizeof(mainhandle),(void *) &mainhandle,__xlater_roPup_mainhandle);

/* REG: mainchare main: Chare{
main(CkArgMsg* impl_msg);
};
*/
  CkIndex_main::__register("main", sizeof(main));

}
extern "C" void CkRegisterMainModule(void) {
  _registerPgm();
}
#endif
