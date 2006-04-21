/* DEFS: message asyncMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_asyncMsg::operator new(size_t s){
  return asyncMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_asyncMsg::operator new(size_t s, int* sz){
  return asyncMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_asyncMsg::operator new(size_t s, int* sz,const int pb){
  return asyncMsg::alloc(__idx, s, sz, pb);
}
void *CMessage_asyncMsg::operator new(size_t s, const int p) {
  return asyncMsg::alloc(__idx, s, 0, p);
}
void* CMessage_asyncMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  asyncMsg *newmsg = (asyncMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_asyncMsg::pack(asyncMsg *msg) {
  return (void *) msg;
}
asyncMsg* CMessage_asyncMsg::unpack(void* buf) {
  asyncMsg *msg = (asyncMsg *) buf;
  return msg;
}
int CMessage_asyncMsg::__idx=0;
#endif

/* DEFS: readonly CProxy_Main mainProxy;
 */
extern CProxy_Main mainProxy;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_mainProxy(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|mainProxy;
}
#endif

/* DEFS: readonly int nPlaces;
 */
extern int nPlaces;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_nPlaces(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|nPlaces;
}
#endif

/* DEFS: readonly CProxy_Places placesProxy;
 */
extern CProxy_Places placesProxy;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_placesProxy(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|placesProxy;
}
#endif

/* DEFS: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
threaded void libThread(void);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_Main::__idx=0;
#endif
#ifndef CK_TEMPLATES_ONLY
/* DEFS: Main(CkArgMsg* impl_msg);
 */
CkChareID CProxy_Main::ckNew(CkArgMsg* impl_msg, int impl_onPE)
{
  CkChareID impl_ret;
  CkCreateChare(CkIndex_Main::__idx, CkIndex_Main::__idx_Main_CkArgMsg, impl_msg, &impl_ret, impl_onPE);
  return impl_ret;
}
void CProxy_Main::ckNew(CkArgMsg* impl_msg, CkChareID* pcid, int impl_onPE)
{
  CkCreateChare(CkIndex_Main::__idx, CkIndex_Main::__idx_Main_CkArgMsg, impl_msg, pcid, impl_onPE);
}
  CProxy_Main::CProxy_Main(CkArgMsg* impl_msg, int impl_onPE)
{
  CkChareID impl_ret;
  CkCreateChare(CkIndex_Main::__idx, CkIndex_Main::__idx_Main_CkArgMsg, impl_msg, &impl_ret, impl_onPE);
  ckSetChareID(impl_ret);
}
 int CkIndex_Main::__idx_Main_CkArgMsg=0;
void CkIndex_Main::_call_Main_CkArgMsg(void* impl_msg,Main * impl_obj)
{
  new (impl_obj) Main((CkArgMsg*)impl_msg);
}

/* DEFS: threaded void libThread(void);
 */
void CProxy_Main::libThread(void)
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  if (ckIsDelegated()) {
    int destPE=CkChareMsgPrep(CkIndex_Main::__idx_libThread_void, impl_msg, &ckGetChareID());
    if (destPE!=-1) ckDelegatedTo()->ChareSend(ckDelegatedPtr(),CkIndex_Main::__idx_libThread_void, impl_msg, &ckGetChareID(),destPE);
  }
  else CkSendMsg(CkIndex_Main::__idx_libThread_void, impl_msg, &ckGetChareID(),0);
}
 int CkIndex_Main::__idx_libThread_void=0;
void CkIndex_Main::_call_libThread_void(void* impl_msg,Main * impl_obj)
{
  CthThread tid = CthCreate((CthVoidFn)_callthr_libThread_void, new CkThrCallArg(impl_msg,impl_obj), 0);
  ((Chare *)impl_obj)->CkAddThreadListeners(tid,impl_msg);
  CthAwaken(tid);
}
void CkIndex_Main::_callthr_libThread_void(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  Main *impl_obj = (Main *) impl_arg->obj;
  delete impl_arg;
  CkFreeSysMsg(impl_msg);
  impl_obj->libThread();
}
#endif /*CK_TEMPLATES_ONLY*/
#ifndef CK_TEMPLATES_ONLY
void CkIndex_Main::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  CkRegisterBase(__idx, CkIndex_Chare::__idx);
// REG: Main(CkArgMsg* impl_msg);
  __idx_Main_CkArgMsg = CkRegisterEp("Main(CkArgMsg* impl_msg)",
     (CkCallFnPtr)_call_Main_CkArgMsg, CMessage_CkArgMsg::__idx, __idx, 0);
  CkRegisterMainChare(__idx, __idx_Main_CkArgMsg);

// REG: threaded void libThread(void);
  __idx_libThread_void = CkRegisterEp("libThread(void)",
     (CkCallFnPtr)_call_libThread_void, 0, __idx, 0);
}
#endif

/* DEFS: array Places: ArrayElement{
Places(CkMigrateMessage* impl_msg);
void Places(void);
threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_Places::__idx=0;
#endif
#ifndef CK_TEMPLATES_ONLY
/* DEFS: Places(CkMigrateMessage* impl_msg);
 */

/* DEFS: void Places(void);
 */
void CProxyElement_Places::insert(int onPE)
{ 
  void *impl_msg = CkAllocSysMsg();
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_Places::__idx_Places_void,onPE);
}

/* DEFS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
void CProxyElement_Places::startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int which_statement, const CkFutureID &ftHandle, int pe_src
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|which_statement;
    //Have to cast away const-ness to get pup routine
    implP|(CkFutureID &)ftHandle;
    implP|pe_src;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|which_statement;
    //Have to cast away const-ness to get pup routine
    implP|(CkFutureID &)ftHandle;
    implP|pe_src;
  }
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Places::__idx_startAsync_marshall2,0);
}
/* DEFS: Places(CkMigrateMessage* impl_msg);
 */
 int CkIndex_Places::__idx_Places_CkMigrateMessage=0;
void CkIndex_Places::_call_Places_CkMigrateMessage(void* impl_msg,Places * impl_obj)
{
  new (impl_obj) Places((CkMigrateMessage*)impl_msg);
}

/* DEFS: void Places(void);
 */
CkArrayID CProxy_Places::ckNew(const CkArrayOptions &opts)
{ 
  void *impl_msg = CkAllocSysMsg();
   return ckCreateArray((CkArrayMessage *)impl_msg,CkIndex_Places::__idx_Places_void,opts);
}
 int CkIndex_Places::__idx_Places_void=0;
void CkIndex_Places::_call_Places_void(void* impl_msg,Places * impl_obj)
{
  CkFreeSysMsg(impl_msg);
  new (impl_obj) Places();
}

/* DEFS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
void CProxy_Places::startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int which_statement, const CkFutureID &ftHandle, int pe_src
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|which_statement;
    //Have to cast away const-ness to get pup routine
    implP|(CkFutureID &)ftHandle;
    implP|pe_src;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|which_statement;
    //Have to cast away const-ness to get pup routine
    implP|(CkFutureID &)ftHandle;
    implP|pe_src;
  }
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Places::__idx_startAsync_marshall2,0);
}
 int CkIndex_Places::__idx_startAsync_marshall2=0;
void CkIndex_Places::_call_startAsync_marshall2(void* impl_msg,Places * impl_obj)
{
  CthThread tid = CthCreate((CthVoidFn)_callthr_startAsync_marshall2, new CkThrCallArg(impl_msg,impl_obj), 0);
  ((Chare *)impl_obj)->CkAddThreadListeners(tid,impl_msg);
  CthAwaken(tid);
}
void CkIndex_Places::_callthr_startAsync_marshall2(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  Places *impl_obj = (Places *) impl_arg->obj;
  delete impl_arg;
  char *impl_buf=((CkMarshallMsg *)impl_msg)->msgBuf;
  /*Unmarshall pup'd fields: int which_statement, const CkFutureID &ftHandle, int pe_src*/
  PUP::fromMem implP(impl_buf);
  int which_statement; implP|which_statement;
  CkFutureID ftHandle; implP|ftHandle;
  int pe_src; implP|pe_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  impl_obj->startAsync(which_statement, ftHandle, pe_src);
  delete (CkMarshallMsg *)impl_msg;
}
void CkIndex_Places::_marshallmessagepup_startAsync_marshall2(PUP::er &implDestP,void *impl_msg) {
  char *impl_buf=((CkMarshallMsg *)impl_msg)->msgBuf;
  /*Unmarshall pup'd fields: int which_statement, const CkFutureID &ftHandle, int pe_src*/
  PUP::fromMem implP(impl_buf);
  int which_statement; implP|which_statement;
  CkFutureID ftHandle; implP|ftHandle;
  int pe_src; implP|pe_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("which_statement");
  implDestP|which_statement;
  if (implDestP.hasComments()) implDestP.comment("ftHandle");
  implDestP|ftHandle;
  if (implDestP.hasComments()) implDestP.comment("pe_src");
  implDestP|pe_src;
}
/* DEFS: Places(CkMigrateMessage* impl_msg);
 */

/* DEFS: void Places(void);
 */

/* DEFS: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
 */
void CProxySection_Places::startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int which_statement, const CkFutureID &ftHandle, int pe_src
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|which_statement;
    //Have to cast away const-ness to get pup routine
    implP|(CkFutureID &)ftHandle;
    implP|pe_src;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|which_statement;
    //Have to cast away const-ness to get pup routine
    implP|(CkFutureID &)ftHandle;
    implP|pe_src;
  }
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Places::__idx_startAsync_marshall2,0);
}
#endif /*CK_TEMPLATES_ONLY*/
#ifndef CK_TEMPLATES_ONLY
void CkIndex_Places::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  CkRegisterBase(__idx, CkIndex_ArrayElement::__idx);
// REG: Places(CkMigrateMessage* impl_msg);
  __idx_Places_CkMigrateMessage = CkRegisterEp("Places(CkMigrateMessage* impl_msg)",
     (CkCallFnPtr)_call_Places_CkMigrateMessage, 0, __idx, 0);
  CkRegisterMigCtor(__idx, __idx_Places_CkMigrateMessage);

// REG: void Places(void);
  __idx_Places_void = CkRegisterEp("Places(void)",
     (CkCallFnPtr)_call_Places_void, 0, __idx, 0);
  CkRegisterDefaultCtor(__idx, __idx_Places_void);

// REG: threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
  __idx_startAsync_marshall2 = CkRegisterEp("startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src)",
     (CkCallFnPtr)_call_startAsync_marshall2, CkMarshallMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(__idx_startAsync_marshall2,(CkMessagePupFn)_marshallmessagepup_startAsync_marshall2);
}
#endif

#ifndef CK_TEMPLATES_ONLY
void _registerX10_test(void)
{
  static int _done = 0; if(_done) return; _done = 1;
/* REG: message asyncMsg;
*/
CMessage_asyncMsg::__register("asyncMsg", sizeof(asyncMsg),(CkPackFnPtr) asyncMsg::pack,(CkUnpackFnPtr) asyncMsg::unpack);

  CkRegisterReadonly("mainProxy","CProxy_Main",sizeof(mainProxy),(void *) &mainProxy,__xlater_roPup_mainProxy);

  CkRegisterReadonly("nPlaces","int",sizeof(nPlaces),(void *) &nPlaces,__xlater_roPup_nPlaces);

  CkRegisterReadonly("placesProxy","CProxy_Places",sizeof(placesProxy),(void *) &placesProxy,__xlater_roPup_placesProxy);

/* REG: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
threaded void libThread(void);
};
*/
  CkIndex_Main::__register("Main", sizeof(Main));

/* REG: array Places: ArrayElement{
Places(CkMigrateMessage* impl_msg);
void Places(void);
threaded void startAsync(int which_statement, const CkFutureID &ftHandle, int pe_src);
};
*/
  CkIndex_Places::__register("Places", sizeof(Places));

}
extern "C" void CkRegisterMainModule(void) {
  _registerX10_test();
}
#endif
