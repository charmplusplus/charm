/* DEFS: message IDXL_DataMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_IDXL_DataMsg::operator new(size_t s){
  return IDXL_DataMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_IDXL_DataMsg::operator new(size_t s,const int pb){
  return IDXL_DataMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_IDXL_DataMsg::operator new(size_t s, int* sz){
  return IDXL_DataMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_IDXL_DataMsg::operator new(size_t s, int* sz,const int pb){
  return IDXL_DataMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_IDXL_DataMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  IDXL_DataMsg *newmsg = (IDXL_DataMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_IDXL_DataMsg::pack(IDXL_DataMsg *msg) {
  return (void *) msg;
}
IDXL_DataMsg* CMessage_IDXL_DataMsg::unpack(void* buf) {
  IDXL_DataMsg *msg = (IDXL_DataMsg *) buf;
  return msg;
}
int CMessage_IDXL_DataMsg::__idx=0;
#endif


/* DEFS: array IDXL_Chunk: ArrayElement{
IDXL_Chunk(CkMigrateMessage* impl_msg);
void IDXL_Chunk(const CkArrayID &threadArrayID);
void idxl_recv(IDXL_DataMsg* impl_msg);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_IDXL_Chunk::__idx=0;
#endif
#ifndef CK_TEMPLATES_ONLY
/* DEFS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */

/* DEFS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */
void CProxyElement_IDXL_Chunk::insert(const CkArrayID &threadArrayID, int onPE, const CkEntryOptions *impl_e_opts)
{ 
  //Marshall: const CkArrayID &threadArrayID
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkArrayID &)threadArrayID;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkArrayID &)threadArrayID;
  }
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_IDXL_Chunk::__idx_IDXL_Chunk_marshall1,onPE);
}

/* DEFS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
void CProxyElement_IDXL_Chunk::idxl_recv(IDXL_DataMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_IDXL_Chunk::__idx_idxl_recv_IDXL_DataMsg);
}
/* DEFS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */
 int CkIndex_IDXL_Chunk::__idx_IDXL_Chunk_CkMigrateMessage=0;
void CkIndex_IDXL_Chunk::_call_IDXL_Chunk_CkMigrateMessage(void* impl_msg,IDXL_Chunk * impl_obj)
{
  new (impl_obj) IDXL_Chunk((CkMigrateMessage*)impl_msg);
}

/* DEFS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */
CkArrayID CProxy_IDXL_Chunk::ckNew(const CkArrayID &threadArrayID, const CkArrayOptions &opts, const CkEntryOptions *impl_e_opts)
{ 
  //Marshall: const CkArrayID &threadArrayID
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkArrayID &)threadArrayID;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkArrayID &)threadArrayID;
  }
   return ckCreateArray((CkArrayMessage *)impl_msg,CkIndex_IDXL_Chunk::__idx_IDXL_Chunk_marshall1,opts);
}
 int CkIndex_IDXL_Chunk::__idx_IDXL_Chunk_marshall1=0;
void CkIndex_IDXL_Chunk::_call_IDXL_Chunk_marshall1(void* impl_msg,IDXL_Chunk * impl_obj)
{
  char *impl_buf=((CkMarshallMsg *)impl_msg)->msgBuf;
  //Unmarshall pup'd fields: const CkArrayID &threadArrayID
  PUP::fromMem implP(impl_buf);
  CkArrayID threadArrayID; implP|threadArrayID;
  impl_buf+=CK_ALIGN(implP.size(),16);
  //Unmarshall arrays:
  new (impl_obj) IDXL_Chunk(threadArrayID);
  delete (CkMarshallMsg *)impl_msg;
}
int CkIndex_IDXL_Chunk::_callmarshall_IDXL_Chunk_marshall1(char* impl_buf,IDXL_Chunk * impl_obj) {
  //Unmarshall pup'd fields: const CkArrayID &threadArrayID
  PUP::fromMem implP(impl_buf);
  CkArrayID threadArrayID; implP|threadArrayID;
  impl_buf+=CK_ALIGN(implP.size(),16);
  //Unmarshall arrays:
  new (impl_obj) IDXL_Chunk(threadArrayID);
  return implP.size();
}

/* DEFS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
void CProxy_IDXL_Chunk::idxl_recv(IDXL_DataMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_IDXL_Chunk::__idx_idxl_recv_IDXL_DataMsg);
}
 int CkIndex_IDXL_Chunk::__idx_idxl_recv_IDXL_DataMsg=0;
void CkIndex_IDXL_Chunk::_call_idxl_recv_IDXL_DataMsg(void* impl_msg,IDXL_Chunk * impl_obj)
{
  impl_obj->idxl_recv((IDXL_DataMsg*)impl_msg);
}
/* DEFS: IDXL_Chunk(CkMigrateMessage* impl_msg);
 */

/* DEFS: void IDXL_Chunk(const CkArrayID &threadArrayID);
 */

/* DEFS: void idxl_recv(IDXL_DataMsg* impl_msg);
 */
void CProxySection_IDXL_Chunk::idxl_recv(IDXL_DataMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_IDXL_Chunk::__idx_idxl_recv_IDXL_DataMsg);
}
#endif /*CK_TEMPLATES_ONLY*/
#ifndef CK_TEMPLATES_ONLY
void CkIndex_IDXL_Chunk::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  _REGISTER_BASE(__idx, CkIndex_ArrayElement::__idx);
// REG: IDXL_Chunk(CkMigrateMessage* impl_msg);
  __idx_IDXL_Chunk_CkMigrateMessage = CkRegisterEp("IDXL_Chunk(CkMigrateMessage* impl_msg)",
     (CkCallFnPtr)_call_IDXL_Chunk_CkMigrateMessage, 0, __idx);
  CkRegisterMigCtor(__idx, __idx_IDXL_Chunk_CkMigrateMessage);

// REG: void IDXL_Chunk(const CkArrayID &threadArrayID);
  __idx_IDXL_Chunk_marshall1 = CkRegisterEp("IDXL_Chunk(const CkArrayID &threadArrayID)",
     (CkCallFnPtr)_call_IDXL_Chunk_marshall1, CkMarshallMsg::__idx, __idx);
  CkRegisterMarshallUnpackFn(__idx_IDXL_Chunk_marshall1,(CkMarshallUnpackFn)_callmarshall_IDXL_Chunk_marshall1);

// REG: void idxl_recv(IDXL_DataMsg* impl_msg);
  __idx_idxl_recv_IDXL_DataMsg = CkRegisterEp("idxl_recv(IDXL_DataMsg* impl_msg)",
     (CkCallFnPtr)_call_idxl_recv_IDXL_DataMsg, CMessage_IDXL_DataMsg::__idx, __idx);
}
#endif

#ifndef CK_TEMPLATES_ONLY
void _registeridxl(void)
{
  static int _done = 0; if(_done) return; _done = 1;
/* REG: message IDXL_DataMsg;
*/
CMessage_IDXL_DataMsg::__register("IDXL_DataMsg", sizeof(IDXL_DataMsg),(CkPackFnPtr) IDXL_DataMsg::pack,(CkUnpackFnPtr) IDXL_DataMsg::unpack);

      IDXLnodeInit();

/* REG: array IDXL_Chunk: ArrayElement{
IDXL_Chunk(CkMigrateMessage* impl_msg);
void IDXL_Chunk(const CkArrayID &threadArrayID);
void idxl_recv(IDXL_DataMsg* impl_msg);
};
*/
  CkIndex_IDXL_Chunk::__register("IDXL_Chunk", sizeof(IDXL_Chunk));

}
#endif
