

/* DEFS: message SmallWorkMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_SmallWorkMsg::operator new(size_t s){
  return SmallWorkMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_SmallWorkMsg::operator new(size_t s,const int pb){
  return SmallWorkMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_SmallWorkMsg::operator new(size_t s, int* sz){
  return SmallWorkMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_SmallWorkMsg::operator new(size_t s, int* sz,const int pb){
  return SmallWorkMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_SmallWorkMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  SmallWorkMsg *newmsg = (SmallWorkMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_SmallWorkMsg::pack(SmallWorkMsg *msg) {
  return (void *) msg;
}
SmallWorkMsg* CMessage_SmallWorkMsg::unpack(void* buf) {
  SmallWorkMsg *msg = (SmallWorkMsg *) buf;
  return msg;
}
int CMessage_SmallWorkMsg::__idx=0;
#endif

/* DEFS: message WorkerData;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_WorkerData::operator new(size_t s){
  return WorkerData::alloc(__idx, s, 0, 0);
}
void *CMessage_WorkerData::operator new(size_t s,const int pb){
  return WorkerData::alloc(__idx, s, 0, pb);
}
void *CMessage_WorkerData::operator new(size_t s, int* sz){
  return WorkerData::alloc(__idx, s, sz, 0);
}
void *CMessage_WorkerData::operator new(size_t s, int* sz,const int pb){
  return WorkerData::alloc(__idx, s, sz, pb);
}
void* CMessage_WorkerData::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  WorkerData *newmsg = (WorkerData *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_WorkerData::pack(WorkerData *msg) {
  return (void *) msg;
}
WorkerData* CMessage_WorkerData::unpack(void* buf) {
  WorkerData *msg = (WorkerData *) buf;
  return msg;
}
int CMessage_WorkerData::__idx=0;
#endif

/* DEFS: message LargeWorkMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_LargeWorkMsg::operator new(size_t s){
  return LargeWorkMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_LargeWorkMsg::operator new(size_t s,const int pb){
  return LargeWorkMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_LargeWorkMsg::operator new(size_t s, int* sz){
  return LargeWorkMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_LargeWorkMsg::operator new(size_t s, int* sz,const int pb){
  return LargeWorkMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_LargeWorkMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  LargeWorkMsg *newmsg = (LargeWorkMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_LargeWorkMsg::pack(LargeWorkMsg *msg) {
  return (void *) msg;
}
LargeWorkMsg* CMessage_LargeWorkMsg::unpack(void* buf) {
  LargeWorkMsg *msg = (LargeWorkMsg *) buf;
  return msg;
}
int CMessage_LargeWorkMsg::__idx=0;
#endif

/* DEFS: message MediumWorkMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_MediumWorkMsg::operator new(size_t s){
  return MediumWorkMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_MediumWorkMsg::operator new(size_t s,const int pb){
  return MediumWorkMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_MediumWorkMsg::operator new(size_t s, int* sz){
  return MediumWorkMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_MediumWorkMsg::operator new(size_t s, int* sz,const int pb){
  return MediumWorkMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_MediumWorkMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  MediumWorkMsg *newmsg = (MediumWorkMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_MediumWorkMsg::pack(MediumWorkMsg *msg) {
  return (void *) msg;
}
MediumWorkMsg* CMessage_MediumWorkMsg::unpack(void* buf) {
  MediumWorkMsg *msg = (MediumWorkMsg *) buf;
  return msg;
}
int CMessage_MediumWorkMsg::__idx=0;
#endif

/* DEFS: array worker: sim{
worker(CkMigrateMessage* impl_msg);
worker(WorkerData* impl_msg);
void workSmall(SmallWorkMsg* impl_msg);
void workMedium(MediumWorkMsg* impl_msg);
void workLarge(LargeWorkMsg* impl_msg);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_worker::__idx=0;
#endif
#ifndef CK_TEMPLATES_ONLY
/* DEFS: worker(CkMigrateMessage* impl_msg);
 */

/* DEFS: worker(WorkerData* impl_msg);
 */
void CProxyElement_worker::insert(WorkerData* impl_msg, int onPE)
{ 
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_worker::__idx_worker_WorkerData,onPE);
}

/* DEFS: void workSmall(SmallWorkMsg* impl_msg);
 */
void CProxyElement_worker::workSmall(SmallWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_worker::__idx_workSmall_SmallWorkMsg);
}

/* DEFS: void workMedium(MediumWorkMsg* impl_msg);
 */
void CProxyElement_worker::workMedium(MediumWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_worker::__idx_workMedium_MediumWorkMsg);
}

/* DEFS: void workLarge(LargeWorkMsg* impl_msg);
 */
void CProxyElement_worker::workLarge(LargeWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_worker::__idx_workLarge_LargeWorkMsg);
}
/* DEFS: worker(CkMigrateMessage* impl_msg);
 */
 int CkIndex_worker::__idx_worker_CkMigrateMessage=0;
void CkIndex_worker::_call_worker_CkMigrateMessage(void* impl_msg,worker * impl_obj)
{
  new (impl_obj) worker((CkMigrateMessage*)impl_msg);
}

/* DEFS: worker(WorkerData* impl_msg);
 */
CkArrayID CProxy_worker::ckNew(WorkerData* impl_msg, const CkArrayOptions &opts)
{ 
   return ckCreateArray((CkArrayMessage *)impl_msg,CkIndex_worker::__idx_worker_WorkerData,opts);
}
 int CkIndex_worker::__idx_worker_WorkerData=0;
void CkIndex_worker::_call_worker_WorkerData(void* impl_msg,worker * impl_obj)
{
  new (impl_obj) worker((WorkerData*)impl_msg);
}

/* DEFS: void workSmall(SmallWorkMsg* impl_msg);
 */
void CProxy_worker::workSmall(SmallWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_worker::__idx_workSmall_SmallWorkMsg);
}
 int CkIndex_worker::__idx_workSmall_SmallWorkMsg=0;
void CkIndex_worker::_call_workSmall_SmallWorkMsg(void* impl_msg,worker * impl_obj)
{
  impl_obj->workSmall((SmallWorkMsg*)impl_msg);
}

/* DEFS: void workMedium(MediumWorkMsg* impl_msg);
 */
void CProxy_worker::workMedium(MediumWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_worker::__idx_workMedium_MediumWorkMsg);
}
 int CkIndex_worker::__idx_workMedium_MediumWorkMsg=0;
void CkIndex_worker::_call_workMedium_MediumWorkMsg(void* impl_msg,worker * impl_obj)
{
  impl_obj->workMedium((MediumWorkMsg*)impl_msg);
}

/* DEFS: void workLarge(LargeWorkMsg* impl_msg);
 */
void CProxy_worker::workLarge(LargeWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_worker::__idx_workLarge_LargeWorkMsg);
}
 int CkIndex_worker::__idx_workLarge_LargeWorkMsg=0;
void CkIndex_worker::_call_workLarge_LargeWorkMsg(void* impl_msg,worker * impl_obj)
{
  impl_obj->workLarge((LargeWorkMsg*)impl_msg);
}
/* DEFS: worker(CkMigrateMessage* impl_msg);
 */

/* DEFS: worker(WorkerData* impl_msg);
 */

/* DEFS: void workSmall(SmallWorkMsg* impl_msg);
 */
void CProxySection_worker::workSmall(SmallWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_worker::__idx_workSmall_SmallWorkMsg);
}

/* DEFS: void workMedium(MediumWorkMsg* impl_msg);
 */
void CProxySection_worker::workMedium(MediumWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_worker::__idx_workMedium_MediumWorkMsg);
}

/* DEFS: void workLarge(LargeWorkMsg* impl_msg);
 */
void CProxySection_worker::workLarge(LargeWorkMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_worker::__idx_workLarge_LargeWorkMsg);
}
#endif /*CK_TEMPLATES_ONLY*/
#ifndef CK_TEMPLATES_ONLY
void CkIndex_worker::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  CkRegisterBase(__idx, CkIndex_sim::__idx);
// REG: worker(CkMigrateMessage* impl_msg);
  __idx_worker_CkMigrateMessage = CkRegisterEp("worker(CkMigrateMessage* impl_msg)",
     (CkCallFnPtr)_call_worker_CkMigrateMessage, 0, __idx, 0);
  CkRegisterMigCtor(__idx, __idx_worker_CkMigrateMessage);

// REG: worker(WorkerData* impl_msg);
  __idx_worker_WorkerData = CkRegisterEp("worker(WorkerData* impl_msg)",
     (CkCallFnPtr)_call_worker_WorkerData, CMessage_WorkerData::__idx, __idx, 0);

// REG: void workSmall(SmallWorkMsg* impl_msg);
  __idx_workSmall_SmallWorkMsg = CkRegisterEp("workSmall(SmallWorkMsg* impl_msg)",
     (CkCallFnPtr)_call_workSmall_SmallWorkMsg, CMessage_SmallWorkMsg::__idx, __idx, 0);

// REG: void workMedium(MediumWorkMsg* impl_msg);
  __idx_workMedium_MediumWorkMsg = CkRegisterEp("workMedium(MediumWorkMsg* impl_msg)",
     (CkCallFnPtr)_call_workMedium_MediumWorkMsg, CMessage_MediumWorkMsg::__idx, __idx, 0);

// REG: void workLarge(LargeWorkMsg* impl_msg);
  __idx_workLarge_LargeWorkMsg = CkRegisterEp("workLarge(LargeWorkMsg* impl_msg)",
     (CkCallFnPtr)_call_workLarge_LargeWorkMsg, CMessage_LargeWorkMsg::__idx, __idx, 0);
}
#endif

#ifndef CK_TEMPLATES_ONLY
void _registerWorker(void)
{
  static int _done = 0; if(_done) return; _done = 1;
      _registersim();

      _registerpose();

/* REG: message SmallWorkMsg;
*/
CMessage_SmallWorkMsg::__register("SmallWorkMsg", sizeof(SmallWorkMsg),(CkPackFnPtr) SmallWorkMsg::pack,(CkUnpackFnPtr) SmallWorkMsg::unpack);

/* REG: message WorkerData;
*/
CMessage_WorkerData::__register("WorkerData", sizeof(WorkerData),(CkPackFnPtr) WorkerData::pack,(CkUnpackFnPtr) WorkerData::unpack);

/* REG: message LargeWorkMsg;
*/
CMessage_LargeWorkMsg::__register("LargeWorkMsg", sizeof(LargeWorkMsg),(CkPackFnPtr) LargeWorkMsg::pack,(CkUnpackFnPtr) LargeWorkMsg::unpack);

/* REG: message MediumWorkMsg;
*/
CMessage_MediumWorkMsg::__register("MediumWorkMsg", sizeof(MediumWorkMsg),(CkPackFnPtr) MediumWorkMsg::pack,(CkUnpackFnPtr) MediumWorkMsg::unpack);

/* REG: array worker: sim{
worker(CkMigrateMessage* impl_msg);
worker(WorkerData* impl_msg);
void workSmall(SmallWorkMsg* impl_msg);
void workMedium(MediumWorkMsg* impl_msg);
void workLarge(LargeWorkMsg* impl_msg);
};
*/
  CkIndex_worker::__register("worker", sizeof(worker));

}
#endif
