/* DEFS: message chunkMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_chunkMsg::operator new(size_t s){
  return chunkMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_chunkMsg::operator new(size_t s,const int pb){
  return chunkMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_chunkMsg::operator new(size_t s, int* sz){
  return chunkMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_chunkMsg::operator new(size_t s, int* sz,const int pb){
  return chunkMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_chunkMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  chunkMsg *newmsg = (chunkMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_chunkMsg::pack(chunkMsg *msg) {
  return (void *) msg;
}
chunkMsg* CMessage_chunkMsg::unpack(void* buf) {
  chunkMsg *msg = (chunkMsg *) buf;
  return msg;
}
int CMessage_chunkMsg::__idx=0;
#endif

/* DEFS: message nodeMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_nodeMsg::operator new(size_t s){
  return nodeMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_nodeMsg::operator new(size_t s,const int pb){
  return nodeMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_nodeMsg::operator new(size_t s, int* sz){
  return nodeMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_nodeMsg::operator new(size_t s, int* sz,const int pb){
  return nodeMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_nodeMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  nodeMsg *newmsg = (nodeMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_nodeMsg::pack(nodeMsg *msg) {
  return (void *) msg;
}
nodeMsg* CMessage_nodeMsg::unpack(void* buf) {
  nodeMsg *msg = (nodeMsg *) buf;
  return msg;
}
int CMessage_nodeMsg::__idx=0;
#endif

/* DEFS: message edgeMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_edgeMsg::operator new(size_t s){
  return edgeMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_edgeMsg::operator new(size_t s,const int pb){
  return edgeMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_edgeMsg::operator new(size_t s, int* sz){
  return edgeMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_edgeMsg::operator new(size_t s, int* sz,const int pb){
  return edgeMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_edgeMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  edgeMsg *newmsg = (edgeMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_edgeMsg::pack(edgeMsg *msg) {
  return (void *) msg;
}
edgeMsg* CMessage_edgeMsg::unpack(void* buf) {
  edgeMsg *msg = (edgeMsg *) buf;
  return msg;
}
int CMessage_edgeMsg::__idx=0;
#endif

/* DEFS: message remoteEdgeMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_remoteEdgeMsg::operator new(size_t s){
  return remoteEdgeMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_remoteEdgeMsg::operator new(size_t s,const int pb){
  return remoteEdgeMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_remoteEdgeMsg::operator new(size_t s, int* sz){
  return remoteEdgeMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_remoteEdgeMsg::operator new(size_t s, int* sz,const int pb){
  return remoteEdgeMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_remoteEdgeMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  remoteEdgeMsg *newmsg = (remoteEdgeMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_remoteEdgeMsg::pack(remoteEdgeMsg *msg) {
  return (void *) msg;
}
remoteEdgeMsg* CMessage_remoteEdgeMsg::unpack(void* buf) {
  remoteEdgeMsg *msg = (remoteEdgeMsg *) buf;
  return msg;
}
int CMessage_remoteEdgeMsg::__idx=0;
#endif

/* DEFS: message elementMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_elementMsg::operator new(size_t s){
  return elementMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_elementMsg::operator new(size_t s,const int pb){
  return elementMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_elementMsg::operator new(size_t s, int* sz){
  return elementMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_elementMsg::operator new(size_t s, int* sz,const int pb){
  return elementMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_elementMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  elementMsg *newmsg = (elementMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_elementMsg::pack(elementMsg *msg) {
  return (void *) msg;
}
elementMsg* CMessage_elementMsg::unpack(void* buf) {
  elementMsg *msg = (elementMsg *) buf;
  return msg;
}
int CMessage_elementMsg::__idx=0;
#endif

/* DEFS: message femElementMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_femElementMsg::operator new(size_t s){
  return femElementMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_femElementMsg::operator new(size_t s,const int pb){
  return femElementMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_femElementMsg::operator new(size_t s, int* sz){
  return femElementMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_femElementMsg::operator new(size_t s, int* sz,const int pb){
  return femElementMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_femElementMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  femElementMsg *newmsg = (femElementMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_femElementMsg::pack(femElementMsg *msg) {
  return (void *) msg;
}
femElementMsg* CMessage_femElementMsg::unpack(void* buf) {
  femElementMsg *msg = (femElementMsg *) buf;
  return msg;
}
int CMessage_femElementMsg::__idx=0;
#endif

/* DEFS: message ghostElementMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_ghostElementMsg::operator new(size_t s){
  return ghostElementMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_ghostElementMsg::operator new(size_t s,const int pb){
  return ghostElementMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_ghostElementMsg::operator new(size_t s, int* sz){
  return ghostElementMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_ghostElementMsg::operator new(size_t s, int* sz,const int pb){
  return ghostElementMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_ghostElementMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  ghostElementMsg *newmsg = (ghostElementMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_ghostElementMsg::pack(ghostElementMsg *msg) {
  return (void *) msg;
}
ghostElementMsg* CMessage_ghostElementMsg::unpack(void* buf) {
  ghostElementMsg *msg = (ghostElementMsg *) buf;
  return msg;
}
int CMessage_ghostElementMsg::__idx=0;
#endif

/* DEFS: message refineMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_refineMsg::operator new(size_t s){
  return refineMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_refineMsg::operator new(size_t s,const int pb){
  return refineMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_refineMsg::operator new(size_t s, int* sz){
  return refineMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_refineMsg::operator new(size_t s, int* sz,const int pb){
  return refineMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_refineMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  refineMsg *newmsg = (refineMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_refineMsg::pack(refineMsg *msg) {
  return (void *) msg;
}
refineMsg* CMessage_refineMsg::unpack(void* buf) {
  refineMsg *msg = (refineMsg *) buf;
  return msg;
}
int CMessage_refineMsg::__idx=0;
#endif

/* DEFS: message coarsenMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_coarsenMsg::operator new(size_t s){
  return coarsenMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_coarsenMsg::operator new(size_t s,const int pb){
  return coarsenMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_coarsenMsg::operator new(size_t s, int* sz){
  return coarsenMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_coarsenMsg::operator new(size_t s, int* sz,const int pb){
  return coarsenMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_coarsenMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  coarsenMsg *newmsg = (coarsenMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_coarsenMsg::pack(coarsenMsg *msg) {
  return (void *) msg;
}
coarsenMsg* CMessage_coarsenMsg::unpack(void* buf) {
  coarsenMsg *msg = (coarsenMsg *) buf;
  return msg;
}
int CMessage_coarsenMsg::__idx=0;
#endif

/* DEFS: message collapseMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_collapseMsg::operator new(size_t s){
  return collapseMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_collapseMsg::operator new(size_t s,const int pb){
  return collapseMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_collapseMsg::operator new(size_t s, int* sz){
  return collapseMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_collapseMsg::operator new(size_t s, int* sz,const int pb){
  return collapseMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_collapseMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  collapseMsg *newmsg = (collapseMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_collapseMsg::pack(collapseMsg *msg) {
  return (void *) msg;
}
collapseMsg* CMessage_collapseMsg::unpack(void* buf) {
  collapseMsg *msg = (collapseMsg *) buf;
  return msg;
}
int CMessage_collapseMsg::__idx=0;
#endif

/* DEFS: message splitInMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_splitInMsg::operator new(size_t s){
  return splitInMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_splitInMsg::operator new(size_t s,const int pb){
  return splitInMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_splitInMsg::operator new(size_t s, int* sz){
  return splitInMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_splitInMsg::operator new(size_t s, int* sz,const int pb){
  return splitInMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_splitInMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  splitInMsg *newmsg = (splitInMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_splitInMsg::pack(splitInMsg *msg) {
  return (void *) msg;
}
splitInMsg* CMessage_splitInMsg::unpack(void* buf) {
  splitInMsg *msg = (splitInMsg *) buf;
  return msg;
}
int CMessage_splitInMsg::__idx=0;
#endif

/* DEFS: message splitOutMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_splitOutMsg::operator new(size_t s){
  return splitOutMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_splitOutMsg::operator new(size_t s,const int pb){
  return splitOutMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_splitOutMsg::operator new(size_t s, int* sz){
  return splitOutMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_splitOutMsg::operator new(size_t s, int* sz,const int pb){
  return splitOutMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_splitOutMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  splitOutMsg *newmsg = (splitOutMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_splitOutMsg::pack(splitOutMsg *msg) {
  return (void *) msg;
}
splitOutMsg* CMessage_splitOutMsg::unpack(void* buf) {
  splitOutMsg *msg = (splitOutMsg *) buf;
  return msg;
}
int CMessage_splitOutMsg::__idx=0;
#endif

/* DEFS: message updateMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_updateMsg::operator new(size_t s){
  return updateMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_updateMsg::operator new(size_t s,const int pb){
  return updateMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_updateMsg::operator new(size_t s, int* sz){
  return updateMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_updateMsg::operator new(size_t s, int* sz,const int pb){
  return updateMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_updateMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  updateMsg *newmsg = (updateMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_updateMsg::pack(updateMsg *msg) {
  return (void *) msg;
}
updateMsg* CMessage_updateMsg::unpack(void* buf) {
  updateMsg *msg = (updateMsg *) buf;
  return msg;
}
int CMessage_updateMsg::__idx=0;
#endif

/* DEFS: message specialRequestMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_specialRequestMsg::operator new(size_t s){
  return specialRequestMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_specialRequestMsg::operator new(size_t s,const int pb){
  return specialRequestMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_specialRequestMsg::operator new(size_t s, int* sz){
  return specialRequestMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_specialRequestMsg::operator new(size_t s, int* sz,const int pb){
  return specialRequestMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_specialRequestMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  specialRequestMsg *newmsg = (specialRequestMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_specialRequestMsg::pack(specialRequestMsg *msg) {
  return (void *) msg;
}
specialRequestMsg* CMessage_specialRequestMsg::unpack(void* buf) {
  specialRequestMsg *msg = (specialRequestMsg *) buf;
  return msg;
}
int CMessage_specialRequestMsg::__idx=0;
#endif

/* DEFS: message specialResponseMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_specialResponseMsg::operator new(size_t s){
  return specialResponseMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_specialResponseMsg::operator new(size_t s,const int pb){
  return specialResponseMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_specialResponseMsg::operator new(size_t s, int* sz){
  return specialResponseMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_specialResponseMsg::operator new(size_t s, int* sz,const int pb){
  return specialResponseMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_specialResponseMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  specialResponseMsg *newmsg = (specialResponseMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_specialResponseMsg::pack(specialResponseMsg *msg) {
  return (void *) msg;
}
specialResponseMsg* CMessage_specialResponseMsg::unpack(void* buf) {
  specialResponseMsg *msg = (specialResponseMsg *) buf;
  return msg;
}
int CMessage_specialResponseMsg::__idx=0;
#endif

/* DEFS: message refMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_refMsg::operator new(size_t s){
  return refMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_refMsg::operator new(size_t s,const int pb){
  return refMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_refMsg::operator new(size_t s, int* sz){
  return refMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_refMsg::operator new(size_t s, int* sz,const int pb){
  return refMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_refMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  refMsg *newmsg = (refMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_refMsg::pack(refMsg *msg) {
  return (void *) msg;
}
refMsg* CMessage_refMsg::unpack(void* buf) {
  refMsg *msg = (refMsg *) buf;
  return msg;
}
int CMessage_refMsg::__idx=0;
#endif

/* DEFS: message drefMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_drefMsg::operator new(size_t s){
  return drefMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_drefMsg::operator new(size_t s,const int pb){
  return drefMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_drefMsg::operator new(size_t s, int* sz){
  return drefMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_drefMsg::operator new(size_t s, int* sz,const int pb){
  return drefMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_drefMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  drefMsg *newmsg = (drefMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_drefMsg::pack(drefMsg *msg) {
  return (void *) msg;
}
drefMsg* CMessage_drefMsg::unpack(void* buf) {
  drefMsg *msg = (drefMsg *) buf;
  return msg;
}
int CMessage_drefMsg::__idx=0;
#endif

/* DEFS: message edgeUpdateMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_edgeUpdateMsg::operator new(size_t s){
  return edgeUpdateMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_edgeUpdateMsg::operator new(size_t s,const int pb){
  return edgeUpdateMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_edgeUpdateMsg::operator new(size_t s, int* sz){
  return edgeUpdateMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_edgeUpdateMsg::operator new(size_t s, int* sz,const int pb){
  return edgeUpdateMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_edgeUpdateMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  edgeUpdateMsg *newmsg = (edgeUpdateMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_edgeUpdateMsg::pack(edgeUpdateMsg *msg) {
  return (void *) msg;
}
edgeUpdateMsg* CMessage_edgeUpdateMsg::unpack(void* buf) {
  edgeUpdateMsg *msg = (edgeUpdateMsg *) buf;
  return msg;
}
int CMessage_edgeUpdateMsg::__idx=0;
#endif

/* DEFS: message intMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_intMsg::operator new(size_t s){
  return intMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_intMsg::operator new(size_t s,const int pb){
  return intMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_intMsg::operator new(size_t s, int* sz){
  return intMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_intMsg::operator new(size_t s, int* sz,const int pb){
  return intMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_intMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  intMsg *newmsg = (intMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_intMsg::pack(intMsg *msg) {
  return (void *) msg;
}
intMsg* CMessage_intMsg::unpack(void* buf) {
  intMsg *msg = (intMsg *) buf;
  return msg;
}
int CMessage_intMsg::__idx=0;
#endif

/* DEFS: message doubleMsg;
 */
#ifndef CK_TEMPLATES_ONLY
void *CMessage_doubleMsg::operator new(size_t s){
  return doubleMsg::alloc(__idx, s, 0, 0);
}
void *CMessage_doubleMsg::operator new(size_t s,const int pb){
  return doubleMsg::alloc(__idx, s, 0, pb);
}
void *CMessage_doubleMsg::operator new(size_t s, int* sz){
  return doubleMsg::alloc(__idx, s, sz, 0);
}
void *CMessage_doubleMsg::operator new(size_t s, int* sz,const int pb){
  return doubleMsg::alloc(__idx, s, sz, pb);
}
void* CMessage_doubleMsg::alloc(int msgnum, size_t sz, int *sizes, int pb) {
  int offsets[1];
  offsets[0] = ALIGN8(sz);
  doubleMsg *newmsg = (doubleMsg *) CkAllocMsg(msgnum, offsets[0], pb);
  return (void *) newmsg;
}
void* CMessage_doubleMsg::pack(doubleMsg *msg) {
  return (void *) msg;
}
doubleMsg* CMessage_doubleMsg::unpack(void* buf) {
  doubleMsg *msg = (doubleMsg *) buf;
  return msg;
}
int CMessage_doubleMsg::__idx=0;
#endif

/* DEFS: readonly CProxy_chunk mesh;
 */
extern CProxy_chunk mesh;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_mesh(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|mesh;
}
#endif


/* DEFS: array chunk: ArrayElement{
chunk(CkMigrateMessage* impl_msg);
chunk(chunkMsg* impl_msg);
void addRemoteEdge(remoteEdgeMsg* impl_msg);
void refineElement(refineMsg* impl_msg);
threaded void refiningElements(void);
void coarsenElement(coarsenMsg* impl_msg);
threaded void coarseningElements(void);
sync nodeMsg* getNode(intMsg* impl_msg);
sync refMsg* getEdge(collapseMsg* impl_msg);
sync void setBorder(intMsg* impl_msg);
sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
sync splitOutMsg* split(splitInMsg* impl_msg);
sync void collapseHelp(collapseMsg* impl_msg);
void checkPending(refMsg* impl_msg);
void checkPending(drefMsg* impl_msg);
sync void updateElement(updateMsg* impl_msg);
sync void updateElementEdge(updateMsg* impl_msg);
void updateReferences(updateMsg* impl_msg);
threaded sync doubleMsg* getArea(intMsg* impl_msg);
threaded sync nodeMsg* midpoint(intMsg* impl_msg);
sync intMsg* setPending(intMsg* impl_msg);
sync void unsetPending(intMsg* impl_msg);
sync intMsg* isPending(intMsg* impl_msg);
sync intMsg* lockNode(intMsg* impl_msg);
sync void unlockNode(intMsg* impl_msg);
threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
sync refMsg* getNeighbor(refMsg* impl_msg);
sync refMsg* getNotNode(refMsg* impl_msg);
sync refMsg* getNotElem(refMsg* impl_msg);
sync void setTargetArea(doubleMsg* impl_msg);
sync void resetTargetArea(doubleMsg* impl_msg);
sync void updateEdges(edgeUpdateMsg* impl_msg);
sync void updateNodeCoords(nodeMsg* impl_msg);
void reportPos(nodeMsg* impl_msg);
threaded sync void print(void);
threaded sync void out_print(void);
void freshen(void);
void deriveBorderNodes(void);
void tweakMesh(void);
void improveChunk(void);
threaded sync void improve(void);
void addNode(nodeMsg* impl_msg);
void addEdge(edgeMsg* impl_msg);
void addElement(elementMsg* impl_msg);
sync void removeNode(intMsg* impl_msg);
sync void removeEdge(intMsg* impl_msg);
sync void removeElement(intMsg* impl_msg);
sync void updateNode(updateMsg* impl_msg);
sync refMsg* getOpposingNode(refMsg* impl_msg);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_chunk::__idx=0;
#endif
#ifndef CK_TEMPLATES_ONLY
/* DEFS: chunk(CkMigrateMessage* impl_msg);
 */

/* DEFS: chunk(chunkMsg* impl_msg);
 */
void CProxyElement_chunk::insert(chunkMsg* impl_msg, int onPE)
{ 
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_chunk::__idx_chunk_chunkMsg,onPE);
}

/* DEFS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
void CProxyElement_chunk::addRemoteEdge(remoteEdgeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addRemoteEdge_remoteEdgeMsg);
}

/* DEFS: void refineElement(refineMsg* impl_msg);
 */
void CProxyElement_chunk::refineElement(refineMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_refineElement_refineMsg);
}

/* DEFS: threaded void refiningElements(void);
 */
void CProxyElement_chunk::refiningElements(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_refiningElements_void);
}

/* DEFS: void coarsenElement(coarsenMsg* impl_msg);
 */
void CProxyElement_chunk::coarsenElement(coarsenMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_coarsenElement_coarsenMsg);
}

/* DEFS: threaded void coarseningElements(void);
 */
void CProxyElement_chunk::coarseningElements(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_coarseningElements_void);
}

/* DEFS: sync nodeMsg* getNode(intMsg* impl_msg);
 */
nodeMsg* CProxyElement_chunk::getNode(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (nodeMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getNode_intMsg));
}

/* DEFS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */
refMsg* CProxyElement_chunk::getEdge(collapseMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (refMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getEdge_collapseMsg));
}

/* DEFS: sync void setBorder(intMsg* impl_msg);
 */
void CProxyElement_chunk::setBorder(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_setBorder_intMsg));
}

/* DEFS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */
intMsg* CProxyElement_chunk::safeToMoveNode(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (intMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_safeToMoveNode_nodeMsg));
}

/* DEFS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */
splitOutMsg* CProxyElement_chunk::split(splitInMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (splitOutMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_split_splitInMsg));
}

/* DEFS: sync void collapseHelp(collapseMsg* impl_msg);
 */
void CProxyElement_chunk::collapseHelp(collapseMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_collapseHelp_collapseMsg));
}

/* DEFS: void checkPending(refMsg* impl_msg);
 */
void CProxyElement_chunk::checkPending(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_checkPending_refMsg);
}

/* DEFS: void checkPending(drefMsg* impl_msg);
 */
void CProxyElement_chunk::checkPending(drefMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_checkPending_drefMsg);
}

/* DEFS: sync void updateElement(updateMsg* impl_msg);
 */
void CProxyElement_chunk::updateElement(updateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_updateElement_updateMsg));
}

/* DEFS: sync void updateElementEdge(updateMsg* impl_msg);
 */
void CProxyElement_chunk::updateElementEdge(updateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_updateElementEdge_updateMsg));
}

/* DEFS: void updateReferences(updateMsg* impl_msg);
 */
void CProxyElement_chunk::updateReferences(updateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_updateReferences_updateMsg);
}

/* DEFS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */
doubleMsg* CProxyElement_chunk::getArea(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (doubleMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getArea_intMsg));
}

/* DEFS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */
nodeMsg* CProxyElement_chunk::midpoint(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (nodeMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_midpoint_intMsg));
}

/* DEFS: sync intMsg* setPending(intMsg* impl_msg);
 */
intMsg* CProxyElement_chunk::setPending(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (intMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_setPending_intMsg));
}

/* DEFS: sync void unsetPending(intMsg* impl_msg);
 */
void CProxyElement_chunk::unsetPending(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_unsetPending_intMsg));
}

/* DEFS: sync intMsg* isPending(intMsg* impl_msg);
 */
intMsg* CProxyElement_chunk::isPending(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (intMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_isPending_intMsg));
}

/* DEFS: sync intMsg* lockNode(intMsg* impl_msg);
 */
intMsg* CProxyElement_chunk::lockNode(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (intMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_lockNode_intMsg));
}

/* DEFS: sync void unlockNode(intMsg* impl_msg);
 */
void CProxyElement_chunk::unlockNode(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_unlockNode_intMsg));
}

/* DEFS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */
intMsg* CProxyElement_chunk::isLongestEdge(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (intMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_isLongestEdge_refMsg));
}

/* DEFS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */
refMsg* CProxyElement_chunk::getNeighbor(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (refMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getNeighbor_refMsg));
}

/* DEFS: sync refMsg* getNotNode(refMsg* impl_msg);
 */
refMsg* CProxyElement_chunk::getNotNode(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (refMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getNotNode_refMsg));
}

/* DEFS: sync refMsg* getNotElem(refMsg* impl_msg);
 */
refMsg* CProxyElement_chunk::getNotElem(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (refMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getNotElem_refMsg));
}

/* DEFS: sync void setTargetArea(doubleMsg* impl_msg);
 */
void CProxyElement_chunk::setTargetArea(doubleMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_setTargetArea_doubleMsg));
}

/* DEFS: sync void resetTargetArea(doubleMsg* impl_msg);
 */
void CProxyElement_chunk::resetTargetArea(doubleMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_resetTargetArea_doubleMsg));
}

/* DEFS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */
void CProxyElement_chunk::updateEdges(edgeUpdateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_updateEdges_edgeUpdateMsg));
}

/* DEFS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */
void CProxyElement_chunk::updateNodeCoords(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_updateNodeCoords_nodeMsg));
}

/* DEFS: void reportPos(nodeMsg* impl_msg);
 */
void CProxyElement_chunk::reportPos(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_reportPos_nodeMsg);
}

/* DEFS: threaded sync void print(void);
 */
void CProxyElement_chunk::print(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_print_void));
}

/* DEFS: threaded sync void out_print(void);
 */
void CProxyElement_chunk::out_print(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_out_print_void));
}

/* DEFS: void freshen(void);
 */
void CProxyElement_chunk::freshen(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_freshen_void);
}

/* DEFS: void deriveBorderNodes(void);
 */
void CProxyElement_chunk::deriveBorderNodes(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_deriveBorderNodes_void);
}

/* DEFS: void tweakMesh(void);
 */
void CProxyElement_chunk::tweakMesh(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_tweakMesh_void);
}

/* DEFS: void improveChunk(void);
 */
void CProxyElement_chunk::improveChunk(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_improveChunk_void);
}

/* DEFS: threaded sync void improve(void);
 */
void CProxyElement_chunk::improve(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_improve_void));
}

/* DEFS: void addNode(nodeMsg* impl_msg);
 */
void CProxyElement_chunk::addNode(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addNode_nodeMsg);
}

/* DEFS: void addEdge(edgeMsg* impl_msg);
 */
void CProxyElement_chunk::addEdge(edgeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addEdge_edgeMsg);
}

/* DEFS: void addElement(elementMsg* impl_msg);
 */
void CProxyElement_chunk::addElement(elementMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addElement_elementMsg);
}

/* DEFS: sync void removeNode(intMsg* impl_msg);
 */
void CProxyElement_chunk::removeNode(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_removeNode_intMsg));
}

/* DEFS: sync void removeEdge(intMsg* impl_msg);
 */
void CProxyElement_chunk::removeEdge(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_removeEdge_intMsg));
}

/* DEFS: sync void removeElement(intMsg* impl_msg);
 */
void CProxyElement_chunk::removeElement(intMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_removeElement_intMsg));
}

/* DEFS: sync void updateNode(updateMsg* impl_msg);
 */
void CProxyElement_chunk::updateNode(updateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  CkFreeSysMsg(ckSendSync(impl_amsg, CkIndex_chunk::__idx_updateNode_updateMsg));
}

/* DEFS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
refMsg* CProxyElement_chunk::getOpposingNode(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  return (refMsg*) (ckSendSync(impl_amsg, CkIndex_chunk::__idx_getOpposingNode_refMsg));
}
/* DEFS: chunk(CkMigrateMessage* impl_msg);
 */
 int CkIndex_chunk::__idx_chunk_CkMigrateMessage=0;
void CkIndex_chunk::_call_chunk_CkMigrateMessage(void* impl_msg,chunk * impl_obj)
{
  new (impl_obj) chunk((CkMigrateMessage*)impl_msg);
}

/* DEFS: chunk(chunkMsg* impl_msg);
 */
CkArrayID CProxy_chunk::ckNew(chunkMsg* impl_msg, const CkArrayOptions &opts)
{ 
   return ckCreateArray((CkArrayMessage *)impl_msg,CkIndex_chunk::__idx_chunk_chunkMsg,opts);
}
 int CkIndex_chunk::__idx_chunk_chunkMsg=0;
void CkIndex_chunk::_call_chunk_chunkMsg(void* impl_msg,chunk * impl_obj)
{
  new (impl_obj) chunk((chunkMsg*)impl_msg);
}

/* DEFS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
void CProxy_chunk::addRemoteEdge(remoteEdgeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_addRemoteEdge_remoteEdgeMsg);
}
 int CkIndex_chunk::__idx_addRemoteEdge_remoteEdgeMsg=0;
void CkIndex_chunk::_call_addRemoteEdge_remoteEdgeMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->addRemoteEdge((remoteEdgeMsg*)impl_msg);
}

/* DEFS: void refineElement(refineMsg* impl_msg);
 */
void CProxy_chunk::refineElement(refineMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_refineElement_refineMsg);
}
 int CkIndex_chunk::__idx_refineElement_refineMsg=0;
void CkIndex_chunk::_call_refineElement_refineMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->refineElement((refineMsg*)impl_msg);
}

/* DEFS: threaded void refiningElements(void);
 */
void CProxy_chunk::refiningElements(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_refiningElements_void);
}
 int CkIndex_chunk::__idx_refiningElements_void=0;
void CkIndex_chunk::_call_refiningElements_void(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_refiningElements_void, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_refiningElements_void(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  CkFreeSysMsg(impl_msg);
  impl_obj->refiningElements();
}

/* DEFS: void coarsenElement(coarsenMsg* impl_msg);
 */
void CProxy_chunk::coarsenElement(coarsenMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_coarsenElement_coarsenMsg);
}
 int CkIndex_chunk::__idx_coarsenElement_coarsenMsg=0;
void CkIndex_chunk::_call_coarsenElement_coarsenMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->coarsenElement((coarsenMsg*)impl_msg);
}

/* DEFS: threaded void coarseningElements(void);
 */
void CProxy_chunk::coarseningElements(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_coarseningElements_void);
}
 int CkIndex_chunk::__idx_coarseningElements_void=0;
void CkIndex_chunk::_call_coarseningElements_void(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_coarseningElements_void, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_coarseningElements_void(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  CkFreeSysMsg(impl_msg);
  impl_obj->coarseningElements();
}

/* DEFS: sync nodeMsg* getNode(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getNode_intMsg=0;
void CkIndex_chunk::_call_getNode_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getNode((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getEdge_collapseMsg=0;
void CkIndex_chunk::_call_getEdge_collapseMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getEdge((collapseMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void setBorder(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_setBorder_intMsg=0;
void CkIndex_chunk::_call_setBorder_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->setBorder((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_safeToMoveNode_nodeMsg=0;
void CkIndex_chunk::_call_safeToMoveNode_nodeMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->safeToMoveNode((nodeMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_split_splitInMsg=0;
void CkIndex_chunk::_call_split_splitInMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->split((splitInMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void collapseHelp(collapseMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_collapseHelp_collapseMsg=0;
void CkIndex_chunk::_call_collapseHelp_collapseMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->collapseHelp((collapseMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: void checkPending(refMsg* impl_msg);
 */
void CProxy_chunk::checkPending(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_checkPending_refMsg);
}
 int CkIndex_chunk::__idx_checkPending_refMsg=0;
void CkIndex_chunk::_call_checkPending_refMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->checkPending((refMsg*)impl_msg);
}

/* DEFS: void checkPending(drefMsg* impl_msg);
 */
void CProxy_chunk::checkPending(drefMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_checkPending_drefMsg);
}
 int CkIndex_chunk::__idx_checkPending_drefMsg=0;
void CkIndex_chunk::_call_checkPending_drefMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->checkPending((drefMsg*)impl_msg);
}

/* DEFS: sync void updateElement(updateMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_updateElement_updateMsg=0;
void CkIndex_chunk::_call_updateElement_updateMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->updateElement((updateMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void updateElementEdge(updateMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_updateElementEdge_updateMsg=0;
void CkIndex_chunk::_call_updateElementEdge_updateMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->updateElementEdge((updateMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: void updateReferences(updateMsg* impl_msg);
 */
void CProxy_chunk::updateReferences(updateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_updateReferences_updateMsg);
}
 int CkIndex_chunk::__idx_updateReferences_updateMsg=0;
void CkIndex_chunk::_call_updateReferences_updateMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->updateReferences((updateMsg*)impl_msg);
}

/* DEFS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getArea_intMsg=0;
void CkIndex_chunk::_call_getArea_intMsg(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_getArea_intMsg, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_getArea_intMsg(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getArea((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_midpoint_intMsg=0;
void CkIndex_chunk::_call_midpoint_intMsg(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_midpoint_intMsg, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_midpoint_intMsg(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->midpoint((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync intMsg* setPending(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_setPending_intMsg=0;
void CkIndex_chunk::_call_setPending_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->setPending((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void unsetPending(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_unsetPending_intMsg=0;
void CkIndex_chunk::_call_unsetPending_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->unsetPending((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync intMsg* isPending(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_isPending_intMsg=0;
void CkIndex_chunk::_call_isPending_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->isPending((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync intMsg* lockNode(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_lockNode_intMsg=0;
void CkIndex_chunk::_call_lockNode_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->lockNode((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void unlockNode(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_unlockNode_intMsg=0;
void CkIndex_chunk::_call_unlockNode_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->unlockNode((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_isLongestEdge_refMsg=0;
void CkIndex_chunk::_call_isLongestEdge_refMsg(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_isLongestEdge_refMsg, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_isLongestEdge_refMsg(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->isLongestEdge((refMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getNeighbor_refMsg=0;
void CkIndex_chunk::_call_getNeighbor_refMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getNeighbor((refMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync refMsg* getNotNode(refMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getNotNode_refMsg=0;
void CkIndex_chunk::_call_getNotNode_refMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getNotNode((refMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync refMsg* getNotElem(refMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getNotElem_refMsg=0;
void CkIndex_chunk::_call_getNotElem_refMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getNotElem((refMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void setTargetArea(doubleMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_setTargetArea_doubleMsg=0;
void CkIndex_chunk::_call_setTargetArea_doubleMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->setTargetArea((doubleMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void resetTargetArea(doubleMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_resetTargetArea_doubleMsg=0;
void CkIndex_chunk::_call_resetTargetArea_doubleMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->resetTargetArea((doubleMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_updateEdges_edgeUpdateMsg=0;
void CkIndex_chunk::_call_updateEdges_edgeUpdateMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->updateEdges((edgeUpdateMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_updateNodeCoords_nodeMsg=0;
void CkIndex_chunk::_call_updateNodeCoords_nodeMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->updateNodeCoords((nodeMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: void reportPos(nodeMsg* impl_msg);
 */
void CProxy_chunk::reportPos(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_reportPos_nodeMsg);
}
 int CkIndex_chunk::__idx_reportPos_nodeMsg=0;
void CkIndex_chunk::_call_reportPos_nodeMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->reportPos((nodeMsg*)impl_msg);
}

/* DEFS: threaded sync void print(void);
 */
 int CkIndex_chunk::__idx_print_void=0;
void CkIndex_chunk::_call_print_void(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_print_void, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_print_void(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  CkFreeSysMsg(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->print();
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: threaded sync void out_print(void);
 */
 int CkIndex_chunk::__idx_out_print_void=0;
void CkIndex_chunk::_call_out_print_void(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_out_print_void, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_out_print_void(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  CkFreeSysMsg(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->out_print();
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: void freshen(void);
 */
void CProxy_chunk::freshen(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_freshen_void);
}
 int CkIndex_chunk::__idx_freshen_void=0;
void CkIndex_chunk::_call_freshen_void(void* impl_msg,chunk * impl_obj)
{
  CkFreeSysMsg(impl_msg);
  impl_obj->freshen();
}

/* DEFS: void deriveBorderNodes(void);
 */
void CProxy_chunk::deriveBorderNodes(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_deriveBorderNodes_void);
}
 int CkIndex_chunk::__idx_deriveBorderNodes_void=0;
void CkIndex_chunk::_call_deriveBorderNodes_void(void* impl_msg,chunk * impl_obj)
{
  CkFreeSysMsg(impl_msg);
  impl_obj->deriveBorderNodes();
}

/* DEFS: void tweakMesh(void);
 */
void CProxy_chunk::tweakMesh(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_tweakMesh_void);
}
 int CkIndex_chunk::__idx_tweakMesh_void=0;
void CkIndex_chunk::_call_tweakMesh_void(void* impl_msg,chunk * impl_obj)
{
  CkFreeSysMsg(impl_msg);
  impl_obj->tweakMesh();
}

/* DEFS: void improveChunk(void);
 */
void CProxy_chunk::improveChunk(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_improveChunk_void);
}
 int CkIndex_chunk::__idx_improveChunk_void=0;
void CkIndex_chunk::_call_improveChunk_void(void* impl_msg,chunk * impl_obj)
{
  CkFreeSysMsg(impl_msg);
  impl_obj->improveChunk();
}

/* DEFS: threaded sync void improve(void);
 */
 int CkIndex_chunk::__idx_improve_void=0;
void CkIndex_chunk::_call_improve_void(void* impl_msg,chunk * impl_obj)
{
  CthAwaken(CthCreate((CthVoidFn)_callthr_improve_void, new CkThrCallArg(impl_msg,impl_obj), 0));
}
void CkIndex_chunk::_callthr_improve_void(CkThrCallArg *impl_arg)
{
  void *impl_msg = impl_arg->msg;
  chunk *impl_obj = (chunk *) impl_arg->obj;
  delete impl_arg;
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  CkFreeSysMsg(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->improve();
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: void addNode(nodeMsg* impl_msg);
 */
void CProxy_chunk::addNode(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_addNode_nodeMsg);
}
 int CkIndex_chunk::__idx_addNode_nodeMsg=0;
void CkIndex_chunk::_call_addNode_nodeMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->addNode((nodeMsg*)impl_msg);
}

/* DEFS: void addEdge(edgeMsg* impl_msg);
 */
void CProxy_chunk::addEdge(edgeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_addEdge_edgeMsg);
}
 int CkIndex_chunk::__idx_addEdge_edgeMsg=0;
void CkIndex_chunk::_call_addEdge_edgeMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->addEdge((edgeMsg*)impl_msg);
}

/* DEFS: void addElement(elementMsg* impl_msg);
 */
void CProxy_chunk::addElement(elementMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_chunk::__idx_addElement_elementMsg);
}
 int CkIndex_chunk::__idx_addElement_elementMsg=0;
void CkIndex_chunk::_call_addElement_elementMsg(void* impl_msg,chunk * impl_obj)
{
  impl_obj->addElement((elementMsg*)impl_msg);
}

/* DEFS: sync void removeNode(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_removeNode_intMsg=0;
void CkIndex_chunk::_call_removeNode_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->removeNode((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void removeEdge(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_removeEdge_intMsg=0;
void CkIndex_chunk::_call_removeEdge_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->removeEdge((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void removeElement(intMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_removeElement_intMsg=0;
void CkIndex_chunk::_call_removeElement_intMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->removeElement((intMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync void updateNode(updateMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_updateNode_updateMsg=0;
void CkIndex_chunk::_call_updateNode_updateMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=CkAllocSysMsg();
    impl_obj->updateNode((updateMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}

/* DEFS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
 int CkIndex_chunk::__idx_getOpposingNode_refMsg=0;
void CkIndex_chunk::_call_getOpposingNode_refMsg(void* impl_msg,chunk * impl_obj)
{
  int impl_ref = CkGetRefNum(impl_msg), impl_src = CkGetSrcPe(impl_msg);
  void *impl_retMsg=(void *)   impl_obj->getOpposingNode((refMsg*)impl_msg);
  CkSendToFuture(impl_ref, impl_retMsg, impl_src);
}
/* DEFS: chunk(CkMigrateMessage* impl_msg);
 */

/* DEFS: chunk(chunkMsg* impl_msg);
 */

/* DEFS: void addRemoteEdge(remoteEdgeMsg* impl_msg);
 */
void CProxySection_chunk::addRemoteEdge(remoteEdgeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addRemoteEdge_remoteEdgeMsg);
}

/* DEFS: void refineElement(refineMsg* impl_msg);
 */
void CProxySection_chunk::refineElement(refineMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_refineElement_refineMsg);
}

/* DEFS: threaded void refiningElements(void);
 */
void CProxySection_chunk::refiningElements(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_refiningElements_void);
}

/* DEFS: void coarsenElement(coarsenMsg* impl_msg);
 */
void CProxySection_chunk::coarsenElement(coarsenMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_coarsenElement_coarsenMsg);
}

/* DEFS: threaded void coarseningElements(void);
 */
void CProxySection_chunk::coarseningElements(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_coarseningElements_void);
}

/* DEFS: sync nodeMsg* getNode(intMsg* impl_msg);
 */

/* DEFS: sync refMsg* getEdge(collapseMsg* impl_msg);
 */

/* DEFS: sync void setBorder(intMsg* impl_msg);
 */

/* DEFS: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
 */

/* DEFS: sync splitOutMsg* split(splitInMsg* impl_msg);
 */

/* DEFS: sync void collapseHelp(collapseMsg* impl_msg);
 */

/* DEFS: void checkPending(refMsg* impl_msg);
 */
void CProxySection_chunk::checkPending(refMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_checkPending_refMsg);
}

/* DEFS: void checkPending(drefMsg* impl_msg);
 */
void CProxySection_chunk::checkPending(drefMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_checkPending_drefMsg);
}

/* DEFS: sync void updateElement(updateMsg* impl_msg);
 */

/* DEFS: sync void updateElementEdge(updateMsg* impl_msg);
 */

/* DEFS: void updateReferences(updateMsg* impl_msg);
 */
void CProxySection_chunk::updateReferences(updateMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_updateReferences_updateMsg);
}

/* DEFS: threaded sync doubleMsg* getArea(intMsg* impl_msg);
 */

/* DEFS: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
 */

/* DEFS: sync intMsg* setPending(intMsg* impl_msg);
 */

/* DEFS: sync void unsetPending(intMsg* impl_msg);
 */

/* DEFS: sync intMsg* isPending(intMsg* impl_msg);
 */

/* DEFS: sync intMsg* lockNode(intMsg* impl_msg);
 */

/* DEFS: sync void unlockNode(intMsg* impl_msg);
 */

/* DEFS: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
 */

/* DEFS: sync refMsg* getNeighbor(refMsg* impl_msg);
 */

/* DEFS: sync refMsg* getNotNode(refMsg* impl_msg);
 */

/* DEFS: sync refMsg* getNotElem(refMsg* impl_msg);
 */

/* DEFS: sync void setTargetArea(doubleMsg* impl_msg);
 */

/* DEFS: sync void resetTargetArea(doubleMsg* impl_msg);
 */

/* DEFS: sync void updateEdges(edgeUpdateMsg* impl_msg);
 */

/* DEFS: sync void updateNodeCoords(nodeMsg* impl_msg);
 */

/* DEFS: void reportPos(nodeMsg* impl_msg);
 */
void CProxySection_chunk::reportPos(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_reportPos_nodeMsg);
}

/* DEFS: threaded sync void print(void);
 */

/* DEFS: threaded sync void out_print(void);
 */

/* DEFS: void freshen(void);
 */
void CProxySection_chunk::freshen(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_freshen_void);
}

/* DEFS: void deriveBorderNodes(void);
 */
void CProxySection_chunk::deriveBorderNodes(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_deriveBorderNodes_void);
}

/* DEFS: void tweakMesh(void);
 */
void CProxySection_chunk::tweakMesh(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_tweakMesh_void);
}

/* DEFS: void improveChunk(void);
 */
void CProxySection_chunk::improveChunk(void) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_improveChunk_void);
}

/* DEFS: threaded sync void improve(void);
 */

/* DEFS: void addNode(nodeMsg* impl_msg);
 */
void CProxySection_chunk::addNode(nodeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addNode_nodeMsg);
}

/* DEFS: void addEdge(edgeMsg* impl_msg);
 */
void CProxySection_chunk::addEdge(edgeMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addEdge_edgeMsg);
}

/* DEFS: void addElement(elementMsg* impl_msg);
 */
void CProxySection_chunk::addElement(elementMsg* impl_msg) 
{
  ckCheck();
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_chunk::__idx_addElement_elementMsg);
}

/* DEFS: sync void removeNode(intMsg* impl_msg);
 */

/* DEFS: sync void removeEdge(intMsg* impl_msg);
 */

/* DEFS: sync void removeElement(intMsg* impl_msg);
 */

/* DEFS: sync void updateNode(updateMsg* impl_msg);
 */

/* DEFS: sync refMsg* getOpposingNode(refMsg* impl_msg);
 */
#endif /*CK_TEMPLATES_ONLY*/
#ifndef CK_TEMPLATES_ONLY
void CkIndex_chunk::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size);
  CkRegisterBase(__idx, CkIndex_ArrayElement::__idx);
// REG: chunk(CkMigrateMessage* impl_msg);
  __idx_chunk_CkMigrateMessage = CkRegisterEp("chunk(CkMigrateMessage* impl_msg)",
     (CkCallFnPtr)_call_chunk_CkMigrateMessage, 0, __idx, 0);
  CkRegisterMigCtor(__idx, __idx_chunk_CkMigrateMessage);

// REG: chunk(chunkMsg* impl_msg);
  __idx_chunk_chunkMsg = CkRegisterEp("chunk(chunkMsg* impl_msg)",
     (CkCallFnPtr)_call_chunk_chunkMsg, CMessage_chunkMsg::__idx, __idx, 0);

// REG: void addRemoteEdge(remoteEdgeMsg* impl_msg);
  __idx_addRemoteEdge_remoteEdgeMsg = CkRegisterEp("addRemoteEdge(remoteEdgeMsg* impl_msg)",
     (CkCallFnPtr)_call_addRemoteEdge_remoteEdgeMsg, CMessage_remoteEdgeMsg::__idx, __idx, 0);

// REG: void refineElement(refineMsg* impl_msg);
  __idx_refineElement_refineMsg = CkRegisterEp("refineElement(refineMsg* impl_msg)",
     (CkCallFnPtr)_call_refineElement_refineMsg, CMessage_refineMsg::__idx, __idx, 0);

// REG: threaded void refiningElements(void);
  __idx_refiningElements_void = CkRegisterEp("refiningElements(void)",
     (CkCallFnPtr)_call_refiningElements_void, 0, __idx, 0);

// REG: void coarsenElement(coarsenMsg* impl_msg);
  __idx_coarsenElement_coarsenMsg = CkRegisterEp("coarsenElement(coarsenMsg* impl_msg)",
     (CkCallFnPtr)_call_coarsenElement_coarsenMsg, CMessage_coarsenMsg::__idx, __idx, 0);

// REG: threaded void coarseningElements(void);
  __idx_coarseningElements_void = CkRegisterEp("coarseningElements(void)",
     (CkCallFnPtr)_call_coarseningElements_void, 0, __idx, 0);

// REG: sync nodeMsg* getNode(intMsg* impl_msg);
  __idx_getNode_intMsg = CkRegisterEp("getNode(intMsg* impl_msg)",
     (CkCallFnPtr)_call_getNode_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync refMsg* getEdge(collapseMsg* impl_msg);
  __idx_getEdge_collapseMsg = CkRegisterEp("getEdge(collapseMsg* impl_msg)",
     (CkCallFnPtr)_call_getEdge_collapseMsg, CMessage_collapseMsg::__idx, __idx, 0);

// REG: sync void setBorder(intMsg* impl_msg);
  __idx_setBorder_intMsg = CkRegisterEp("setBorder(intMsg* impl_msg)",
     (CkCallFnPtr)_call_setBorder_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
  __idx_safeToMoveNode_nodeMsg = CkRegisterEp("safeToMoveNode(nodeMsg* impl_msg)",
     (CkCallFnPtr)_call_safeToMoveNode_nodeMsg, CMessage_nodeMsg::__idx, __idx, 0);

// REG: sync splitOutMsg* split(splitInMsg* impl_msg);
  __idx_split_splitInMsg = CkRegisterEp("split(splitInMsg* impl_msg)",
     (CkCallFnPtr)_call_split_splitInMsg, CMessage_splitInMsg::__idx, __idx, 0);

// REG: sync void collapseHelp(collapseMsg* impl_msg);
  __idx_collapseHelp_collapseMsg = CkRegisterEp("collapseHelp(collapseMsg* impl_msg)",
     (CkCallFnPtr)_call_collapseHelp_collapseMsg, CMessage_collapseMsg::__idx, __idx, 0);

// REG: void checkPending(refMsg* impl_msg);
  __idx_checkPending_refMsg = CkRegisterEp("checkPending(refMsg* impl_msg)",
     (CkCallFnPtr)_call_checkPending_refMsg, CMessage_refMsg::__idx, __idx, 0);

// REG: void checkPending(drefMsg* impl_msg);
  __idx_checkPending_drefMsg = CkRegisterEp("checkPending(drefMsg* impl_msg)",
     (CkCallFnPtr)_call_checkPending_drefMsg, CMessage_drefMsg::__idx, __idx, 0);

// REG: sync void updateElement(updateMsg* impl_msg);
  __idx_updateElement_updateMsg = CkRegisterEp("updateElement(updateMsg* impl_msg)",
     (CkCallFnPtr)_call_updateElement_updateMsg, CMessage_updateMsg::__idx, __idx, 0);

// REG: sync void updateElementEdge(updateMsg* impl_msg);
  __idx_updateElementEdge_updateMsg = CkRegisterEp("updateElementEdge(updateMsg* impl_msg)",
     (CkCallFnPtr)_call_updateElementEdge_updateMsg, CMessage_updateMsg::__idx, __idx, 0);

// REG: void updateReferences(updateMsg* impl_msg);
  __idx_updateReferences_updateMsg = CkRegisterEp("updateReferences(updateMsg* impl_msg)",
     (CkCallFnPtr)_call_updateReferences_updateMsg, CMessage_updateMsg::__idx, __idx, 0);

// REG: threaded sync doubleMsg* getArea(intMsg* impl_msg);
  __idx_getArea_intMsg = CkRegisterEp("getArea(intMsg* impl_msg)",
     (CkCallFnPtr)_call_getArea_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: threaded sync nodeMsg* midpoint(intMsg* impl_msg);
  __idx_midpoint_intMsg = CkRegisterEp("midpoint(intMsg* impl_msg)",
     (CkCallFnPtr)_call_midpoint_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync intMsg* setPending(intMsg* impl_msg);
  __idx_setPending_intMsg = CkRegisterEp("setPending(intMsg* impl_msg)",
     (CkCallFnPtr)_call_setPending_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync void unsetPending(intMsg* impl_msg);
  __idx_unsetPending_intMsg = CkRegisterEp("unsetPending(intMsg* impl_msg)",
     (CkCallFnPtr)_call_unsetPending_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync intMsg* isPending(intMsg* impl_msg);
  __idx_isPending_intMsg = CkRegisterEp("isPending(intMsg* impl_msg)",
     (CkCallFnPtr)_call_isPending_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync intMsg* lockNode(intMsg* impl_msg);
  __idx_lockNode_intMsg = CkRegisterEp("lockNode(intMsg* impl_msg)",
     (CkCallFnPtr)_call_lockNode_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync void unlockNode(intMsg* impl_msg);
  __idx_unlockNode_intMsg = CkRegisterEp("unlockNode(intMsg* impl_msg)",
     (CkCallFnPtr)_call_unlockNode_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
  __idx_isLongestEdge_refMsg = CkRegisterEp("isLongestEdge(refMsg* impl_msg)",
     (CkCallFnPtr)_call_isLongestEdge_refMsg, CMessage_refMsg::__idx, __idx, 0);

// REG: sync refMsg* getNeighbor(refMsg* impl_msg);
  __idx_getNeighbor_refMsg = CkRegisterEp("getNeighbor(refMsg* impl_msg)",
     (CkCallFnPtr)_call_getNeighbor_refMsg, CMessage_refMsg::__idx, __idx, 0);

// REG: sync refMsg* getNotNode(refMsg* impl_msg);
  __idx_getNotNode_refMsg = CkRegisterEp("getNotNode(refMsg* impl_msg)",
     (CkCallFnPtr)_call_getNotNode_refMsg, CMessage_refMsg::__idx, __idx, 0);

// REG: sync refMsg* getNotElem(refMsg* impl_msg);
  __idx_getNotElem_refMsg = CkRegisterEp("getNotElem(refMsg* impl_msg)",
     (CkCallFnPtr)_call_getNotElem_refMsg, CMessage_refMsg::__idx, __idx, 0);

// REG: sync void setTargetArea(doubleMsg* impl_msg);
  __idx_setTargetArea_doubleMsg = CkRegisterEp("setTargetArea(doubleMsg* impl_msg)",
     (CkCallFnPtr)_call_setTargetArea_doubleMsg, CMessage_doubleMsg::__idx, __idx, 0);

// REG: sync void resetTargetArea(doubleMsg* impl_msg);
  __idx_resetTargetArea_doubleMsg = CkRegisterEp("resetTargetArea(doubleMsg* impl_msg)",
     (CkCallFnPtr)_call_resetTargetArea_doubleMsg, CMessage_doubleMsg::__idx, __idx, 0);

// REG: sync void updateEdges(edgeUpdateMsg* impl_msg);
  __idx_updateEdges_edgeUpdateMsg = CkRegisterEp("updateEdges(edgeUpdateMsg* impl_msg)",
     (CkCallFnPtr)_call_updateEdges_edgeUpdateMsg, CMessage_edgeUpdateMsg::__idx, __idx, 0);

// REG: sync void updateNodeCoords(nodeMsg* impl_msg);
  __idx_updateNodeCoords_nodeMsg = CkRegisterEp("updateNodeCoords(nodeMsg* impl_msg)",
     (CkCallFnPtr)_call_updateNodeCoords_nodeMsg, CMessage_nodeMsg::__idx, __idx, 0);

// REG: void reportPos(nodeMsg* impl_msg);
  __idx_reportPos_nodeMsg = CkRegisterEp("reportPos(nodeMsg* impl_msg)",
     (CkCallFnPtr)_call_reportPos_nodeMsg, CMessage_nodeMsg::__idx, __idx, 0);

// REG: threaded sync void print(void);
  __idx_print_void = CkRegisterEp("print(void)",
     (CkCallFnPtr)_call_print_void, 0, __idx, 0);

// REG: threaded sync void out_print(void);
  __idx_out_print_void = CkRegisterEp("out_print(void)",
     (CkCallFnPtr)_call_out_print_void, 0, __idx, 0);

// REG: void freshen(void);
  __idx_freshen_void = CkRegisterEp("freshen(void)",
     (CkCallFnPtr)_call_freshen_void, 0, __idx, 0);

// REG: void deriveBorderNodes(void);
  __idx_deriveBorderNodes_void = CkRegisterEp("deriveBorderNodes(void)",
     (CkCallFnPtr)_call_deriveBorderNodes_void, 0, __idx, 0);

// REG: void tweakMesh(void);
  __idx_tweakMesh_void = CkRegisterEp("tweakMesh(void)",
     (CkCallFnPtr)_call_tweakMesh_void, 0, __idx, 0);

// REG: void improveChunk(void);
  __idx_improveChunk_void = CkRegisterEp("improveChunk(void)",
     (CkCallFnPtr)_call_improveChunk_void, 0, __idx, 0);

// REG: threaded sync void improve(void);
  __idx_improve_void = CkRegisterEp("improve(void)",
     (CkCallFnPtr)_call_improve_void, 0, __idx, 0);

// REG: void addNode(nodeMsg* impl_msg);
  __idx_addNode_nodeMsg = CkRegisterEp("addNode(nodeMsg* impl_msg)",
     (CkCallFnPtr)_call_addNode_nodeMsg, CMessage_nodeMsg::__idx, __idx, 0);

// REG: void addEdge(edgeMsg* impl_msg);
  __idx_addEdge_edgeMsg = CkRegisterEp("addEdge(edgeMsg* impl_msg)",
     (CkCallFnPtr)_call_addEdge_edgeMsg, CMessage_edgeMsg::__idx, __idx, 0);

// REG: void addElement(elementMsg* impl_msg);
  __idx_addElement_elementMsg = CkRegisterEp("addElement(elementMsg* impl_msg)",
     (CkCallFnPtr)_call_addElement_elementMsg, CMessage_elementMsg::__idx, __idx, 0);

// REG: sync void removeNode(intMsg* impl_msg);
  __idx_removeNode_intMsg = CkRegisterEp("removeNode(intMsg* impl_msg)",
     (CkCallFnPtr)_call_removeNode_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync void removeEdge(intMsg* impl_msg);
  __idx_removeEdge_intMsg = CkRegisterEp("removeEdge(intMsg* impl_msg)",
     (CkCallFnPtr)_call_removeEdge_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync void removeElement(intMsg* impl_msg);
  __idx_removeElement_intMsg = CkRegisterEp("removeElement(intMsg* impl_msg)",
     (CkCallFnPtr)_call_removeElement_intMsg, CMessage_intMsg::__idx, __idx, 0);

// REG: sync void updateNode(updateMsg* impl_msg);
  __idx_updateNode_updateMsg = CkRegisterEp("updateNode(updateMsg* impl_msg)",
     (CkCallFnPtr)_call_updateNode_updateMsg, CMessage_updateMsg::__idx, __idx, 0);

// REG: sync refMsg* getOpposingNode(refMsg* impl_msg);
  __idx_getOpposingNode_refMsg = CkRegisterEp("getOpposingNode(refMsg* impl_msg)",
     (CkCallFnPtr)_call_getOpposingNode_refMsg, CMessage_refMsg::__idx, __idx, 0);
}
#endif

#ifndef CK_TEMPLATES_ONLY
void _registerrefine(void)
{
  static int _done = 0; if(_done) return; _done = 1;
/* REG: message chunkMsg;
*/
CMessage_chunkMsg::__register("chunkMsg", sizeof(chunkMsg),(CkPackFnPtr) chunkMsg::pack,(CkUnpackFnPtr) chunkMsg::unpack);

/* REG: message nodeMsg;
*/
CMessage_nodeMsg::__register("nodeMsg", sizeof(nodeMsg),(CkPackFnPtr) nodeMsg::pack,(CkUnpackFnPtr) nodeMsg::unpack);

/* REG: message edgeMsg;
*/
CMessage_edgeMsg::__register("edgeMsg", sizeof(edgeMsg),(CkPackFnPtr) edgeMsg::pack,(CkUnpackFnPtr) edgeMsg::unpack);

/* REG: message remoteEdgeMsg;
*/
CMessage_remoteEdgeMsg::__register("remoteEdgeMsg", sizeof(remoteEdgeMsg),(CkPackFnPtr) remoteEdgeMsg::pack,(CkUnpackFnPtr) remoteEdgeMsg::unpack);

/* REG: message elementMsg;
*/
CMessage_elementMsg::__register("elementMsg", sizeof(elementMsg),(CkPackFnPtr) elementMsg::pack,(CkUnpackFnPtr) elementMsg::unpack);

/* REG: message femElementMsg;
*/
CMessage_femElementMsg::__register("femElementMsg", sizeof(femElementMsg),(CkPackFnPtr) femElementMsg::pack,(CkUnpackFnPtr) femElementMsg::unpack);

/* REG: message ghostElementMsg;
*/
CMessage_ghostElementMsg::__register("ghostElementMsg", sizeof(ghostElementMsg),(CkPackFnPtr) ghostElementMsg::pack,(CkUnpackFnPtr) ghostElementMsg::unpack);

/* REG: message refineMsg;
*/
CMessage_refineMsg::__register("refineMsg", sizeof(refineMsg),(CkPackFnPtr) refineMsg::pack,(CkUnpackFnPtr) refineMsg::unpack);

/* REG: message coarsenMsg;
*/
CMessage_coarsenMsg::__register("coarsenMsg", sizeof(coarsenMsg),(CkPackFnPtr) coarsenMsg::pack,(CkUnpackFnPtr) coarsenMsg::unpack);

/* REG: message collapseMsg;
*/
CMessage_collapseMsg::__register("collapseMsg", sizeof(collapseMsg),(CkPackFnPtr) collapseMsg::pack,(CkUnpackFnPtr) collapseMsg::unpack);

/* REG: message splitInMsg;
*/
CMessage_splitInMsg::__register("splitInMsg", sizeof(splitInMsg),(CkPackFnPtr) splitInMsg::pack,(CkUnpackFnPtr) splitInMsg::unpack);

/* REG: message splitOutMsg;
*/
CMessage_splitOutMsg::__register("splitOutMsg", sizeof(splitOutMsg),(CkPackFnPtr) splitOutMsg::pack,(CkUnpackFnPtr) splitOutMsg::unpack);

/* REG: message updateMsg;
*/
CMessage_updateMsg::__register("updateMsg", sizeof(updateMsg),(CkPackFnPtr) updateMsg::pack,(CkUnpackFnPtr) updateMsg::unpack);

/* REG: message specialRequestMsg;
*/
CMessage_specialRequestMsg::__register("specialRequestMsg", sizeof(specialRequestMsg),(CkPackFnPtr) specialRequestMsg::pack,(CkUnpackFnPtr) specialRequestMsg::unpack);

/* REG: message specialResponseMsg;
*/
CMessage_specialResponseMsg::__register("specialResponseMsg", sizeof(specialResponseMsg),(CkPackFnPtr) specialResponseMsg::pack,(CkUnpackFnPtr) specialResponseMsg::unpack);

/* REG: message refMsg;
*/
CMessage_refMsg::__register("refMsg", sizeof(refMsg),(CkPackFnPtr) refMsg::pack,(CkUnpackFnPtr) refMsg::unpack);

/* REG: message drefMsg;
*/
CMessage_drefMsg::__register("drefMsg", sizeof(drefMsg),(CkPackFnPtr) drefMsg::pack,(CkUnpackFnPtr) drefMsg::unpack);

/* REG: message edgeUpdateMsg;
*/
CMessage_edgeUpdateMsg::__register("edgeUpdateMsg", sizeof(edgeUpdateMsg),(CkPackFnPtr) edgeUpdateMsg::pack,(CkUnpackFnPtr) edgeUpdateMsg::unpack);

/* REG: message intMsg;
*/
CMessage_intMsg::__register("intMsg", sizeof(intMsg),(CkPackFnPtr) intMsg::pack,(CkUnpackFnPtr) intMsg::unpack);

/* REG: message doubleMsg;
*/
CMessage_doubleMsg::__register("doubleMsg", sizeof(doubleMsg),(CkPackFnPtr) doubleMsg::pack,(CkUnpackFnPtr) doubleMsg::unpack);

  CkRegisterReadonly("mesh","CProxy_chunk",sizeof(mesh),(void *) &mesh,__xlater_roPup_mesh);

      _registerInitCall(refineChunkInit,0);

/* REG: array chunk: ArrayElement{
chunk(CkMigrateMessage* impl_msg);
chunk(chunkMsg* impl_msg);
void addRemoteEdge(remoteEdgeMsg* impl_msg);
void refineElement(refineMsg* impl_msg);
threaded void refiningElements(void);
void coarsenElement(coarsenMsg* impl_msg);
threaded void coarseningElements(void);
sync nodeMsg* getNode(intMsg* impl_msg);
sync refMsg* getEdge(collapseMsg* impl_msg);
sync void setBorder(intMsg* impl_msg);
sync intMsg* safeToMoveNode(nodeMsg* impl_msg);
sync splitOutMsg* split(splitInMsg* impl_msg);
sync void collapseHelp(collapseMsg* impl_msg);
void checkPending(refMsg* impl_msg);
void checkPending(drefMsg* impl_msg);
sync void updateElement(updateMsg* impl_msg);
sync void updateElementEdge(updateMsg* impl_msg);
void updateReferences(updateMsg* impl_msg);
threaded sync doubleMsg* getArea(intMsg* impl_msg);
threaded sync nodeMsg* midpoint(intMsg* impl_msg);
sync intMsg* setPending(intMsg* impl_msg);
sync void unsetPending(intMsg* impl_msg);
sync intMsg* isPending(intMsg* impl_msg);
sync intMsg* lockNode(intMsg* impl_msg);
sync void unlockNode(intMsg* impl_msg);
threaded sync intMsg* isLongestEdge(refMsg* impl_msg);
sync refMsg* getNeighbor(refMsg* impl_msg);
sync refMsg* getNotNode(refMsg* impl_msg);
sync refMsg* getNotElem(refMsg* impl_msg);
sync void setTargetArea(doubleMsg* impl_msg);
sync void resetTargetArea(doubleMsg* impl_msg);
sync void updateEdges(edgeUpdateMsg* impl_msg);
sync void updateNodeCoords(nodeMsg* impl_msg);
void reportPos(nodeMsg* impl_msg);
threaded sync void print(void);
threaded sync void out_print(void);
void freshen(void);
void deriveBorderNodes(void);
void tweakMesh(void);
void improveChunk(void);
threaded sync void improve(void);
void addNode(nodeMsg* impl_msg);
void addEdge(edgeMsg* impl_msg);
void addElement(elementMsg* impl_msg);
sync void removeNode(intMsg* impl_msg);
sync void removeEdge(intMsg* impl_msg);
sync void removeElement(intMsg* impl_msg);
sync void updateNode(updateMsg* impl_msg);
sync refMsg* getOpposingNode(refMsg* impl_msg);
};
*/
  CkIndex_chunk::__register("chunk", sizeof(chunk));

}
#endif
