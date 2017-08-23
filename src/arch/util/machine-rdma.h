#ifndef _MACHINE_RDMA_H_
#define _MACHINE_RDMA_H_

typedef void (*RdmaSingleAckHandlerFn)(void *cbPtr, int pe, const void *ptr);
/*Function Pointer to Acknowledgement Handler*/
typedef void (*RdmaAckHandlerFn)(void *token);

/*Acknowledgement constisting of handler and token*/
typedef struct _cmi_rdma_ack{
  // Function Pointer to Acknowledgment handler function for the Indirect API
  RdmaAckHandlerFn fnPtr;
  void *token;
} CmiRdmaAck;


/*Lrts Function declarations*/

/*Sender Functions*/
void LrtsSetRdmaInfo(void *dest, int destPE, int numOps);
void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE);
int LrtsGetRdmaOpInfoSize();
int LrtsGetRdmaGenInfoSize();
int LrtsGetRdmaInfoSize(int numOps);
void LrtsSetRdmaRecvInfo(void *dest, int numOps, void *charmMsg, void *rdmaInfo, int msgSize);

/*Receiver Functions*/
void LrtsSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaRecv);
int LrtsGetRdmaOpRecvInfoSize();
int LrtsGetRdmaGenRecvInfoSize();
int LrtsGetRdmaRecvInfoSize(int numOps);
void LrtsIssueRgets(void *recv, int pe);

#if CMK_ONESIDED_DIRECT_IMPL

// Function Pointer to the individual Acknowledement handler function for the Direct API
RdmaSingleAckHandlerFn ncpyAckHandlerFn;

typedef struct _cmi_rdma_direct_ack {
  const void *srcAddr;
  int srcPe;
  const void *tgtAddr;
  int tgtPe;
  int ackSize;
} CmiRdmaDirectAck;

/* Support for Nocopy Direct API */
void LrtsSetRdmaSrcInfo(void *info, const void *ptr, int size);
void LrtsSetRdmaTgtInfo(void *info, const void *ptr, int size);
void LrtsSetRdmaNcpyAck(RdmaAckHandlerFn fn);
void LrtsIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  int size);
void LrtsIssueRput(
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size);
void LrtsReleaseSourceResource(void *info, int pe);
void LrtsReleaseTargetResource(void *info, int pe);
#endif

/* Converse Machine Interface Functions*/

/* Sender Side Functions */

/* Set the machine layer info generic to RDMA ops*/
void CmiSetRdmaInfo(void *dest, int destPE, int numOps){
  LrtsSetRdmaInfo(dest, destPE, numOps);
}

/* Set the machine layer info specific to RDMA op*/
void CmiSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE){
  LrtsSetRdmaOpInfo(dest, ptr, size, ack, destPE);
}

/* Getter for size help upper layers allocate space for machine layer info
 * while allocating the message*/

/* Get the size occupied by the machine layer info specific to RDMA op*/
int CmiGetRdmaOpInfoSize(){
  return LrtsGetRdmaOpInfoSize();
}

/* Get the size occupied by the macine layer info generic to RDMA ops*/
int CmiGetRdmaGenInfoSize(){
  return LrtsGetRdmaGenInfoSize();
}

/* Get the total size occupied by the machine layer info (specific + generic)*/
int CmiGetRdmaInfoSize(int numOps){
  return LrtsGetRdmaInfoSize(numOps);
}

/* Set the ack function handler and token*/
void *CmiSetRdmaAck(RdmaAckHandlerFn fn, void *token){
  CmiRdmaAck *ack = malloc(sizeof(CmiRdmaAck));
  ack->fnPtr = fn;
  ack->token = token;
  return ack;
}


/* Receiver side functions */

/* Set the receiver specific machine layer info generic to RDMA ops*/
void CmiSetRdmaRecvInfo(void *dest, int numOps, void *charmMsg, void *rdmaInfo, int msgSize){
  LrtsSetRdmaRecvInfo(dest, numOps, charmMsg, rdmaInfo, msgSize);
}

/* Set the receiver specific machine layer info specific to RDMA ops*/
void CmiSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaInfo){
  LrtsSetRdmaRecvOpInfo(dest, buffer, src_ref, size, opIndex, rdmaInfo);
}

/* Get the size occupied by the receiver specific machine layer specific to RDMA op*/
int CmiGetRdmaOpRecvInfoSize(){
  return LrtsGetRdmaOpRecvInfoSize();
}

/* Get the size occupied by the receiver specific machine layer info generic to RDMA ops*/
int CmiGetRdmaGenRecvInfoSize(){
  return LrtsGetRdmaGenRecvInfoSize();
}

/* Get the total size occupied by the receiver specific machine layer info*/
int CmiGetRdmaRecvInfoSize(int numOps){
  return LrtsGetRdmaRecvInfoSize(numOps);
}

/* Issue RDMA get calls on the pe using the message containing the metadata information*/
void CmiIssueRgets(void *recv, int pe){
  return LrtsIssueRgets(recv, pe);
}

#if CMK_ONESIDED_DIRECT_IMPL
/* Support for Nocopy Direct API */

/* Set the machine specific information for a nocopy source pointer */
void CmiSetRdmaSrcInfo(void *info, const void *ptr, int size){
  LrtsSetRdmaSrcInfo(info, ptr, size);
}

/* Set the machine specific information for a nocopy target pointer */
void CmiSetRdmaTgtInfo(void *info, const void *ptr, int size){
  LrtsSetRdmaTgtInfo(info, ptr, size);
}

void *CmiGetNcpyAck(const void *srcAddr, void *srcAck, int srcPe, const void *tgtAddr, void *tgtAck, int tgtPe, int ackSize) {
  CmiRdmaDirectAck *directAck = (CmiRdmaDirectAck *)malloc(sizeof(CmiRdmaDirectAck) + 2*ackSize);
  directAck->srcAddr = srcAddr;
  directAck->srcPe = srcPe;
  directAck->tgtAddr = tgtAddr;
  directAck->tgtPe = tgtPe;
  directAck->ackSize = ackSize;

  // copy source ack
  memcpy((char *)directAck + sizeof(CmiRdmaDirectAck), srcAck, ackSize);

  // copy target ack
  memcpy((char *)directAck + sizeof(CmiRdmaDirectAck) + ackSize, tgtAck, ackSize);

  return (void *)directAck;
}

void CmiInvokeNcpyAck(void *ack) {
  CmiRdmaDirectAck *directAck = (CmiRdmaDirectAck *)ack;

  // Retrieve source ack
  void *srcAck = (char *)directAck + sizeof(CmiRdmaDirectAck);

  // Retrieve target ack
  void *tgtAck = (char *)srcAck + directAck->ackSize;

  ncpyAckHandlerFn(srcAck, directAck->srcPe, directAck->srcAddr);
  ncpyAckHandlerFn(tgtAck, directAck->tgtPe, directAck->tgtAddr);

  // free the allocated ack information
  free(directAck);
}

/* Set the ack handler function used in the Direct API */
void CmiSetRdmaNcpyAck(RdmaSingleAckHandlerFn fn){
  ncpyAckHandlerFn = fn;
}

/* Perform an RDMA Get operation into the local target address from the remote source address*/
void CmiIssueRget(
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  int size) {
  LrtsIssueRget(srcAddr,
                srcInfo,
                srcAck,
                srcAckSize,
                srcPe,
                tgtAddr,
                tgtInfo,
                tgtAck,
                tgtAckSize,
                tgtPe,
                size);
}

/* Perform an RDMA Put operation into the remote target address from the local source address */
void CmiIssueRput(
  const void* tgtAddr,
  void *tgtInfo,
  void *tgtAck,
  int tgtAckSize,
  int tgtPe,
  const void* srcAddr,
  void *srcInfo,
  void *srcAck,
  int srcAckSize,
  int srcPe,
  int size) {
  LrtsIssueRput(tgtAddr,
                tgtInfo,
                tgtAck,
                tgtAckSize,
                tgtPe,
                srcAddr,
                srcInfo,
                srcAck,
                srcAckSize,
                srcPe,
                size);
}

/* Resource cleanup for source pointer */
void CmiReleaseSourceResource(void *info, int pe){
  LrtsReleaseSourceResource(info, pe);
}

/* Resource cleanup for target pointer */
void CmiReleaseTargetResource(void *info, int pe){
  LrtsReleaseTargetResource(info, pe);
}

#endif /*End of CMK_ONESIDED_DIRECT_IMPL */
#endif
