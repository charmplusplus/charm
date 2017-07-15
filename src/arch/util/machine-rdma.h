#ifndef _MACHINE_RDMA_H_
#define _MACHINE_RDMA_H_

/*Function Pointer to Acknowledgement Handler*/
typedef void (*RdmaAckHandlerFn)(void *token);

/*Acknowledgement constisting of handler and token*/
typedef struct _cmi_rdma_ack{
  RdmaAckHandlerFn fnPtr;
  void *token;
} CmiRdmaAck;


/*Lrts Function declarations*/

/*Sender Functions*/
void LrtsSetRdmaInfo(void *dest, int destPE, int numOps);
void LrtsSetRdmaOpInfo(void *dest, const void *ptr, int size, void *ack, int destPE);
int LrtsGetRdmaOpInfoSize(void);
int LrtsGetRdmaGenInfoSize(void);
int LrtsGetRdmaInfoSize(int numOps);
void LrtsSetRdmaRecvInfo(void *dest, int numOps, void *charmMsg, void *rdmaInfo, int msgSize);

/*Receiver Functions*/
void LrtsSetRdmaRecvOpInfo(void *dest, void *buffer, void *src_ref, int size, int opIndex, void *rdmaRecv);
int LrtsGetRdmaOpRecvInfoSize(void);
int LrtsGetRdmaGenRecvInfoSize(void);
int LrtsGetRdmaRecvInfoSize(int numOps);
void LrtsIssueRgets(void *recv, int pe);



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
int CmiGetRdmaOpInfoSize(void){
  return LrtsGetRdmaOpInfoSize();
}

/* Get the size occupied by the macine layer info generic to RDMA ops*/
int CmiGetRdmaGenInfoSize(void){
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
int CmiGetRdmaOpRecvInfoSize(void){
  return LrtsGetRdmaOpRecvInfoSize();
}

/* Get the size occupied by the receiver specific machine layer info generic to RDMA ops*/
int CmiGetRdmaGenRecvInfoSize(void){
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

#endif
