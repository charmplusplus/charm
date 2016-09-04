/*
 * Charm Onesided API Utility Functions
 */

#ifndef _CKRDMA_H_
#define _CKRDMA_H_

#include "envelope.h"

/* CK_MSG_RDMA is passed in as entry method opts in the generated code for an entry
 * method containing RDMA parameters. In the SMP mode with IMMEDIATE message support,
 * it is used to mark the entry method invocation as IMMEDIATE to have the comm thread
 * handle the metadata message. In all other cases (Non-SMP mode, No comm thread support),
 * its value is used as 0.
 */

#if CMK_ONESIDED_IMPL && CMK_SMP && CK_MSG_IMMEDIATE
#define CK_MSG_RDMA CK_MSG_IMMEDIATE
#else
#define CK_MSG_RDMA 0
#endif

#if CMK_ONESIDED_IMPL

/* Sender Functions */

//Prepare metadata message with the relevant machine specific info
void CkRdmaPrepareMsg(envelope **env, int pe);

//Create a new message with machine specific information
envelope* CkRdmaCreateMetadataMsg(envelope *env, int pe);

//Handle ack received on the sender by invoking callback
void CkHandleRdmaCookie(void *cookie);



/* Receiver Functions */

//Copy the message using pointers when it's on the same PE/node
envelope* CkRdmaCopyMsg(envelope *env);

/*
 * Extract rdma based information from the metadata message,
 * allocate buffers and issue RDMA get call
 */
void CkRdmaIssueRgets(envelope *env, bool free);

/*
 * Method called to update machine specific information and pointers
 * inside Ckrdmawrappers
 */
void CkUpdateRdmaPtrs(envelope *msg, int msgsize, char *recv_md, char *src_md);

/*
 * Method called to pack rdma pointers
 * inside Ckrdmawrappers
 */
void CkPackRdmaPtrs(char *msgBuf);

/*
 * Method called to unpack rdma pointers
 * inside Ckrdmawrappers
 */
void CkUnpackRdmaPtrs(char *msgBuf);

//Get the number of rdma ops using the metadata message
int getRdmaNumOps(envelope *env);

//Get the sum of rdma buffer sizes using the metadata message
int getRdmaBufSize(envelope *env);

CkRdmaPostHandle* createRdmaPostHandle(int numops);

CkRdmaPostHandle* CkGetRdmaPostHandle(envelope *env);

void CkRdmaPost(CkRdmaPostHandle *handle);

void CkUpdateRdmaPtrsPost(envelope *env, int msgsize, char *recv_md, char *src_md, CkRdmaPostHandle* handle);

void CkUpdateRdmaPtrsPost(envelope *env, CkRdmaPostHandle* handle);


#endif
#endif
