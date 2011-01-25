#ifndef _CMIDIRECT_H_
#define _CMIDIRECT_H_
/* This file provides an interface for users to the CmiDirect functionality.

*/
#ifdef CMK_BLUEGENEP
typedef struct {
    CmiFloat8 space[2];
} cmkquad;  
/* is equivalent to DCQUAD, but without including dmcf.h */
#endif


/* handle type definition */
/* sender is the one who initiates the request.
   recver is the one who receives the request.
   Put: sender=source recver=target of the one-sided buffer operation
   Get: sender=target recver=source of the one-sided buffer operation
*/
#ifdef CMK_BLUEGENEP
#include "dcmf.h"
#endif
struct infiDirectUserHandle{
    int handle;
    int senderNode;
    int recverNode;
    void *recverBuf;
    int recverBufSize;
#ifdef CMK_BLUEGENEP
    void *senderBuf;
    void (*callbackFnPtr)(void *);
    void *callbackData;
    void *DCMF_notify_buf;
    DCMF_Request_t *DCMF_rq_trecv;
    DCMF_Request_t *DCMF_rq_tsend;
    DCMF_Memregion_t DCMF_recverMemregion;
    DCMF_Memregion_t DCMF_senderMemregion;
    DCMF_Callback_t DCMF_notify_cb;
#else
	char recverKey[64];
#endif
	double initialValue;
};


/* functions */

#ifdef __cplusplus
extern "C" {
#endif
/**
 To be called on the receiver to create a handle and return its number
**/
struct infiDirectUserHandle CmiDirect_createHandle(int senderNode,void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData,double initialValue);

/****
 To be called on the sender to attach the sender's buffer to this handle
******/
void CmiDirect_assocLocalBuffer(struct infiDirectUserHandle *userHandle,void *sendBuf,int sendBufSize);


/**** up to the user to safely call this */
void CmiDirect_deassocLocalBuffer(struct infiDirectUserHandle *userHandle);

/**** up to the user to safely call this */
void CmiDirect_destroyHandle(struct infiDirectUserHandle *userHandle);

/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(struct infiDirectUserHandle *userHandle);


/****
To be called on the receiver to initiate the actual data transfer
******/
void CmiDirect_get(struct infiDirectUserHandle *userHandle);

/**** Should not be called the first time *********/
void CmiDirect_ready(struct infiDirectUserHandle *userHandle);

/**** Should not be called the first time *********/
void CmiDirect_readyMark(struct infiDirectUserHandle *userHandle);

/**** Should not be called the first time *********/
void CmiDirect_readyPollQ(struct infiDirectUserHandle *userHandle);
#ifdef __cplusplus
}
#endif

#endif
