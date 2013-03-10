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
#elif  CMK_CONVERSE_UGNI
#include "gni_pub.h"
#endif
typedef struct infiDirectUserHandle{
    int handle;
#ifdef CMK_BLUEGENEP
    int senderNode;
    int recverNode;
    void *recverBuf;
    int recverBufSize;
    void *senderBuf;
    void (*callbackFnPtr)(void *);
    void *callbackData;
    void *DCMF_notify_buf;
    DCMF_Request_t *DCMF_rq_trecv;
    DCMF_Request_t *DCMF_rq_tsend;
    DCMF_Memregion_t DCMF_recverMemregion;
    DCMF_Memregion_t DCMF_senderMemregion;
    DCMF_Callback_t DCMF_notify_cb;
#elif  CMK_CONVERSE_UGNI
    int localNode;
    int remoteRank;
    int remoteNode;
    void *remoteBuf;
    void *remoteHandler;
    int transSize;
    void *localBuf;
    void (*callbackFnPtr)(void *);
    void *callbackData;
    gni_mem_handle_t    localMdh;
    gni_mem_handle_t    remoteMdh;
    int ack_index;
#else
    int senderNode;
    int recverNode;
    void *recverBuf;
    int recverBufSize;
	char recverKey[64];
#endif
	double initialValue;
}CmiDirectUserHandle;

#ifdef  CMK_CONVERSE_UGNI
typedef gni_mem_handle_t    CmiDirectMemoryHandler;
CmiDirectMemoryHandler CmiDirect_registerMemory(void *buff, int size);
struct infiDirectUserHandle CmiDirect_createHandle_mem(CmiDirectMemoryHandler *mem_hndl, void *recvBuf, int recvBufSize, void (*callbackFnPtr)(void *), void *callbackData);
void CmiDirect_assocLocalBuffer_mem(struct infiDirectUserHandle *userHandle, CmiDirectMemoryHandler *mem_hndl, void *sendBuf,int sendBufSize);
void CmiDirect_saveHandler(CmiDirectUserHandle* h, void *ptr);
#endif
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
