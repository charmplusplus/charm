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
    /*DCMF_Request_t *DCMF_rq_t;*/
    void  *DCMF_rq_trecv;
#endif
#ifdef CMK_BLUEGENEP
	void *DCMF_rq_tsend;
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

/****
To be called on the sender to do the actual data transfer
******/
void CmiDirect_put(struct infiDirectUserHandle *userHandle);

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
