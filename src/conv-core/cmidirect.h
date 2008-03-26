#ifndef _CMIDIRECT_H_
#define _CMIDIRECT_H_
/* This file provides an interface for users to the CmiDirect functionality.

*/


/* handle type definition */
struct infiDirectUserHandle{
	int handle;
	int senderNode;
	int recverNode;
	void *recverBuf;
	int recverBufSize;
	char recverKey[32];
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
#ifdef __cplusplus
}
#endif

#endif
