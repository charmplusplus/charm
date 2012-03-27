/*****************************************************************************
                  Persistent Communication API for ELAN

* PersistentHandle: 
	persistent communication handler, created by CmiCreatePersistent()
* void CmiPersistentInit():
	initialize persistent communication module, used by converseInit.
* PersistentHandle CmiCreatePersistent(int destPE, int maxBytes):
	Sender initiates the setting up of persistent communication.
	create a persistent communication handler, with dest PE and maximum 
	bytes for this persistent communication. Machine layer will send 
	message to destPE and setup a persistent communication. a buffer 
	of size maxBytes is allocated in the destination PE.
* PersistentReq CmiCreateReceiverPersistent(int maxBytes);
  PersistentHandle CmiRegisterReceivePersistent(PersistentReq req);
	Alternatively, a receiver can initiate the setting up of persistent 
	communication.
	At receiver side, user calls CmiCreateReceiverPersistent() which 
	returns a temporary handle type - PersistentRecvHandle. Send this 
	handle to the sender side and the sender should call 
	CmiRegisterReceivePersistent() to setup the persistent communication. 
	The function returns a PersistentHandle which can then be used for 
	the following persistent communication.
* void CmiUsePersistentHandle(PersistentHandle *p, int n);
	ask Charm machine layer to use an array of PersistentHandle "p" 
	(array size of n) for all the following communication. Calling with 
	p = NULL will cancel the persistent communication. n = 1 is for 
	sending message to one Chare, n > 1 is for message in multicast - 
	one PersistentHandle for each PE.
* void CmiDestoryPersistent(PersistentHandle h);
	Destory a persistent communication specified by PersistentHandle h.
* void CmiDestoryAllPersistent();
	Destory all persistent communication on the local processor.

*****************************************************************************/

#include "conv-config.h"

#ifndef __PERSISTENT_H__
#define __PERSISTENT_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef void * PersistentHandle;

#if CMK_PERSISTENT_COMM

typedef struct {
  int pe;
  int maxBytes;
  void **bufPtr;
  PersistentHandle myHand;
} PersistentReq;

void CmiPersistentInit();
PersistentHandle CmiCreatePersistent(int destPE, int maxBytes);
PersistentHandle CmiCreateNodePersistent(int destNode, int maxBytes);
PersistentReq CmiCreateReceiverPersistent(int maxBytes);
PersistentHandle CmiRegisterReceivePersistent(PersistentReq req);
void CmiUsePersistentHandle(PersistentHandle *p, int n);
void CmiDestoryPersistent(PersistentHandle h);
void CmiDestoryAllPersistent();

void CmiPersistentOneSend();
#else

typedef int PersistentRecvHandle;

#define CmiPersistentInit()
#define CmiCreatePersistent(x,y)  0
#define CmiCreateReceiverPersistent(maxBytes)   0
#define CmiRegisterReceivePersistent(req)  0
#define CmiUsePersistentHandle(x,y)
#define CmiDestoryAllPersistent()

#endif

#ifdef __cplusplus
}
#endif

#endif
