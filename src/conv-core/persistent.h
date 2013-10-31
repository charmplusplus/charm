/*****************************************************************************
                  Persistent Communication API

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
* void CmiDestroyPersistent(PersistentHandle h);
	Destroy a persistent communication specified by PersistentHandle h.
* void CmiDestroyAllPersistent();
	Destroy all persistent communication on the local processor.

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
PersistentHandle CmiCreateCompressPersistent(int destPE, int maxBytes, int start, int type);
PersistentHandle CmiCreateCompressNodePersistent(int destNode, int maxBytes, int start, int type);
PersistentHandle CmiCreateCompressPersistentSize(int destPE, int maxBytes, int start, int size, int type);
PersistentHandle CmiCreateCompressNodePersistentSize(int destNode, int maxBytes, int start, int size, int type);
PersistentReq CmiCreateReceiverPersistent(int maxBytes);
PersistentHandle CmiRegisterReceivePersistent(PersistentReq req);
void CmiUsePersistentHandle(PersistentHandle *p, int n);
void CmiDestroyPersistent(PersistentHandle h);
void CmiDestroyAllPersistent();

void CmiPersistentOneSend();
#else

typedef int PersistentRecvHandle;

#define CmiPersistentInit()
#define CmiCreatePersistent(x,y)  
#define CmiCreateNodePersistent(x,y)
#define CmiCreateCompressPersistent(x,y,z,m) 
#define CmiCreateCompressPersistentSize(x,y,z,t,m) 
#define CmiCreateCompressNodePersistent(x,y,z,m)
#define CmiCreateCompressNodePersistentSize(x,y,z,t,m)
#define CmiCreateReceiverPersistent(maxBytes)   
#define CmiRegisterReceivePersistent(req)  
#define CmiUsePersistentHandle(x,y)
#define CmiDestroyAllPersistent()

#endif

#ifdef __cplusplus
}
#endif

#endif
