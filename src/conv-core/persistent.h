/*****************************************************************************
                  Persistent Communication API for ELAN

* PersistentHandle: 
	persistent communication handler, created by CmiCreatePersistent()
* void CmiPersistentInit():
	initialize persistent communication module, used by converseInit.
* PersistentHandle CmiCreatePersistent(int destPE, int maxBytes):
	create a persistent communication handler, with dest PE and maximum 
	bytes for this persistent communication. Machine layer will send 
	message to destPE and setup a persistent communication. a buffer 
	of size maxBytes is allocated in the destination PE.
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

typedef void * PersistentHandle;

#if CMK_PERSISTENT_COMM

void CmiPersistentInit();
PersistentHandle CmiCreatePersistent(int destPE, int maxBytes);
void CmiUsePersistentHandle(PersistentHandle *p, int n);
void CmiDestoryPersistent(PersistentHandle h);
void CmiDestoryAllPersistent();

#else

#define CmiPersistentInit()
#define CmiCreatePersistent(x,y)  0
#define CmiUsePersistentHandle(x,y)
#define CmiDestoryAllPersistent()

#endif
