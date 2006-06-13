/*
 * This is a one sided communication model to support 
 * Get/Put like communiation in converse
 * Author: Nilesh
 * Date: 05/17/2006
 */
#include "converse.h"
#ifdef __ONESIDED_IMPL

#ifndef _CONV_ONESIDED_H_
#define _CONV_ONESIDED_H_

typedef void (*CmiRdmaCallbackFn)(void *param);

#ifdef __ONESIDED_GM_HARDWARE
void *CmiDMAAlloc(int size);
#endif

int CmiRegisterMemory(void *addr, unsigned int size);
int CmiUnRegisterMemory(void *addr, unsigned int size);


/* Version of One sided communication when there is no callback,
 * so a handle is returned which needs to polled to check for completion
 * of the operation.
 */
void *CmiPut(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size);
void *CmiGet(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size);
int CmiWaitTest(void *obj);


/* Version of One sided communication when there is a callback
 * immediately when the operation finishes. So, there is no need
 * to poll for completion of the operation.
 */
void CmiPutCb(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size, CmiRdmaCallbackFn fn, void *param);
void CmiGetCb(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size, CmiRdmaCallbackFn fn, void *param);

#endif
#endif

