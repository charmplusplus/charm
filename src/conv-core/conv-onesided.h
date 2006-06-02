/*
 * This is a one sided communication model to support 
 * Get/Put like communiation in converse
 * Author: Nilesh
 * Date: 05/17/2006
 */
#include "converse.h"
#ifdef __ONESIDED_IMPL

#ifdef __ONESIDED_GM_HARDWARE
void *CmiDMAAlloc(int size);
#endif

int CmiRegisterMemory(void *addr, unsigned int size);
int CmiUnRegisterMemory(void *addr, unsigned int size);

/* Machine layer functions, must be provided by the machine layer 
 * which wants to provide one sided communication operations
 */
void *CmiPut(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size);
void *CmiGet(unsigned int sourceId, unsigned int targetId, void *Saddr, void *Taddr, unsigned int size);
int CmiWaitTest(void *obj);

#endif

