/*
 Interface to Converse portion of parallel debugger.
 Moved here from converse.h 4/10/2001 by Orion Sky Lawlor, olawlor@acm.org
 */
#ifndef __CMK_DEBUG_CONV_H
#define __CMK_DEBUG_CONV_H

#include "pup_c.h"

#ifdef __cplusplus
extern "C" {
#endif

void CpdInit(void); 
void CpdFreeze(void);  
void CpdUnFreeze(void);
void CpdFreezeModeScheduler(void);
void Cpd_CmiHandleMessage(void *msg);

/* C bindings for CpdList functions: */
typedef int  (*CpdListLengthFn_c)(void *lenParam);

typedef struct {
	int lo,hi; /*Range of requested items in list is (lo .. hi-1)*/
	int extraLen; /*Amount of data pointed to below*/
	void *extra; /*List-defined request data*/
} CpdListItemsRequest;
void CpdListBeginItem(pup_er p,int itemNo);
typedef void (*CpdListItemsFn_c)(void *itemsParam,pup_er p,
				CpdListItemsRequest *req);

void CpdListRegister_c(const char *path,
	    CpdListLengthFn_c len,void *lenParam,
	    CpdListItemsFn_c items,void *itemsParam);

#ifdef __cplusplus
};
#endif

#endif
