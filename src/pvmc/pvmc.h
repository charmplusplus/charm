#ifndef __PVMC_H__
#define __PVMC_H__

#include "converse.h"
#include "pvm3.h"

#define PRINTF		CmiPrintf
#define MALLOC(x)	CmiAlloc(x)
#define FREE(x)		CmiFree(x)
#define MYPE()		CmiMyPe()
#define NUMPES()	CmiMyPe()
#define TID2PE(x)       ((x)-1)
#define PE2TID(x)       ((x)+1)

#define PVMC_CTRL_AT_BARRIER		1
#define PVMC_CTRL_THROUGH_BARRIER	2
#define PVMC_CTRL_KILL			3

#ifndef FALSE
#define FALSE	0
#endif

#ifndef TRUE
#define TRUE	1
#endif

#endif
