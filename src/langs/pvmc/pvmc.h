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

void pvmc_init_bufs(void);
void pvmc_init_comm(void);
void pvmc_init_groups(void);
void pvmc_user_main(int argc, char **argv);

int pvmc_sendmsgsz(void);
int pvmc_settidtag(int pvm_tid, int tag);
int pvmc_packmsg(void *msgbuf);
int pvmc_unpackmsg(void *msgbuf, void *start_of_msg);
int pvmc_gettag(void *msgbuf);
void *pvmc_mkitem(int nbytes, int type);
void *pvmc_getitem(int n_bytes, int type);
void *pvmc_getstritem(int *n_bytes);
void pvmc_send_control_msg(int type, int pe);

#endif
