#ifndef __PVM3_H__
#define __PVM3_H__

struct pvmhostinfo {
	int hi_tid;			/* pvmd tid */
	char *hi_name;		/* host name */
	char *hi_arch;		/* host arch */
	int hi_speed;		/* cpu relative speed */
};

struct pvmtaskinfo {
	int ti_tid;				/* task id */
	int ti_ptid;			/* parent tid */
	int ti_host;			/* pvmd tid */
	int ti_flag;			/* status flags */
	char *ti_a_out;			/* a.out name */
	int ti_pid;				/* task (O/S dependent) process id */
};

#ifdef __cplusplus
extern "C" {
#endif

void pvmc_init(void);
int pvm_mkbuf(int encoding);
int pvm_freebuf(int bufid);
int pvm_getsbuf(void);
int pvm_setsbuf(int bufid);
int pvm_getrbuf(void);
int pvm_setrbuf(int bufid);
int pvm_initsend(int encoding);
int pvm_bufinfo(int bufid, int *bytes, int *msgtag, int *tid);

int pvm_send(int tid, int tag);
int pvm_mcast(int *tids, int ntask, int msgtag);
int pvm_nrecv(int tid, int tag);
int pvm_recv(int tid, int tag);
int pvm_probe(int tid, int tag);

int pvm_mytid(void);
int pvm_exit(void);
int pvm_kill(int tid);
int pvm_spawn(char *task, char **argv, int flag,
	      char *where, int ntask, int *tids);
int pvm_parent(void);
int pvm_config(int *nhost, int *narch, struct pvmhostinfo **hostp);
int pvm_tasks(int which, int *ntask, struct pvmtaskinfo **taskp);
int pvm_setopt(int what, int val);
int pvm_gsize(char *group);
int pvm_gettid(char *group, int inum);

int pvm_upkbyte(char *cp, int cnt, int std);
int pvm_upkcplx(float *xp, int cnt, int std);
int pvm_upkdcplx(double *zp, int cnt, int std);
int pvm_upkdouble(double *dp, int cnt, int std);
int pvm_upkfloat(float *fp, int cnt, int std);
int pvm_upkint(int *np, int cnt, int std);
int pvm_upklong(long *np, int cnt, int std);
int pvm_upkshort(short *np, int cnt, int std);
int pvm_upkuint(unsigned int *np, int cnt, int std);
int pvm_upkulong(unsigned long *np, int cnt, int std);
int pvm_upkushort(unsigned short *np, int cnt, int std);
int pvm_upkstr(char *cp);

int pvm_pkbyte(char *cp, int cnt, int std);
int pvm_pkcplx(float *xp, int cnt, int std);
int pvm_pkdcplx(double *zp, int cnt, int std);
int pvm_pkdouble(double *dp, int cnt, int std);
int pvm_pkfloat(float *fp, int cnt, int std);
int pvm_pkint(int *np, int cnt, int std);
int pvm_pklong(long *np, int cnt, int std);
int pvm_pkshort(short *np, int cnt, int std);
int pvm_pkuint(unsigned int *np, int cnt, int std);
int pvm_pkulong(unsigned long *np, int cnt, int std);
int pvm_pkushort(unsigned short *np, int cnt, int std);
int pvm_pkstr(char *cp);

int pvm_bcast(const char *group, int msgtag);
int pvm_joingroup(const char *group);
int pvm_lvgroup(const char *group);
int pvm_barrier(const char *group, int count);
#ifdef __cplusplus
}
#endif



/* defines from pvm3.h */

/*
*	Data packing styles for pvm_initsend()
*/

#define	PvmDataDefault	0
#define	PvmDataRaw		1
#define	PvmDataInPlace	2
#define	PvmDataFoo		3

/*
*	pvm_spawn options
*/

#define	PvmTaskDefault	0
#define	PvmTaskHost		1	/* specify host */
#define	PvmTaskArch		2	/* specify architecture */
#define	PvmTaskDebug	4	/* start task in debugger */
#define	PvmTaskTrace	8	/* process generates trace data */
/* for MPP ports */
#define	PvmMppFront		16	/* spawn task on service node */
#define	PvmHostCompl	32	/* complement host set */

/*
*	pvm_notify types
*/

#define	PvmTaskExit		1	/* on task exit */
#define	PvmHostDelete	2	/* on host fail/delete */
#define	PvmHostAdd		3	/* on host startup */

/*
*	for pvm_setopt and pvm_getopt
*/

#define	PvmRoute			1	/* routing policy */
#define		PvmDontRoute		1	/* don't allow direct task-task links */
#define		PvmAllowDirect		2	/* allow direct links, but don't request */
#define		PvmRouteDirect		3	/* request direct links */
#define	PvmDebugMask		2	/* debugmask */
#define	PvmAutoErr			3	/* auto error reporting */
#define	PvmOutputTid		4	/* stdout destination for children */
#define	PvmOutputCode		5	/* stdout message tag */
#define	PvmTraceTid			6	/* trace destination for children */
#define	PvmTraceCode		7	/* trace message tag */
#define	PvmFragSize			8	/* message fragment size */
#define	PvmResvTids			9	/* allow reserved message tids and codes */
#define	PvmSelfOutputTid	10	/* stdout destination for task */
#define	PvmSelfOutputCode	11	/* stdout message tag */
#define	PvmSelfTraceTid		12	/* trace destination for task */
#define	PvmSelfTraceCode	13	/* trace message tag */
#define	PvmShowTids			14	/* pvm_catchout prints task ids with output */
#define	PvmPollType			15	/* shared memory wait method */
#define		PvmPollConstant	1
#define		PvmPollSleep	2
#define	PvmPollTime			16	/* time before sleep if PvmPollSleep */

/*
*	for pvm_[sg]ettmask
*/

#define	PvmTaskSelf		0	/* this task */
#define	PvmTaskChild	1	/* (future) child tasks */

/*
*	Libpvm error codes
*/

#define	PvmOk			0	/* Error 0 */
#define	PvmBadParam		-2	/* Bad parameter */
#define	PvmMismatch		-3	/* Count mismatch */
#define	PvmOverflow		-4	/* Value too large */
#define	PvmNoData		-5	/* End of buffer */
#define	PvmNoHost		-6	/* No such host */
#define	PvmNoFile		-7	/* No such file */
#define	PvmNoMem		-10	/* Malloc failed */
#define	PvmBadMsg		-12	/* Can't decode message */
#define	PvmSysErr		-14	/* Can't contact local daemon */
#define	PvmNoBuf		-15	/* No current buffer */
#define	PvmNoSuchBuf	-16	/* No such buffer */
#define	PvmNullGroup	-17	/* Null group name */
#define	PvmDupGroup		-18	/* Already in group */
#define	PvmNoGroup		-19	/* No such group */
#define	PvmNotInGroup	-20	/* Not in group */
#define	PvmNoInst		-21	/* No such instance */
#define	PvmHostFail		-22	/* Host failed */
#define	PvmNoParent		-23	/* No parent task */
#define	PvmNotImpl		-24	/* Not implemented */
#define	PvmDSysErr		-25	/* Pvmd system error */
#define	PvmBadVersion	-26	/* Version mismatch */
#define	PvmOutOfRes		-27	/* Out of resources */
#define	PvmDupHost		-28	/* Duplicate host */
#define	PvmCantStart	-29	/* Can't start pvmd */
#define	PvmAlready		-30	/* Already in progress */
#define	PvmNoTask		-31	/* No such task */
#define	PvmNoEntry		-32	/* No such entry */
#define	PvmDupEntry		-33	/* Duplicate entry */

/*
*	Data types for pvm_reduce(), pvm_psend(), pvm_precv()
*/

#define	PVM_STR			0	/* string */
#define	PVM_BYTE		1	/* byte */
#define	PVM_SHORT		2	/* short */
#define	PVM_INT			3	/* int */
#define	PVM_FLOAT		4	/* real */
#define	PVM_CPLX		5	/* complex */
#define	PVM_DOUBLE		6	/* double */
#define	PVM_DCPLX		7	/* double complex */
#define	PVM_LONG		8	/* long integer */
#define	PVM_USHORT		9	/* unsigned short int */
#define	PVM_UINT		10	/* unsigned int */
#define	PVM_ULONG		11	/* unsigned long int */

#endif
