#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
/*   #include "procfs.h"    */
/*   Insert .h file here: */

#define MAXCOMMSIZ 16
#define MEM_USE_VECTOR_SIZE 100

float vm[MEM_USE_VECTOR_SIZE];
float rss[MEM_USE_VECTOR_SIZE];


typedef struct {
    pid_t pid;               /* Process ID */

    char comm[MAXCOMMSIZ];   /* The filename of the executable, in
                              * parentheses.  This is visible     
                              * whether or not the executable is  
                              * swapped out.
                              */

    char state;              /* One character from the string
                              * "RSDZTW" where R is running, S is
                              * sleeping in an interruptible wait, D
                              * is sleeping in an uninterruptible
                              * disk sleep, Z is zombie, T
                              * is traced or stopped (on a signal),
                              * and W is paging.
                              */

    pid_t ppid;              /* The PID of the parent. */

    pid_t pgrp;              /* The process group ID of the process. */

    pid_t session;           /* The session ID of the process. */

    int tty;                 /* The tty the process uses. */

    int tpgid;               /* The process group ID of the process
                              * which currently owns the tty that
                              * the process is connected to.
                              */

    unsigned long flags;     /* The flags of the process.
                              * The math bit is decimal 4, and the
                              * traced bit is a decimal 10.
                              */

    unsigned long minflt;    /* The number of minor faults the
                              * process has made, those which have
                              * not required loading a memory page
                              * from disk.
                              */

    unsigned long cminflt;   /* The number of minor faults that the
                              * process and its children have made.
                              */

    unsigned long majflt;    /* The number of major faults the
                              * process has made, those which have
                              * required loading a memory page from
                              * disk.
                              */

    unsigned long cmajflt;   /* The number of major faults that the
                              * process and its children have made.
                              */

    clock_t utime;           /* The number of jiffies that this
                              * process has been scheduled in user
                              * mode.
                              */

    clock_t stime;           /* The number of jiffies that this
                              * process has been scheduled in kernel
                              * mode.
                              */

    clock_t cutime;          /* The number of jiffies that this
                              * process and its children have been
                              * scheduled in kernel mode.
                              */

    clock_t cstime;          /* The number of jiffies that this
                              * process and its children have been
                              * scheduled in kernel mode.
                              */


    long priority;           /* The standard nice value, plus
                              * fifteen.  The value is never
                              * negative in the kernel.
                              */

    long nice;               /* The nice value ranges from 19 (nicest)
                              * to -19 (not nice to others).
                              */

    /* @@ constant 0UL in array.c here... */

    unsigned long itrealvalue;/* The time (in jiffies) before the
                              * next SIGALRM is sent to the process
                              * due to an interval timer.
                              */

    unsigned long starttime; /* Time the process started in jiffies
                              * after system boot.
                              */

    unsigned long vsize;     /* Virtual memory size in bytes */

    unsigned long rss;       /* Resident Set Size: number of pages
                              * the process has in real memory,
                              * minus 3 for administrative purposes.
                              * This is just the pages which count
                              * towards text, data, or stack space.
                              * This does not include pages which
                              * have not been demand-loaded in, or
                              * which are swapped out.
                              */

    unsigned long rlim;      /* Current limit in bytes on the rss of
                              * the process (usually 4,294,967,295).
                              */

    unsigned long startcode; /* The address above which program text
                              * can run.
                              */

    unsigned long endcode;   /* The address below which program text
                              * can run.
                              */

    unsigned long startstack;/* The address of the start of the
                              * stack.
                              */

    unsigned long kstkesp;   /* The current value of esp (stack
                              * pointer), as found in the
                              * kernel stack page for the process.
                              */

    unsigned long kstkeip;   /* The current EIP (instruction
                              * pointer).
                              */

    int signal;              /* The bitmap of pending signals
                              * (usually 0).
                              */

    int blocked;             /* The bitmap of blocked signals
                              * (usually 0, 2 for shells).
                              */

    int sigignore;           /* The bitmap of ignored signals. */

    int sigcatch;            /* The bitmap of catched signals. */

    unsigned long wchan;     /* This is the "channel" in which the
                              * process is waiting.  This is the
                              * address of a system call, and can be
                              * looked up in a namelist if you need
                              * a textual name. (If you have an up-
                              * to-date /etc/psdatabase, then try ps -l
                              * to see the WCHAN field in action)
                              */

    unsigned long nswap;     /* Number of pages swapped - not maintained. */

    unsigned long cnswap;    /* Cumulative nswap for child processes. */

    int exitsignal;          /* Signal to be sent to parent when we die. */

    int processor;           /* Processor number last executed on. */
} ps_procstat_t;

/* End of procfs.h  */

#define MEGABYTE 1048576

static int pgsize;

int ps_procstat(ps_procstat_t *);

#ifdef MEMDUMMY

void get_mem_use(p_vm,p_rss)
float *p_vm, *p_rss;
{
	*p_vm=0.0; *p_rss=0.0;
}

#else

/* get the values and return them */
void get_mem_use(p_vm,p_rss)
float *p_vm, *p_rss;
{
    int ret = 0;
    float vsize,rss;
    float vsizemb, rssmb;
    ps_procstat_t p;

    if ( ( ret = ps_procstat(&p) ) == 0 ) {
	vsize = p.vsize;
	rss = p.rss;
    }

    pgsize = getpagesize();
    vsizemb = ( vsize ) / (float) MEGABYTE;
    rssmb = ( rss * pgsize ) / (float) MEGABYTE;

    *p_vm = vsizemb;
    *p_rss = rssmb;
}

#endif

int ps_procstat(ps_procstat_t *p)
{
    FILE *fd;
    char fname[64];

    sprintf(fname, "/proc/self/stat");

    if ( ( fd=fopen(fname, "r") ) == 0 ) {
        return -1;
    }

    fscanf(fd, "%d (%[^)]) %c %d", &p->pid, p->comm, &p->state, &p->ppid);
    fscanf(fd, "%d %d %d %d %lu",  &p->pgrp, &p->session, &p->tty, &p->tpgid,
           &p->flags);
    fscanf(fd, "%lu %lu %lu %lu",  &p->minflt, &p->cminflt, &p->majflt,
           &p->cmajflt);
    fscanf(fd, "%ld %ld %ld %ld",  &p->utime, &p->stime, &p->cutime,
           &p->cstime);
    fscanf(fd, "%ld %ld %*d %lu",  &p->priority, &p->nice, &p->itrealvalue);
    fscanf(fd, "%lu %lu %lu %lu",  &p->starttime, &p->vsize, &p->rss,
           &p->rlim);
    fscanf(fd, "%lu %lu %lu %lu %lu", &p->startcode, &p->endcode,
           &p->startstack, &p->kstkesp, &p->kstkeip);
    fscanf(fd, "%d %d %d %d %lu",   &p->signal, &p->blocked, &p->sigignore,
           &p->sigcatch, &p->wchan);
    fscanf(fd, "%lu %lu %d %d",    &p->nswap, &p->cnswap, &p->exitsignal,
           &p->processor);

    fclose(fd);

    return 0;
}

// function to add the allocated memory at a particular iteration
void set_mem_use_(int *iter){
	get_mem_use(&vm[*iter],&rss[*iter]);
}

// function to print all the allocated memory per iteration
void dump_mem_use_(int *upper_limit){
	int i;
	for(i=1; i<=*upper_limit; i++){
		printf("%d vm=%.2f rss=%.2f\n",i,vm[i],rss[i]);
	}
}
