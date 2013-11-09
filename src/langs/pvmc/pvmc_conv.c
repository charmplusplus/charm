#include <stdio.h>
#include <stddef.h>
#include <converse.h>
#include "pvmc.h"

void pvmc_init(void)
{
  pvmc_init_bufs();
  pvmc_init_comm();
  pvmc_init_groups();
}

int pvm_mytid(void)
{
  /*
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_mytid()\n",MYPE(),PE2TID(MYPE()+1));
#endif
*/
  return PE2TID(MYPE());
}

int pvm_exit(void)
{
  int sleepval;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_exit()\n",MYPE(),pvm_mytid());
#endif

  ConverseExit();
}

int pvm_spawn(char *task, char **argv, int flag,
	      char *where, int ntask, int *tids)
{
  int i;
  int numt;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_spawn()\n",MYPE(),pvm_mytid());
#endif

  numt = ntask;
  if (numt > CmiNumPes())
    numt = CmiNumPes();
#ifdef PVM_DEBUG
  PRINTF("%s: preping %d tids (wanted to prep %d)\n",__FILE__,numt,ntask);
#endif
  for(i=0; i<numt; i++)
  {
    tids[i]=PE2TID(i);
#ifdef PVM_DEBUG
    PRINTF("Pe(%d) tids[%d]=%d  (%d)\n",MYPE(),i,PE2TID(i),tids[i]);
#endif
  }
    
  return numt;
}

int pvm_parent(void)
{
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_parent()\n",MYPE(),pvm_mytid());
#endif

  /*  
   *  I think it would be better to return PvmNoParent, but
   *  this may make sense too, and it makes namd2/DPMTA work.
   */
  return 1;
}

int pvm_config(int *nhost, int *narch, struct pvmhostinfo **hostp)
{
  int i, nh=0;

  /*  sleep(10); */
  
  if (nhost)
    {
#ifdef PVM_DEBUG
  PRINTF("tid=%d:pvm_config(%x,%x,%x)\n",pvm_mytid(),nhost,narch,hostp);
#endif
      printf("tid=%d:pvm_config(%x,%x,%x)\n",pvm_mytid(),nhost,narch,hostp);
      printf("%d\n",*nhost);
      *nhost=nh=CmiNumPes();
    }
  else
    return -1;

  if (narch)
    *narch=1;

  *hostp = (struct pvmhostinfo *)MALLOC(nh*sizeof(struct pvmhostinfo));

  if (*hostp == (struct pvmhostinfo *)NULL)
    return -1;

  for(i=0; i<nh; i++) {
    hostp[i]->hi_tid=PE2TID(i);
    hostp[i]->hi_name="";
    hostp[i]->hi_arch="CONVERSE";
    hostp[i]->hi_speed=1000;
  }

  return 0;
}

int pvm_tasks(int which, int *ntask, struct pvmtaskinfo **taskp)
{
  int i;

#ifdef PVM_DEBUG
  PRINTF("tid=%d:pvm_tasks(%d,%x,%x)\n",pvm_mytid(),which,ntask,taskp);
#endif

  if (which==0)
    *ntask=CmiNumPes();
  else
    *ntask=1;

  *taskp = (struct pvmtaskinfo *)MALLOC(*ntask * sizeof(struct pvmtaskinfo));
  
  if (*taskp == (struct pvmtaskinfo *)NULL)
    return -1;

  for(i=0; i<*ntask; i++) {
    taskp[i]->ti_tid=PE2TID(i);
    taskp[i]->ti_ptid=PE2TID(0);
    taskp[i]->ti_host=PE2TID(i);
    taskp[i]->ti_flag=0;
    taskp[i]->ti_a_out="";
  }     

  return 0;
}

int pvm_setopt(int what, int val)
{
#ifdef PVM_DEBUG
  PRINTF("tid=%d:pvm_setopt(%d,%d)\n",pvm_mytid(),what,val);
#endif
  return val;
}

int pvm_gsize(char *group)
{
#ifdef PVM_DEBUG
  PRINTF("tid=%d:pvm_gsize(%s)\n",pvm_mytid(),group);
#endif
  return CmiNumPes();
}

int pvm_gettid(char *group, int inum)
{
#ifdef PVM_DEBUG
  PRINTF("tid=%d:pvm_gettid(%s,%d)\n",pvm_mytid(),group,inum);
#endif
  return inum;
}

int pvm_catchout(FILE *ff)
{
  PRINTF("Warning: pvm_catchout not implemented\n");
}
