#include <stddef.h>
#include <converse.h>
#include "pvmc.h"

CpvDeclare(int,pvmc_at_barrier_num);
CpvDeclare(int,pvmc_barrier_num);
CpvStaticDeclare(int,pvmc_last_at_barrier_num);
CpvStaticDeclare(int,pvmc_last_barrier_num);

CpvExtern(int,pvmc_control_handler);

void pvmc_init_groups()
{
  CpvInitialize(int,pvmc_barrier_num);
  CpvAccess(pvmc_barrier_num)=0;
  CpvInitialize(int,pvmc_at_barrier_num);
  CpvAccess(pvmc_at_barrier_num)=0;
  CpvInitialize(int,pvmc_last_barrier_num);
  CpvAccess(pvmc_last_barrier_num)=0;
  CpvInitialize(int,pvmc_last_at_barrier_num);
  CpvAccess(pvmc_last_at_barrier_num)=0;
}

int pvm_joingroup(const char *group)
{
  PRINTF("TID %d joining group %s -- group support is limited\n",pvm_mytid(),group);

  return pvm_mytid();
}

int pvm_lvgroup(const char *group)
{
  PRINTF("TID %d leaving group %s -- group support is limited\n",pvm_mytid(),group);

  return 0;
}

int pvm_bcast(const char *group, int msgtag)
{
  int i;
  int return_val=0;

  for(i=0; i<NUMPES(); i++)
    if(i!=MYPE())
      return_val+=pvm_send(PE2TID(i),msgtag);

  return return_val;

}

int pvm_barrier(const char *group, int count)
{
  int i;
  
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_barrier(%s,%d) waiting for barrier %d\n",
	MYPE(),pvm_mytid(),group,count,CpvAccess(pvmc_last_barrier_num)+1);
#endif

  /*
   * First, everyone except Node 0 sends a message to node 0
   */
  if (MYPE() != 0)
    pvmc_send_control_msg(PVMC_CTRL_AT_BARRIER,0);

  /* 
   * Node 0 will wait for NUMPES-1 messages, then send the response back
   */
  if (MYPE() == 0)
  {
    while(CpvAccess(pvmc_at_barrier_num) < 
	  CpvAccess(pvmc_last_at_barrier_num) + NUMPES()-1)
    {
      /* Empty network buffer */
      while(CmiDeliverMsgs(1)==0)
	;
    }
    /*
     * Now, node 0 has received the proper number of messages, so it must
     * tell the other nodes to continue
     */
    for (i=1; i < NUMPES(); i++)
      pvmc_send_control_msg(PVMC_CTRL_THROUGH_BARRIER,i);
    /*
     * Now node 0 must set itself up for the next call
     */
    CpvAccess(pvmc_last_at_barrier_num) += NUMPES() - 1;
    /*
     * And finally, node 0 tells itself that it has passed the barrier
     */
    CpvAccess(pvmc_barrier_num)++;
  }

  /*
   * Now, all processors wait for barrier passage, by looking 
   * at pvmc_barrier_num counter
   */
  while(CpvAccess(pvmc_barrier_num) == 	CpvAccess(pvmc_last_barrier_num))
  {
    /* Empty network buffer */
    while(CmiDeliverMsgs(1)==0)
      ;
  }
  /*
   *  Barrier reached. Set up for next barrier
   */
  CpvAccess(pvmc_last_barrier_num)++;
  
}

