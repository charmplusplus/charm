/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>
#include <BaseLB.h>

#if CMK_LBDB_ON

BaseLB::BaseLB() {
  CkpvAccess(numLoadBalancers) ++;
//CmiPrintf("[%d] BaseLB created!\n", CkMyPe());
  if (CkpvAccess(numLoadBalancers) - CkpvAccess(hasNullLB) > 1)
    CmiAbort("Error: try to create more than one load balancer strategies!");
  lbname = "Unknown";
}

void BaseLB::unregister() {
  theLbdb=CProxy_LBDatabase(lbdb).ckLocalBranch();
  theLbdb->RemoveLocalBarrierReceiver(receiver);
  CkpvAccess(numLoadBalancers) --;
}

#else
BaseLB::BaseLB() {}
void BaseLB::unregister() {}
#endif

void* LBMigrateMsg::alloc(int msgnum, size_t size, int* array, int priobits)
{
  int totalsize = ALIGN8(size) + ALIGN8(array[0] * sizeof(MigrateInfo)) 
    + ALIGN8(CkNumPes() * sizeof(char))+
    + ALIGN8(CkNumPes() * sizeof(double));

  LBMigrateMsg* ret =
    (LBMigrateMsg*)(CkAllocMsg(msgnum,totalsize,priobits));

  ret->moves = (MigrateInfo*) ((char*)(ret)+ ALIGN8(size));
  ret->avail_vector = (char *)(ret->moves + array[0]);
  ret->expectedLoad = (double *)(ret->avail_vector + ALIGN8(CkNumPes()*sizeof(char)));
  return (void*)(ret);
}

void* LBMigrateMsg::pack(LBMigrateMsg* m)
{
  m->moves = (MigrateInfo*)
    ((char*)(m->moves) - (char*)(&m->moves));

  m->avail_vector =(char*)(m->avail_vector
      - (char*)(&m->avail_vector));

  m->expectedLoad = (double*)((char *)m->expectedLoad
      - (char*)(&m->expectedLoad));

  return (void*)(m);
}

LBMigrateMsg* LBMigrateMsg::unpack(void *m)
{
  LBMigrateMsg* ret_val = (LBMigrateMsg*)(m);

  ret_val->moves = (MigrateInfo*)
    ((char*)(&ret_val->moves)
     + (size_t)(ret_val->moves));

  ret_val->avail_vector =
    (char*)((char*)(&ret_val->avail_vector)
			    +(size_t)(ret_val->avail_vector));

  ret_val->expectedLoad =
    (double*)((char*)(&ret_val->expectedLoad)
			    +(size_t)(ret_val->expectedLoad));

  return ret_val;
}

#include "BaseLB.def.h"

/*@}*/
