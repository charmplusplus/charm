/**
 * \addtogroup CkLdb
*/
/*@{*/

/** Cluster Manager Code, 
   Accepts external bit vectors and then feeds it into the
   loadbalancer so that programs can shrink and expand. 
*/

#include "manager.h"
#include "CentralLB.h"
#include "converse.h"
#include "conv-ccs.h"

#if CMK_SHRINK_EXPAND
realloc_state pending_realloc_state;
char * se_avail_vector;
extern "C" int numProcessAfterRestart;
extern "C" CcsDelayedReply shrinkExpandreplyToken;
extern "C" char willContinue=0;
#endif
extern int load_balancer_created;
static void handler(char *bit_map)
{
#if CMK_SHRINK_EXPAND
    shrinkExpandreplyToken = CcsDelayReply();
    bit_map += CmiMsgHeaderSizeBytes;
    pending_realloc_state = REALLOC_MSG_RECEIVED;

    if((CkMyPe() == 0) && (load_balancer_created))
    LBDatabaseObj()->set_avail_vector(bit_map);

    se_avail_vector = (char *)malloc(sizeof(char) * CkNumPes());
    LBDatabaseObj()->get_avail_vector(se_avail_vector);

    numProcessAfterRestart = *((int *)(bit_map + CkNumPes()));
#endif
}

void manager_init(){
#if CMK_SHRINK_EXPAND
    static int inited = 0;
    willContinue = 0;
    if (inited) return;
    CcsRegisterHandler("set_bitmap", (CmiHandler) handler);
    inited = 1;
    pending_realloc_state = NO_REALLOC;
#endif
}


/*@}*/
