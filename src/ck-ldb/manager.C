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
int numProcessAfterRestart;
extern "C" CcsDelayedReply shrinkExpandreplyToken;
extern "C" char willContinue;
char willContinue;
#endif
bool load_balancer_created;

void realloc(char* reallocMsg)
{
    numProcessAfterRestart = *((int *)(reallocMsg + CkNumPes()));

    if (LBManagerObj()->lb_in_progress)
    {
        CkPrintf("Charm> Rescaling called while load balancing is in progress!\n");
        LBManagerObj()->bufferRealloc(reallocMsg);
    }
    else
    {
        if (numProcessAfterRestart > CkNumPes())
            pending_realloc_state = EXPAND_MSG_RECEIVED;
        else
            pending_realloc_state = SHRINK_MSG_RECEIVED;

        if((CkMyPe() == 0) && (load_balancer_created))
        LBManagerObj()->set_avail_vector(reallocMsg, 0);

        se_avail_vector = (char *)malloc(sizeof(char) * CkNumPes());
        LBManagerObj()->get_avail_vector(se_avail_vector);
    }
}

static void handler(char *bit_map)
{
#if CMK_SHRINK_EXPAND
    printf("Charm> Rescaling called!\n");
    shrinkExpandreplyToken = CcsDelayReply();
    bit_map += CmiMsgHeaderSizeBytes;
    realloc(bit_map);
#endif
}

void rescale(char* bit_map)
{
    handler(bit_map);
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
