/**
 * \addtogroup CkLdb
*/
/*@{*/

/** Cluster Manager Code, 
   Accpets external bit vectors and then feeds it into the 
   loadbalancer so that programs can shrink and expand. 
*/

#include "manager.h"
#include "CentralLB.h"
#include "converse.h"
#include "conv-ccs.h"

extern int load_balancer_created;

static void handler(char *bit_map)
{
    bit_map += CmiMsgHeaderSizeBytes;
    
    CkPrintf("in handler\n");
    
    for(int i=0; i < CkNumPes() ; i++) 
	CkPrintf("%d, ",bit_map[i]);

    if((CkMyPe() == 0) && (load_balancer_created))
	set_avail_vector(bit_map);
}

void manager_init(){    
    static int inited = 0;
    if (inited) return;
    CcsRegisterHandler("set_bitmap", (CmiHandler) handler);
    inited = 1;
}


/*@}*/
