#include "manager.h"
#include "CentralLB.h"
#include "converse.h"
#include "conv-ccs.h"

/* Cluster Manager Code, 
   Accpets external bit vectors and then feeds it into the 
   loadbalancer so that programs can shrink and expand. 
*/

CkGroupID manager_group_id;
extern int load_balancer_created;
int group_created;

void handler(char *bit_map)
{
    bit_map += CmiMsgHeaderSizeBytes;
    
    CkPrintf("in handler\n");
    
    for(int i=0; i < CkNumPes() ; i++) 
	CkPrintf("%d, ",bit_map[i]);

    if((CkMyPe() == 0) && (load_balancer_created))
	set_avail_vector(bit_map);
}

void manager_init(){    
    CcsRegisterHandler("set_bitmap", (CmiHandler) handler);
}





