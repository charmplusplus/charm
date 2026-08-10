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
#include <vector>


#if CMK_SHRINK_EXPAND
realloc_state pending_realloc_state;
char * se_avail_vector;
// Durable copy of the availability vector, taken at set_bitmap time. The
// raw se_avail_vector malloc on PE 0 has been observed clobbered (heap
// reuse) between the CCS handler and the rescale exit path, producing
// garbage kill sets; downstream PE-0 consumers read this snapshot instead.
std::vector<char> se_avail_snapshot;
int numProcessAfterRestart;
extern "C" CcsDelayedReply shrinkExpandreplyToken;
extern "C" char willContinue;
char willContinue;
#endif
bool load_balancer_created;

void realloc(char* reallocMsg)
{
#if CMK_SHRINK_EXPAND
    numProcessAfterRestart = *((int *)reallocMsg);
    reallocMsg += sizeof(int);
    int numBits = *((int *)reallocMsg);
    reallocMsg += sizeof(int);

    CkPrintf("Charm> numProcessAfterRestart = %d, numBits = %d\n", numProcessAfterRestart, numBits);

    if (LBManagerObj()->lb_in_progress)
    {
        CkPrintf("Charm> Rescaling called while load balancing is in progress!\n");
        LBManagerObj()->bufferRealloc(reallocMsg - 2 * sizeof(int));
    }
    else
    {
        //if (numProcessAfterRestart > CkNumPes())
        //    pending_realloc_state = EXPAND_MSG_RECEIVED;
        //else
        //    pending_realloc_state = SHRINK_MSG_RECEIVED;

        char* old_bitmap = (char *)malloc(sizeof(char) * CkNumPes());
        LBManagerObj()->get_avail_vector(old_bitmap);

        char* new_bitmap = (char *)malloc(sizeof(char) * CkNumPes());
        memcpy(new_bitmap, old_bitmap, sizeof(char) * CkNumPes());

        int last_pe = -1;
        int j = 0;
        for (int i = 0; i < numBits; i++)
        {
            if (reallocMsg[i] == 0)
            {
                while (last_pe < i && j < CkNumPes())
                    last_pe += old_bitmap[j++];
                
                if (last_pe == i)
                    new_bitmap[j-1] = 0;
            }
        }

        for (int i = 0; i < CkNumPes(); i++)
        {
            CkPrintf("Charm> before old_bitmap[%d] = %d\n", i, old_bitmap[i]);
            CkPrintf("Charm> reallocMsg[%d] = %d\n", i, reallocMsg[i]);
            //new_bitmap[i] = reallocMsg[i] & new_bitmap[i];
            CkPrintf("Charm> after new_bitmap[%d] = %d\n", i, new_bitmap[i]);
        }

        if((CkMyPe() == 0) && (load_balancer_created))
        LBManagerObj()->set_avail_vector(new_bitmap, 0);

        se_avail_vector = (char *)malloc(sizeof(char) * CkNumPes());
        LBManagerObj()->get_avail_vector(se_avail_vector);
        se_avail_snapshot.assign(se_avail_vector, se_avail_vector + CkNumPes());

        // now find whether this is shrink/expand
        pending_realloc_state = NO_REALLOC;
        for (int i = 0; i < CkNumPes(); i++)
            if (se_avail_vector[i] == 0)
            {
                pending_realloc_state = SHRINK_MSG_RECEIVED;
                break;
            }

        if (numProcessAfterRestart > CkNumPes() || (numProcessAfterRestart == CkNumPes() && 
                pending_realloc_state == SHRINK_MSG_RECEIVED))
            pending_realloc_state = static_cast<realloc_state>(static_cast<uint8_t>(pending_realloc_state) | 
                static_cast<uint8_t>(EXPAND_MSG_RECEIVED));

        //free(reallocMsg);
        free(new_bitmap);
        free(old_bitmap);
    }
#endif
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

static void realloc_handler(char *msg)
{
#if CMK_SHRINK_EXPAND
    printf("Charm> Rescaling called!\n");
    int myPes = CkNumPes();
    shrinkExpandreplyToken = CcsDelayReply();
    msg += CmiMsgHeaderSizeBytes;
    bool isExpand = *((bool *)msg);
    int numPes = *((int *)(msg + sizeof(bool)));
    printf("Charm> realloc_handler: isExpand=%d numPes=%d CkNumPes()=%d\n", isExpand, numPes, CkNumPes());
    
    char* bit_map = (char *)malloc(CkNumPes() + 2 * sizeof(int));
    memcpy(bit_map, &numPes, sizeof(int));
    memcpy(&bit_map[sizeof(int)], &myPes, sizeof(int));
    char* start_bitmap = bit_map + 2 * sizeof(int);
    
    if (isExpand)
    {
        for (int i = 0; i < CkNumPes(); i++) {
            start_bitmap[i] = 1;
        }
    }
    else
    {
        for (int i = 0; i < CkNumPes(); i++) {
            if (i < numPes)
                start_bitmap[i] = 1;
            else
                start_bitmap[i] = 0;
        }
    }
    
    realloc(bit_map);
#endif
}

void rescale(char* bit_map)
{
    realloc(bit_map);
}

void manager_init(){
#if CMK_SHRINK_EXPAND
    // CCS handler registration is idempotent (CkHashtablePut overwrites) and the
    // ccsTab is reset on every Converse re-init from a longjmp restart, so we
    // must re-register every time. Otherwise the second-and-later rescales fail
    // with "CCS Client Not Alive" because the set_bitmap handler is missing.
    static int stateInited = 0;
    willContinue = 0;
    CcsRegisterHandler("set_bitmap", (CmiHandler) handler);
    CcsRegisterHandler("realloc", (CmiHandler) realloc_handler);
    if (stateInited) return;
    stateInited = 1;
    pending_realloc_state = NO_REALLOC;
#endif
}


/*@}*/
