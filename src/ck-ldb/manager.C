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
#include <regex>
#include <iostream>
#include <fstream>
#include <string>

#if CMK_SHRINK_EXPAND
realloc_state pending_realloc_state;
char * se_avail_vector;
int numProcessAfterRestart;
extern "C" CcsDelayedReply shrinkExpandreplyToken;
extern "C" char willContinue;
char willContinue;
#endif
bool load_balancer_created;

void write_hostfile(int numProcesses) 
{
    std::ifstream infile("/etc/mpi/hostfile");

    if (infile.good())
    {
        std::string sLine;
        getline(infile, sLine);
        std::regex rgx("host (.*)-worker-(\\d) ++cpus (\\d)");
        std::smatch match;
        char hostStr[200];

        if (std::regex_search(sLine, match, rgx, std::regex_constants::match_default))
        {
            std::string name = match[0];
            int slots = std::stoi(match[2]);

            infile.close();

            std::ofstream outfile("/etc/mpi/hostfile");

            for (int i = 0; i < numProcesses; i++)
            {
                sprintf(hostStr, "host %s-worker-%i ++cpus %i\n", name.c_str(), i, slots);
                outfile << hostStr;
            }
        }
        else
        {
            printf("Error parsing hostfile regex\n");
        }
    }
    else
    {
        printf("Error opening hostfile\n");
    }

}

static void handler(char *bit_map)
{
#if CMK_SHRINK_EXPAND
    printf("Charm> Rescaling called!\n");
    shrinkExpandreplyToken = CcsDelayReply();
    bit_map += CmiMsgHeaderSizeBytes;
    pending_realloc_state = REALLOC_MSG_RECEIVED;

    if((CkMyPe() == 0) && (load_balancer_created))
    LBManagerObj()->set_avail_vector(bit_map);

    se_avail_vector = (char *)malloc(sizeof(char) * CkNumPes());
    LBManagerObj()->get_avail_vector(se_avail_vector);

    numProcessAfterRestart = *((int *)(bit_map + CkNumPes()));

    write_hostfile(numProcessAfterRestart);
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
