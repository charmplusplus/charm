#include "main.decl.h"

#include "searchEngine.h"
int branchfactor;
int depth;
int initial_grainsize;
int target;

class Main
{
public:
    Main(CkArgMsg* msg )
    {
        branchfactor = 2;
        CkPrintf("main initial_grainsize branchfactor depth target\n");
        if(msg->argc == 5)
        {
            initial_grainsize = atoi(msg->argv[1]);
            branchfactor = atoi(msg->argv[2]);
            depth = atoi(msg->argv[3]);
            target = atoi(msg->argv[4]);            
            delete msg;
        }else
        {
            CkPrintf("Check input parameter\n");
            delete msg;
            CkExit();
        }

        searchEngineProxy.start();
    }

};

#include "main.def.h"
