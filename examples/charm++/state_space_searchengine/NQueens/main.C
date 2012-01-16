#include "main.decl.h"

#include "searchEngine.h"
int initial_grainsize;
int numQueens;

class Main
{
public:
    Main(CkArgMsg* msg )
    {
        numQueens = 5;
        if(msg->argc > 2)
        {
            numQueens = atoi(msg->argv[1]);
            initial_grainsize = atoi(msg->argv[2]);
        }
        delete msg;

        searchEngineProxy.start();
    }

};

#include "main.def.h"
