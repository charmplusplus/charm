#include "main.decl.h"

#include "searchEngine.h"
int initial_grainsize;
char inputfile[50];

class Main
{
public:
    Main(CkArgMsg* msg )
    {
        if(msg->argc > 2)
        {
            initial_grainsize = atoi(msg->argv[2]);
        }
        delete msg;

        CkPrintf("\nInstance file:%s\ngrainsize:t%d\nprocessor number:%d\n", msg->argv[1], initial_grainsize, CkNumPes()); 
        strcpy(inputfile, msg->argv[1]);
        searchEngineProxy.start();
    }

};

#include "main.def.h"
