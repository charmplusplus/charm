#include "main.decl.h"

#include "searchEngine.h"
#include "uts.h"

int initial_grainsize;

class Main
{
public:
    Main(CkArgMsg* msg )
    {
        uts_parseParams(msg->argc, msg->argv);                                                              
        initial_grainsize = atoi(msg->argv[1]);                 
        delete msg;
        uts_printParams();                                                                                
        searchEngineProxy.start();
    }
};

#include "main.def.h"
