#include "startup.decl.h"

class Main : public CBase_Main
{
public:

    Main(CkArgMsg* m) {
        CkExit();
    }
};

#include "startup.def.h"
