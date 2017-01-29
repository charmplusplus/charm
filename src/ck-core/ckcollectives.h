#ifndef _COLLECTIVES_MGR
#define _COLLECTIVES_MGR

#include "pup.h"

#include "CkCollectives.decl.h"

#define CMK_COLLECTIVE_MGR 1 

/**
 * CkCollectiveMgr class is meant to provide a variety of collective operations API.
 */
class CkCollectiveMgr: public CBase_CkCollectiveMgr
{
    private:
        
    public:
        // ------------------------- Cons/Des-tructors ------------------------
        CkCollectiveMgr(CkMigrateMessage *m)  {}
        CkCollectiveMgr(){}
        void pup(PUP::er &p){ 
          CBase_CkCollectiveMgr::pup(p);
        }
        void ckScatter(void *msg, CkArrayID aid);
        void scatterSendAll(CkMessage *msg);
};

#endif
