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
        //--------------------Scatterv functions
        void stripScatterMsg(CkMessage *msg, CkScatterWrapper &w);
        void ckScatterSpanningTree(CkMessage *msg, CkScatterWrapper &w, bool free);
        void scatterSendAll(CkMessage *msg, CkScatterWrapper &w, bool local, bool free);
        
    public:
        // ------------------------- Cons/Des-tructors ------------------------
        CkCollectiveMgr(CkMigrateMessage *m)  {}
        CkCollectiveMgr(){}
        void pup(PUP::er &p){ 
          CBase_CkCollectiveMgr::pup(p);
        }
        //--------------------Scatterv functions
        void ckScatter(void *msg, CkArrayID aid);
        void ckScatterTree(void *msg, CkArrayID aid);
        void ckScatterSpanningTreeEntry(CkMessage *msg);
        void scatterSendAllEntry(CkMessage *msg);
};

#endif
