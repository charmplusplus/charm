/**
   @addtogroup ComlibConverseStrategy
   @{
   @file   
   Class that calls Krishnan's routers from the new Comlib.
   Developed to be called from Converse
   Sameer Kumar 05/14/04
*/
   

#ifndef ROUTER_STRATEGY
#define ROUTER_STRATEGY
#include "convcomlibmanager.h"

class RouterStrategy : public Strategy {
 public:
    RouterStrategy();
    RouterStrategy(CkMigrateMessage *m): Strategy(m){}

    void insertMessage(MessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);
    PUPable_decl(RouterStrategy);
};
#endif
/*@}*/
