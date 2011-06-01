/**
   @addtogroup ComlibCharmStrategy
   @{
   @file 
   @brief Optimized all-to-all communication that can combine messages and send them along virtual topologies.
*/

#ifndef EACH_TO_MANY_MULTICAST_STRATEGY
#define EACH_TO_MANY_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "routerstrategy.h"

/**
   The EachToManyMulticast Strategy optimizes all-to-all
   communication. It combines messages and sends them along
   virtual topologies 2d mesh, 3d mesh and hypercube using
   the RouterStrategy as underlying strategy.

   For large messages send them directly.

   This is the object level strategy. For processor level
   optimizations the underlying RouterStrategy is called.

   @author Sameer Kumar, Filippo, and Isaac

*/
class EachToManyMulticastStrategy : public RouterStrategy, public CharmStrategy {
 protected:
 
    /// Executes common code just after array and group constructors
    virtual void commonInit(int*);

 public:
    /// Group constructor
    /// If only the first three parameters are provided, the whole group will be used for the multicast(0 to CkNumPes)
    /// TODO verify that the 0 parameter 
    EachToManyMulticastStrategy(int strategyId, CkGroupID src, CkGroupID dest, 
    		int nsrcpes=0, int *srcpelist=0, 
    		int ndestpes =0, int *destpelist =0);

    /// Array constructor
    /// TODO: Fix this to allow for the same parameters as would be given to an array section creation(ranges of indices).
    EachToManyMulticastStrategy(int substrategy, CkArrayID src, 
                                CkArrayID dest, int nsrc=0, 
                                CkArrayIndex *srcelements=0, int ndest=0, 
                                CkArrayIndex *destelements=0);
    
    EachToManyMulticastStrategy(CkMigrateMessage *m) : RouterStrategy(m), CharmStrategy(m) {
      ComlibPrintf("[%d] EachToManyMulticast migration constructor\n",CkMyPe());
    };
    
    ~EachToManyMulticastStrategy();

    void insertMessage(MessageHolder *msg) {
      ((CharmMessageHolder*)msg) -> checkme();
      insertMessage((CharmMessageHolder*)msg);
    }

    //Basic function, subclasses should not have to change it
    virtual void insertMessage(CharmMessageHolder *msg);

    virtual void pup(PUP::er &p);    
    virtual void localMulticast(void *msg);

    virtual void notifyDone();

    /** Called by handleMessage at the destinations for the broadcast if in DIRECT mode. */
    virtual void deliver(char *, int);

    /// this method can be called when the strategy is in DIRECT mode, so the
    /// message will go the comlib_handler and then arrive here.
    virtual void handleMessage(void *msg) {
      envelope *env = (envelope*)msg;
      
      deliver((char*)msg, env->getTotalsize());
    }
    
    PUPable_decl(EachToManyMulticastStrategy);

};
#endif

/*@}*/
