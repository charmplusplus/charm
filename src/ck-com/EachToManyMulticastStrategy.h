#ifndef EACH_TO_MANY_MULTICAST_STRATEGY
#define EACH_TO_MANY_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "routerstrategy.h"

class EachToManyMulticastStrategy: public CharmStrategy {
 protected:
    int routerID;      //Which topology
    int npes, *pelist; //Domain of the topology
    int MyPe;          //My id in that domain

    int ndestpes, *destpelist; //Destination processors
    int handlerId;
    
    //Executes common code just after array and group constructors
    virtual void commonInit();

    RouterStrategy *rstrat;
    int useLearner;

 public:
    //Group constructor
    EachToManyMulticastStrategy(int strategyId, int nsrcpes=0, 
                                int *srcpelist=0, 
                                int ndestpes =0, int *destpelist =0);
    
    //Array constructor
    EachToManyMulticastStrategy(int substrategy, CkArrayID src, 
                                CkArrayID dest, int nsrc=0, 
                                CkArrayIndexMax *srcelements=0, int ndest=0, 
                                CkArrayIndexMax *destelements=0);
    
    EachToManyMulticastStrategy(CkMigrateMessage *m) : CharmStrategy(m){};
    
    ~EachToManyMulticastStrategy();

    //Basic function, subclasses should not have to change it
    virtual void insertMessage(CharmMessageHolder *msg);
    //More specielized function
    virtual void doneInserting();

    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    virtual void finalizeProcessing();
    virtual void localMulticast(void *msg);
    
    PUPable_decl(EachToManyMulticastStrategy);
    
    inline void enableLearning() {useLearner = 1;}
};
#endif

