#ifndef EACH_TO_MANY_MULTICAST_STRATEGY
#define EACH_TO_MANY_MULTICAST_STRATEGY

#include "ComlibManager.h"

class EachToManyMulticastStrategy: public Strategy {
    CkQ <CharmMessageHolder*> *messageBuf;
    int routerID;      //Which topology
    comID comid;
    
    int npes, *pelist, *procMap; //Domain of the topology
    int MyPe;          //My id in that domain

    int ndestpes, *destpelist, *destMap; //Destination processors

    long handler;  //Multicast Handler function pointer to be called on the 
                   //receiving processors.
    int handlerId;

    CkArrayID destArrayID;
    int nDestElements;  //0 for all array elements
    CkArrayIndexMax *destIndices; //NULL for all indices
    
    CkVec<CkArrayIndexMax> localDestIndices;
    
    void init();
    void localMulticast(CkVec<CkArrayIndexMax> vec, envelope *env);
    void setReverseMap();

 public:
    EachToManyMulticastStrategy(int strategyId,int nsrcpes =0,int *srcpelist=0);
    EachToManyMulticastStrategy(int strategyId, int nsrcpes, int *srcpelist, 
                                int ndestpes =0, int *destpelist =0);
    
    EachToManyMulticastStrategy(int substrategy, CkArrayID src, int nsrc=0, 
                                CkArrayIndexMax *srcelements=0);

    EachToManyMulticastStrategy(int substrategy, CkArrayID src, 
                                CkArrayID dest, int ndest=0, 
                                CkArrayIndexMax *destelements=0);
    
    EachToManyMulticastStrategy(int substrategy, CkArrayID src, int nsrc, 
                                CkArrayIndexMax *srcelements, CkArrayID dest,
                                int ndest=0, 
                                CkArrayIndexMax *destelements=0);

    EachToManyMulticastStrategy(CkMigrateMessage *m){}

    //ComlibMulticastHandler getHandler()
    //  {return (ComlibMulticastHandler)handler;};
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    void localMulticast(void *msg);
    void setDestArray(CkArrayID dest) {destArrayID=dest;}
    
    PUPable_decl(EachToManyMulticastStrategy);
};
#endif

