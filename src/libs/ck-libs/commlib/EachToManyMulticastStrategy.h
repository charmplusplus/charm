#ifndef EACH_TO_MANY_MULTICAST_STRATEGY
#define EACH_TO_MANY_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "DirectMulticastStrategy.h"

class EachToManyMulticastStrategy: public Strategy {
 protected:
    CkQ <CharmMessageHolder*> *messageBuf;
    int routerID;      //Which topology
    comID comid;
    
    int npes, *pelist, *procMap; //Domain of the topology
    int MyPe;          //My id in that domain

    int ndestpes, *destpelist, *destMap; //Destination processors

    int handlerId;

    CkArrayID destArrayID;
    int nDestElements;  //0 for all array elements
    CkArrayIndexMax *destIndices; //NULL for all indices
    CkVec<CkArrayIndexMax> localDestIndices;
    //Dynamically set by the application
    CkHashtableT<ComlibSectionHashKey, void *> sec_ht;

    void localMulticast(CkVec<CkArrayIndexMax> *vec, envelope *env);
    ComlibMulticastMsg *getPackedMulticastMessage(CharmMessageHolder *cmsg);
    void setReverseMap();    
    
    //Executes common code just after array and group constructors
    virtual void commonInit();
    virtual void initSectionID(CkSectionID *sid);

    int MaxSectionID;

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
    
    EachToManyMulticastStrategy(CkMigrateMessage *m){}

    //Basic function, subclasses should not have to change it
    virtual void insertMessage(CharmMessageHolder *msg);
    //More specielized function
    virtual void doneInserting();

    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    virtual void localMulticast(void *msg);
    virtual void setDestArray(CkArrayID dest) {destArrayID=dest;}
    
    ComlibMulticastMsg *getNewMulticastMessage(CharmMessageHolder *m);

    PUPable_decl(EachToManyMulticastStrategy);

};
#endif

