#ifndef DIRECT_MULTICAST_STRATEGY
#define DIRECT_MULTICAST_STRATEGY

#include "ComlibManager.h"

void *DMHandler(void *msg);

class ComlibSectionHashKey{
 public:

    int srcPe;
    int id;
    ComlibSectionHashKey(int _pe, int _id):srcPe(_pe), id(_id){};

    //These routines allow ComlibSectionHashKey to be used in
    //  a CkHashtableT
    CkHashCode hash(void) const;
    static CkHashCode staticHash(const void *a,size_t);
    int compare(const ComlibSectionHashKey &ind) const;
    static int staticCompare(const void *a,const void *b,size_t);
};

inline CkHashCode ComlibSectionHashKey::hash(void) const
{
    register int _id = id;
    register int _pe = srcPe;
    
    register CkHashCode ret = (_id << 16) + _pe;
    return ret;
}

inline int ComlibSectionHashKey::compare(const ComlibSectionHashKey &k2) const
{
    if(id == k2.id && srcPe == k2.srcPe)
        return 1;
    
    return 0;
}

/*For calls to qsort*/
int intCompare(void *a, void *b);

class DirectMulticastStrategy: public Strategy {
 protected:
    CkQ <CharmMessageHolder*> *messageBuf;

    int ndestpes, *destpelist; //Destination processors
    int handlerId;
    int MaxSectionID;

    int isDestinationArray, isDestinationGroup;

    //Array support
    CkArrayID destArrayID;
    CkVec<CkArrayIndexMax> localDestIndices;
    //Array section support
    CkHashtableT<ComlibSectionHashKey, void *> sec_ht; 
    
    //Initialize and cache information in a section id which can be
    //used the next time the section is multicast to.
    virtual void initSectionID(CkSectionID *sid);
    
    //Common Initializer for group and array constructors
    //Every substrategy should implement its own
    void commonInit();
    
    //Called to multicast an array message locally
    void localMulticast(CkVec<CkArrayIndexMax> *, envelope *env);

    //Create a new multicast message with the array section in it
    ComlibMulticastMsg * getNewMulticastMessage(CharmMessageHolder *cmsg);

 public:
    
    //Group constructor
    DirectMulticastStrategy(int ndestpes = 0, int *destpelist = 0);    

    //Array constructor
    DirectMulticastStrategy(CkArrayID aid);

    DirectMulticastStrategy(CkMigrateMessage *m): Strategy(m){}
    
    virtual void insertMessage(CharmMessageHolder *msg);
    virtual void doneInserting();

    //Called by the converse handler function
    virtual void handleMulticastMessage(void *msg);
    
    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    
    PUPable_decl(DirectMulticastStrategy);

};
#endif

