#ifndef DIRECT_MULTICAST_STRATEGY
#define DIRECT_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

void *DMHandler(void *msg);

class DirectMulticastStrategy: public CharmStrategy {
 protected:
    CkQ <CharmMessageHolder*> *messageBuf;

    int ndestpes, *destpelist; //Destination processors
    int handlerId;
    
    ComlibSectionInfo sinfo;

    //Array section support
    CkHashtableT<ComlibSectionHashKey, void *> sec_ht; 
    
    //Common Initializer for group and array constructors
    //Every substrategy should implement its own
    void commonInit();
    
 public:
    
    //Group constructor
    DirectMulticastStrategy(int ndestpes = 0, int *destpelist = 0);    

    //Array constructor
    DirectMulticastStrategy(CkArrayID aid);

    DirectMulticastStrategy(CkMigrateMessage *m): CharmStrategy(m){}
    
    virtual void insertMessage(CharmMessageHolder *msg);
    virtual void doneInserting();

    //Called by the converse handler function
    virtual void handleMulticastMessage(void *msg);
    
    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    
    PUPable_decl(DirectMulticastStrategy);
};
#endif

