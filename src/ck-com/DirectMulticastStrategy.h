#ifndef DIRECT_MULTICAST_STRATEGY
#define DIRECT_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

void *DMHandler(void *msg);

class DirectMulticastStrategy: public CharmStrategy {
 protected:
    //   int handlerId;    
    ComlibSectionInfo sinfo;
    
    //Array section support
    CkHashtableT<ComlibSectionHashKey, ComlibSectionHashObject *> sec_ht; 
    
    //Add this section to the hash table locally
    void insertSectionID(CkSectionID *sid);

    //Called when a new section multicast is called by the user locally.
    //The strategy should then create a topology for it and 
    //return a hash object to store that topology
    virtual ComlibSectionHashObject *createObjectOnSrcPe
        (int nindices, CkArrayIndexMax *idx_list);
   
    //Similar to createHashObjectOnSrcPe, but that this call 
    //is made on the destination or intermediate processor
    virtual ComlibSectionHashObject *createObjectOnIntermediatePe
        (int nindices, CkArrayIndexMax *idx_list, int srcpe);
        
    //Called to multicast the message to local array elements
    void localMulticast(envelope *env, ComlibSectionHashObject *obj);
    
    //Called to send to message out to the remote destinations
    //This method can be overridden to call converse level strategies 
    virtual void remoteMulticast(envelope *env, ComlibSectionHashObject *obj);

    //Process a new message by extracting the array elements 
    //from it and creating a new hash object by calling createObjectOnIntermediatePe();
    void handleNewMulticastMessage(envelope *env);

 public:

    DirectMulticastStrategy(CkMigrateMessage *m): CharmStrategy(m){}
                
    //Array constructor
    DirectMulticastStrategy(CkArrayID aid);
        
    //Destuctor
    ~DirectMulticastStrategy();
        
    virtual void insertMessage(CharmMessageHolder *msg);
    virtual void doneInserting();

    //Called by the converse handler function
    virtual void handleMessage(void *msg);    

    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    
    PUPable_decl(DirectMulticastStrategy);
};
#endif

