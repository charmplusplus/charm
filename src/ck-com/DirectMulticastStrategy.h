#ifndef DIRECT_MULTICAST_STRATEGY
#define DIRECT_MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

void *DMHandler(void *msg);
/**
 * Main class for multicast strategies. It defaults to sending a direct message
 * to all the processors involved in the multicast.

 * The definition of the section, as well as the location of all the elements in
 * the processors is determined by the sending processor. The other will obey to
 * it, even some elements have already migrated elsewhere.
 */
class DirectMulticastStrategy: public CharmStrategy {
 protected:
    //   int handlerId;    
    ComlibSectionInfo sinfo;

    int isPersistent; 
    
    ///Array section support.
    CkHashtableT<ComlibSectionHashKey, ComlibSectionHashObject *> sec_ht; 
    
    ///Add this section to the hash table locally.
    void insertSectionID(CkSectionID *sid);

    ///Called when a new section multicast is called by the user locally.
    ///The strategy should then create a topology for it and return a hash
    ///object to store that topology.
    virtual ComlibSectionHashObject *createObjectOnSrcPe(int nindices, CkArrayIndexMax *idx_list);

    /**   
     * Similar to createHashObjectOnSrcPe, but that this call is made on the
     * destination or intermediate processor. I receives all the information in
     * the parameters, and it does not use ComlibLastKnown, since in some cases
     * it can be incoherent.

     * @param nindices number of local elements to multicast
     * @param idxlist list of local elements
     * @param npes number of processors involved in the multicast
     * @param counts list of all the processors involved int the multicast
     * @param srcpe processor which started the multicast
     * @return a hash object describing the section
     */
    virtual ComlibSectionHashObject *createObjectOnIntermediatePe(int nindices, CkArrayIndexMax *idxlist, int npes, ComlibMulticastIndexCount *counts, int srcpe);
        
    ///Needed for getNewMulticastMessage, to specify if the list of processors need to be ordered
    virtual int needSorting() { return 0; }

    ///Called to multicast the message to local array elements.
    void localMulticast(envelope *env, ComlibSectionHashObject *obj);
    
    ///Called to send to message out to the remote destinations.
    ///This method can be overridden to call converse level strategies.
    virtual void remoteMulticast(envelope *env, ComlibSectionHashObject *obj);

    ///Process a new message by extracting the array elements from it and
    ///creating a new hash object by calling createObjectOnIntermediatePe().
    void handleNewMulticastMessage(envelope *env);

 public:

    DirectMulticastStrategy(CkMigrateMessage *m): CharmStrategy(m){}
                
    ///Array constructor
    DirectMulticastStrategy(CkArrayID aid, int isPersistent = 0);
        
    //Destuctor
    ~DirectMulticastStrategy();
        
    virtual void insertMessage(CharmMessageHolder *msg);
    virtual void doneInserting();

    ///Called by the converse handler function
    virtual void handleMessage(void *msg);    

    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    
    PUPable_decl(DirectMulticastStrategy);
};
#endif

