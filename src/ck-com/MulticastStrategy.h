/**
   @addtogroup ComlibCharmStrategy
   @{
   
   @file 
   @brief Contains the abstract parent class for all multicast strategies
   
*/

#ifndef MULTICAST_STRATEGY
#define MULTICAST_STRATEGY

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

/**
 * Abstract parent class for multicast strategies. All multicast strategies mush inherit
 * from this.
 *
 * The definition of the array section, as well as the location of all the elements in
 * the processors is determined by the sending processor. All recipient processors will 
 * deliver messages locally to any array elements whose destination PE in the message is
 * the local PE. If the element is not located locally, the array manager will handle 
 * the delivery to the actual location of the element.
 *
 */
class MulticastStrategy: public Strategy, public CharmStrategy {
 protected:
    /// Helper that generates unique ids for successive uses of the strategy.
    ComlibSectionInfo sinfo;

    /// Are the array sections used repeatedly? 
    /// If so, the strategies could opitimize behavior by storing persistent information in sec_ht
    int isPersistent; 
    
    /// A container to hold persistent information about the previously used array sections. 
    /// The strategy could refer to information here instead of decoding the section info
    /// from the message.
    CkHashtableT<ComlibSectionHashKey, ComlibSectionHashObject *> sec_ht; 
    
    /// Add this section to the hash table locally.
    /// This used to be a void function, but to maintain some sanity checking, it now returns the object
    ComlibSectionHashObject * insertSectionID(CkSectionID *sid, int npes, ComlibMulticastIndexCount *pelist);

    ///Called when a new section multicast is called by the user locally.
    ///The strategy should then create a topology for it and return a hash
    ///object to store that topology.
    virtual void createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist)=0;

    /**   
     * Similar to createHashObjectOnSrcPe, but that this call is made on the
     * destination or intermediate processor. It receives all the information in
     * the parameters, and it does not use ComlibLastKnown, since in some cases
     * it can be incoherent.

     * @param nindices number of local elements to multicast
     * @param idxlist list of local elements
     * @param npes number of processors involved in the multicast
     * @param counts list of all the processors involved int the multicast
     * @param srcpe processor which started the multicast
     * @return a hash object describing the section
     */
    virtual void createObjectOnIntermediatePe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *counts, int srcpe)=0;
        
    /// Needed for getNewMulticastMessage, to specify if the list of processors
    /// need to be ordered. By default it doesn't.
    virtual int needSorting() { return 0; }

    /// Called to multicast the message to local array elements.
    virtual void localMulticast(envelope *env, ComlibSectionHashObject *obj, CkMcastBaseMsg *msg);
    
    /// Called to send to message out to the remote destinations.
    /// This method can be overridden to call converse level strategies.
    virtual void remoteMulticast(envelope *env, ComlibSectionHashObject *obj);

    /// This function is called when a multicast is received with a new section
    /// definition. Process the new message by extracting the array elements
    /// from it and creating a new hash object by calling
    /// createObjectOnIntermediatePe(). Objects are deleting from the hashtable
    /// only during spring cleaning.
    void handleNewMulticastMessage(envelope *env);

 public:

    MulticastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m){}
    
    ///Array constructor
    MulticastStrategy(int isPersistent = 0);
    
    //Destuctor
    ~MulticastStrategy();

    void insertMessage(MessageHolder *msg) {insertMessage((CharmMessageHolder*)msg);}
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();

    ///Called by the converse handler function
    void handleMessage(void *msg);    

    virtual void pup(PUP::er &p);
    
    PUPable_abstract(MulticastStrategy);
};
#endif

/*@}*/
