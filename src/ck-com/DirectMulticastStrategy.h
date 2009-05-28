/**
   @addtogroup ComlibCharmStrategy
   @{
   @file 
   @brief Send a multicast by sending once directly to each processor owning destination elements.

*/

#ifndef DIRECT_MULTICAST_STRATEGY
#define DIRECT_MULTICAST_STRATEGY

#include "MulticastStrategy.h"


/** 
  Send the multicast by sending once directly to each processor owning destination elements.

  The definition of the section, as well as the location of all the elements in
  the processors is determined by the sending processor. The other will obey to
  it, even some elements have already migrated elsewhere.

*/
class DirectMulticastStrategy: public MulticastStrategy {
 protected:
 
    ///Called when a new section multicast is called by the user locally.
    ///The strategy should then create a topology for it and return a hash
    ///object to store that topology.
  virtual void createObjectOnSrcPe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *pelist);

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
    virtual void createObjectOnIntermediatePe(ComlibSectionHashObject *obj, int npes, ComlibMulticastIndexCount *counts, int srcpe);
   
 public:
    DirectMulticastStrategy(CkMigrateMessage *m): MulticastStrategy(m){}
                
    ///Array constructor
    DirectMulticastStrategy(int isPersistent = 0): MulticastStrategy(isPersistent) {}

    PUPable_decl(DirectMulticastStrategy);

    virtual void pup(PUP::er &p){ MulticastStrategy::pup(p);}   
    
};
#endif

/*@}*/
