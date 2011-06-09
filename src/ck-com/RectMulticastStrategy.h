#ifndef RECT_MULTICAST_STRATEGY
#define RECT_MULTICAST_STRATEGY


/**
   @addtogroup ComlibCharmStrategy
   @{
   @file
   @brief RectMulticastStrategy for BG/L native layer
*/


// only works on BG/L Native Layer

#include "ComlibManager.h"
#include "ComlibSectionInfo.h"

typedef CkHashtableTslow < unsigned int, void *> comRectHashType;
//typedef CkHashtableT < unsigned int, int > comRectHashType;

CkpvExtern(comRectHashType *, com_rect_ptr);


// need place to record 'src in rectangle' state
class ComlibRectSectionHashObject : public ComlibSectionHashObject
{
 public:
  bool sourceInRectangle;
  int cornerRoot;
  CkArrayID aid;
};

void *DMHandler(void *msg);
/**
 * Main class for multicast strategies. It defaults to sending a direct message
 * to all the processors involved in the multicast.

 * The definition of the section, as well as the location of all the elements in
 * the processors is determined by the sending processor. The other will obey to
 * it, even some elements have already migrated elsewhere.
 */
#ifdef CMK_RECT_API
#include "bgml.h"
class RectMulticastStrategy: public Strategy, public CharmStrategy {
 protected:
    int handlerId;    
    ComlibSectionInfo sinfo;

    ///Array section support.
    CkHashtableT<ComlibSectionHashKey, ComlibRectSectionHashObject *> sec_ht; 


    inline unsigned int computeKey(int sectionID, int srcPe, CkArrayID arrId) {
      //      fprintf(stderr,"key %d %d %d = %d\n",(((CkGroupID) arrId).idx), (sectionID), srcPe, (((CkGroupID) arrId).idx << 24) | (sectionID<<16)| srcPe);
      return ((unsigned int) (((CkGroupID) arrId).idx << 24) | (sectionID<<16)| srcPe);}
    
    inline unsigned int computeKey(int sectionID, int srcPe, int id) { 
      //      fprintf(stderr,"key %d %d %d = %d", id ,sectionID,  srcPe, (id  << 24) | (sectionID<<16)| srcPe);
      return ((unsigned int) (id  << 24) | (sectionID<<16)| srcPe);}
    
    ///Add this section to the hash table locally.
    void insertSectionID(CkSectionID *sid);

    ///Called when a new section multicast is called by the user locally.
    ///The strategy should then create a topology for it and return a hash
    ///object to store that topology.
    virtual ComlibRectSectionHashObject *createObjectOnSrcPe(int nindices, CkArrayIndex *idx_list, unsigned int sectionid);

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
    virtual ComlibRectSectionHashObject *createObjectOnIntermediatePe(int nindices, CkArrayIndex *idxlist, int npes, ComlibMulticastIndexCount *counts, int srcpe, int sectionID);
        
    ///Needed for getNewMulticastMessage, to specify if the list of processors need to be ordered
    virtual int needSorting() { return 0; }

    ///Called to multicast the message to local array elements.
    void localMulticast(envelope *env, ComlibRectSectionHashObject *obj);
    
    ///Called to send to message out to the remote destinations.
    ///This method can be overridden to call converse level strategies.
    virtual void remoteMulticast(envelope *env, ComlibRectSectionHashObject *obj);

    ///Called to forward to someone who can do the heavy sending work
    virtual void forwardMulticast(envelope *env, ComlibRectSectionHashObject *obj);


    ///Process a new message by extracting the array elements from it and
    ///creating a new hash object by calling createObjectOnIntermediatePe().
    void handleNewMulticastMessage(envelope *env);

    void sendRectDest(ComlibRectSectionHashObject *obj, int srcpe, envelope *env);
    BGTsRC_Geometry_t *getRectGeometry(ComlibRectSectionHashObject *obj, int srcpe);
    int assignCornerRoot(BGTsRC_Geometry_t *geom, int srcpe);

 public:



    RectMulticastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m){}
                
    ///Array constructor
    RectMulticastStrategy(CkArrayID aid);
        
        
    //Destuctor
    ~RectMulticastStrategy();
        
    virtual void insertMessage(CharmMessageHolder *msg);
    virtual void doneInserting();

    ///Called by the converse handler function
    virtual void handleMessage(void *msg);    
    virtual void handleMessageForward(void *msg);    

    virtual void pup(PUP::er &p);    
    virtual void beginProcessing(int nelements);
    
    PUPable_decl(RectMulticastStrategy);
};
#else
class RectMulticastStrategy : public Strategy, public CharmStrategy {   
  RectMulticastStrategy(CkMigrateMessage *m): Strategy(m), CharmStrategy(m){}
  //    RectMulticastStrategy(CkArrayID aid){}
  ~RectMulticastStrategy(){}
  void insertMessage(MessageHolder*) {}
  void handleMessage(void*) {}
  void pup(PUP::er &p) {}
  PUPable_decl(RectMulticastStrategy);   
};
#endif


/*@}*/

#endif
