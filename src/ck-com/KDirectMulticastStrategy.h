/* #ifdef filippo */

/* /\*****************  DISCLAMER ********************* */
/*  * This class is old and not compatible. Deprecated */
/*  **************************************************\/ */

/* #ifndef KDIRECT_MULTICAST_STRATEGY */
/* #define KDIRECT_MULTICAST_STRATEGY */

/* #include "DirectMulticastStrategy.h" */

/* class KDirectHashObject{ */
/*  public: */
/*     CkVec<CkArrayIndex> indices; */
/*     int npes; */
/*     int *pelist; */
/* }; */


/* class KDirectMulticastStrategy: public DirectMultcastStrategy { */
/*  protected: */
/*     int kfactor; */

/*     //Initialize and cache information in a section id which can be */
/*     //used the next time the section is multicast to. */
/*     virtual void initSectionID(CkSectionID *sid); */
    
/*     //Common Initializer for group and array constructors */
/*     //Every substrategy should implement its own */
/*     void commonKDirectInit(); */
    
/*     //Create a new multicast message with the array section in it */
/*     ComlibMulticastMsg * getNewMulticastMessage(CharmMessageHolder *cmsg); */

/*  public: */
    
/*     //Group constructor */
/*     KDirectMulticastStrategy(int ndestpes = 0, int *destpelist = 0);     */

/*     //Array constructor */
/*     KDirectMulticastStrategy(CkArrayID aid); */

/*     KDirectMulticastStrategy(CkMigrateMessage *m): Strategy(m){} */
    
/*     //    virtual void insertMessage(CharmMessageHolder *msg); */
/*     virtual void doneInserting(); */

/*     //Called by the converse handler function */
/*     virtual void handleMulticastMessage(void *msg); */
    
/*     //virtual void beginProcessing(int nelements); */
    
/*     void setKFactor(int k){ kfactor = k; } */
    
/*     virtual void pup(PUP::er &p);     */
/*     PUPable_decl(KDirectMulticastStrategy); */
/* }; */

/* #endif */

/* #endif */
