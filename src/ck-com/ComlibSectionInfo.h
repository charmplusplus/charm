/**
   @addtogroup CharmComlib
   @{
   @file
   
   @brief Utility classes to handle sections in comlib where they are needed
   (basically multicast strategies).
*/

#ifndef COMLIB_SECTION_INFO
#define COMLIB_SECTION_INFO

/***********
  Helper classes that help strategies manage array sections 
***********/

/** Hash key that lets a strategy access a section id data structure
    given the source processor and the MaxSectionId on that processor
*/
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

/*********** CkHashTable functions ******************/
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

/**For calls to qsort*/
inline int intCompare(const void *a, const void *b){
    int a1 = *(int *) a;
    int b1 = *(int *) b;

    if(a1 < b1)
        return -1;
    
    if(a1 == b1)
        return 0;

    if(a1 > b1)
        return 1;

    return 0;
}    

//ComlibSectionHashKey CODE
inline int ComlibSectionHashKey::staticCompare(const void *k1, const void *k2, 
                                                size_t ){
    return ((const ComlibSectionHashKey *)k1)->
                compare(*(const ComlibSectionHashKey *)k2);
}

inline CkHashCode ComlibSectionHashKey::staticHash(const void *v,size_t){
    return ((const ComlibSectionHashKey *)v)->hash();
}

/*************** End CkHashtable Functions *****************/


/// Holds information about a strategy, in case it is used multiple times
/// All this information is derived from the incoming messages
class ComlibSectionHashObject {
 public:
    //My local indices
    CkVec<CkArrayIndex> indices;
    
    //Other processors to send this message to
    int npes;
    int *pelist;

    void *msg;

    //Flags associated with the section
    int isOld; // 1 if the section is indeed old
    
    ComlibSectionHashObject(): indices(0), isOld(0) {
        npes = 0;
        pelist = NULL;
	msg = NULL;
    }

    ~ComlibSectionHashObject() {
        delete [] pelist;
	//delete msg;
    }
};


/** 
    Helps a communication library strategy manage array sections, creating unique
    identifiers for sections, and parsing messages.

    The class maintains a counter for generating unique ids for each section.
    It also provides utility functions that can extract information such as PE 
    lists from messages.
 */
class ComlibSectionInfo {
 
  /// Maximum section id used so far. Incremented for each new CkSectionID that gets created.
  /// Will be used (along with the source PE) to access a hashtable on remote processors, when
  /// looking up persistent data for the section.
  int MaxSectionID;

 public:

    ComlibSectionInfo() { 
      MaxSectionID = 1;
    }
   
    /// Create a unique identifier for the supplied CkSectionID
    inline void initSectionID(CkSectionID *sid) {
      if (MaxSectionID > 65000) {
        CkAbort("Too many sections allocated, wrapping of ints should be done!\n");
      }
      ComlibPrintf("[%d] ComlibSectionInfo::initSectionID: creating section number %d for proc %d\n",
          CkMyPe(), MaxSectionID, sid->_cookie.get_pe());
      sid->_cookie.info.sInfo.cInfo.id = MaxSectionID ++;
      sid->_cookie.get_pe() = CkMyPe();
    }
    
    void processOldSectionMessage(CharmMessageHolder *cmsg);

    /**
     * Create a new message to be sent to the root processor of this broadcast
     * to tell it that we missed some objects during delivery. This new message
     * is returned, and it contains the same section id contained in the
     * CkMcastBaseMsg passed as parameter.
     */
    CkMcastBaseMsg *getNewDeliveryErrorMsg(CkMcastBaseMsg *base);

    /**
     * Starting from a message to be sent, it generates a new message containing
     * the information about the multicast, together with the message itself.
     * The info about the multicast is contained in the field sec_id of cmsg.
     * The processors will be order by MyPe() if requested.
     * The destination array to lookup is in the envelope of the message.
     */
    ComlibMulticastMsg *getNewMulticastMessage(CharmMessageHolder *cmsg, int needSort, int instanceID);

    /**
     * Given a ComlibMulticastMsg arrived through the network as input (cb_env),
     * separate it into its basic components.

     * destIndeces is the pointer to the first element in this processor (as by
     * the knowledge of the sender); nLocalElems is the count of how many
     * elements are local. env is a new allocated memory containing the user
     * message.
     */
    void unpack(envelope *cb_env, int &nLocalElems, CkArrayIndex *&destIndices, 
                envelope *&env);

    void localMulticast(envelope *env);

    /**
     * Returns the number of remote procs involved in the array index list, the
     * number of elements each proc has, and the mapping between indices and
     * procs.
     
     * @param nindices size of the array idxlist (input)
     * @param idxlist array of indices of the section (input)
     * @param destArrayID array ID to which the indeces refer (input)
     * @param npes number of remote processors involved (output)
     * @param nidx number of indices that are remote (output)
     * @param counts array of associations pe-count: number of elements in proc pe (output, new'ed(CkNumPes()))
     * @param belongs array of integers expressing association of elements with pes: belongs[i] = index in counts of the processor having index i (output, new'ed(nindices))
    */
    void getPeCount(int nindices, CkArrayIndex *idxlist, const CkArrayID &destArrayID,
		    int &npes, int &nidx,
		    ComlibMulticastIndexCount *&counts, int *&belongs);

    void getPeList(envelope *cb_env, int npes, int *&pelist);

    /**
     * Returns the list of remote procs (therefore not including our processor)
     * involved in the array index list.

     * @param nindices size of the array idxlist (input
     * @param idxlist array of indices of the section (input)
     * @param destArrayID array ID to which the indeces refer (input)
     * @param npes number of processors involved (output)
     * @param pelist list of the processors involved (output, new'ed)
     */
    void getRemotePelist(int nindices, CkArrayIndex *idxlist, CkArrayID &destArrayID,
                         int &npes, int *&pelist);

    /** Returns the same list as getRemotePeList, only that it does not exclude
	the local processor from the list if it is involved. */
    void getPeList(int nindices, CkArrayIndex *idxlist, CkArrayID &destArrayID,
                   int &npes, int *&pelist);

    void getLocalIndices(int nindices, CkArrayIndex *idxlist, CkArrayID &destArrayID,
                         CkVec<CkArrayIndex> &idx_vec);   
        
    void getNodeLocalIndices(int nindices, CkArrayIndex *idxlist, CkArrayID &destArrayID,
                         CkVec<CkArrayIndex> &idx_vec);   
        
    void pup(PUP::er &p) {
        p | MaxSectionID;
    }
};

#endif
