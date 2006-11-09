#ifndef COMLIB_SECTION_INFO
#define COMLIB_SECTION_INFO

/***********
  Helper classes that help strategies manage array sections 
***********/

/* Hash key that lets a strategy access a section id data structure
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

/*For calls to qsort*/
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



class ComlibSectionHashObject {
 public:
    //My local indices
    CkVec<CkArrayIndexMax> indices;
    
    //Other processors to send this message to
    int npes;
    int *pelist;

    void *msg;
    
    ComlibSectionHashObject(): indices(0) {
        npes = 0;
        pelist = NULL;
	msg = NULL;
    }

    ~ComlibSectionHashObject() {
        delete [] pelist;
	//delete msg;
    }
};


/*** Class that helps a communication library strategy manage array
     sections
***************/

class ComlibSectionInfo {
    /* Array ID of the array section */
    CkArrayID destArrayID;
    
    //Unique section id for this section
    //Will be used to access a hashtable on remote processors
    int MaxSectionID;

    //Instance ID of the strategy
    int instanceID;

    CkVec<CkArrayIndexMax> localDestIndexVec;

 public:

    ComlibSectionInfo() { 
        destArrayID.setZero(); MaxSectionID = 1; instanceID = 0;    
    }

    ComlibSectionInfo(CkArrayID dest, int instance){
        destArrayID = dest;
        instanceID = instance;

        MaxSectionID = 1;
    }
    
    inline void initSectionID(CkSectionID *sid) {
        sid->_cookie.sInfo.cInfo.id = MaxSectionID ++;            
    }

    inline int getInstID(){ return(instanceID);}
    
    void processOldSectionMessage(CharmMessageHolder *cmsg);

    /**
     * Starting from a message to be sent, it generates a new message containing
     * the information about the multicast, together with the message itself.
     * The info about the multicast is contained in the field sec_id of cmsg.
     * The processors will be order by MyPe() if requested.
     */
    ComlibMulticastMsg *getNewMulticastMessage(CharmMessageHolder *cmsg, int needSort);

    /**
     * Given a ComlibMulticastMsg arrived through the network as input (cb_env),
     * separate it into its basic components.

     * destIndeces is the pointer to the first element in this processor (as by
     * the knowledge of the sender); nLocalElems is the count of how many
     * elements are local. env is a new allocated memory containing the user
     * message.
     */
    void unpack(envelope *cb_env, int &nLocalElems, CkArrayIndexMax *&destIndices, 
                envelope *&env);

    void localMulticast(envelope *env);

    /**
     * Returns the number of remote procs involved in the array index list, the
     * number of elements each proc has, and the mapping between indices and
     * procs.
     
     * @param nindices size of the array idxlist (input)
     * @param idxlist array of indeces of the section (input)
     * @param npes number of processors involved (output)
     * @param nidx number of indices that are remote (output)
     * @param counts array of associations pe-count: number of elements in proc pe (output, new'ed(CkNumPes()))
     * @param belongs array of integers expressing association of elements with pes: belongs[i] = index in counts of the processor having index i (output, new'ed(nidx))
    */
    void getPeCount(int nindices, CkArrayIndexMax *idxlist, 
		    int &npes, int &nidx,
		    ComlibMulticastIndexCount *&counts, int *&belongs);

    void getPeList(envelope *cb_env, int npes, int *&pelist);

    void getRemotePelist(int nindices, CkArrayIndexMax *idxlist, 
                         int &npes, int *&pelist);

    void getPeList(int nindices, CkArrayIndexMax *idxlist, 
                   int &npes, int *&pelist);

    void getLocalIndices(int nindices, CkArrayIndexMax *idxlist, 
                         CkVec<CkArrayIndexMax> &idx_vec);   
        
    static inline int getSectionID(CkSectionID id) {
        return id._cookie.sInfo.cInfo.id;
    }

    inline CkArrayID getDestArrayID() {
        return destArrayID;
    }

    void pup(PUP::er &p) {
        p | destArrayID;
        p | MaxSectionID;
        p | instanceID;
        p | localDestIndexVec;
    }
};

#endif
