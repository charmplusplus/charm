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
inline int intCompare(void *a, void *b){
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
    
    void initSectionID(CkSectionID *sid);
    
    void processOldSectionMessage(CharmMessageHolder *cmsg);

    ComlibMulticastMsg *getNewMulticastMessage(CharmMessageHolder *cmsg);

    void unpack(envelope *cb_env, CkVec<CkArrayIndexMax> *&destIndices, 
                envelope *&env);

    void localMulticast(envelope *env);

    static inline int getSectionID(CkSectionID id) {
        return id._cookie.sInfo.cInfo.id;
    }
    
};

PUPbytes(ComlibSectionInfo);

#endif
