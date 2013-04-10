/**
@file 
 
 These classes implement array sections which is a subset of ckarray.
  Supported operation includes section multicast and reduction.
  It is currently implemented using delegation.

  Extracted from charm++.h into separate file on 6/22/2003 by
  Gengbin Zheng.

  Modified to make it suitable for the Communication Library by 
  Sameer Kumar.
*/

#ifndef _CKMULTICAST_H
#define _CKMULTICAST_H

#include "charm.h"
#include "ckarrayindex.h"

#define  MulticastMsg          1
#define  COMLIB_MULTICAST_MESSAGE    2

//#define COMLIB_MULTICAST_ALL 0
#define COMLIB_MULTICAST_OLD_SECTION 1
#define COMLIB_MULTICAST_NEW_SECTION 2
#define COMLIB_MULTICAST_SECTION_ERROR 3

/** Structure that holds info relevant to the use of an array/group section
 */
typedef struct _CkSectionInfoStruct {
    /// The array ID of the array that has been sectioned
    CkGroupID aid; ///< @note: Also used to store a CkGroupID for group multicasts
    /// The pe on which this object has been created
    int pe;
    /// Info needed by the section comm managers
    union section_type {
        // Used when section is delegated to CkMulticast
        struct sec_mcast {
            /// Counter tracking the last reduction that has traversed this section
            int redNo;
            // Pointer to mCastCookie
            void *val;
        } 
        sCookie;

        // Used when section is delegated to Comlib
        struct commlibInfo{
            // The instance of the comm. lib.
            short  instId;
            // This field indicates local array indices to multicast to:
            // (Deprecated) COMLIB_MULTICAST_ALL for all local elements, 
            // COMLIB_MULTICAST_NEW_SECTION, elements are attached 
            // to this message
            // COMLIB_MULTICAST_OLD_SECTION use previously created section
            // COMLIB_MULTICAST_SECTION_ERROR mark the section as old
            short status;      
            // Used to compare section ID's
            int id;
        } 
        cInfo;
    } sInfo;
    // Indicates which library has been delegated the section comm
    char type;
} CkSectionInfoStruct;

class CkSectionInfo {
 public:
    CkSectionInfoStruct  info;
    
    CkSectionInfo()  {
        info.type = 0; info.pe = -1;
        info.sInfo.sCookie.val=NULL; info.sInfo.sCookie.redNo=0;
        info.sInfo.cInfo.instId = 0;
        info.sInfo.cInfo.status = 0;
        info.sInfo.cInfo.id     = 0;
    }

    CkSectionInfo(CkSectionInfoStruct i): info(i) {}

    CkSectionInfo(int t) {
      info.type = t; info.pe = -1;
      switch (info.type) {
      case MulticastMsg:
        info.sInfo.sCookie.val=NULL; 
        info.sInfo.sCookie.redNo=0;
        break;
      case COMLIB_MULTICAST_MESSAGE:
        info.sInfo.cInfo.instId=0;
        info.sInfo.cInfo.status=0;
        info.sInfo.cInfo.id=0;
        break;
      default:
        CmiAssert(0);
      }
    }

    CkSectionInfo(CkArrayID _aid, void *p = NULL) {
      info.type = MulticastMsg;
      info.pe = CkMyPe();
      info.aid = _aid;
      info.sInfo.sCookie.val=p;
      info.sInfo.sCookie.redNo=0;
    }

    CkSectionInfo(int e, void *p, int r, CkArrayID _aid) {
      info.type = MulticastMsg;
      info.pe = e; 
      info.aid = _aid;
      info.sInfo.sCookie.val=p;
      info.sInfo.sCookie.redNo=r;
    }

    inline char  &get_type() { return info.type; }
    inline int   &get_pe()    { return info.pe; }
    inline int   &get_redNo() { CmiAssert(info.type==MulticastMsg); return info.sInfo.sCookie.redNo; }
    inline void  set_redNo(int redNo) { CmiAssert(info.type==MulticastMsg); info.sInfo.sCookie.redNo = redNo; }
    inline void* &get_val()   { CmiAssert(info.type==MulticastMsg); return info.sInfo.sCookie.val; }
    inline CkGroupID   &get_aid()    { return info.aid; }
    inline CkGroupID   get_aid() const   { return info.aid; }

    /*
    void pup(PUP::er &p) {
      p | aid;
      p | pe;
      p | type;
      switch (type) {
      case MulticastMsg:
        p | sInfo.sCookie.redNo;
        p | sInfo.sCookie.val;
        break;
      case COMLIB_MULTICAST_MESSAGE:
        p | sInfo.cInfo.instId;
        p | sInfo.cInfo.status;
        p | sInfo.cInfo.id = 0;
        break;
      }
    }
    */
};

PUPbytes(CkSectionInfo) //FIXME: write a real pup routine
PUPmarshall(CkSectionInfo)



class CkArrayIndex1D;
class CkArrayIndex2D;
class CkArrayIndex3D;
class CkArrayIndex4D;
class CkArrayIndex5D;
class CkArrayIndex6D;

#define _SECTION_MAGIC     88       /* multicast magic number for error checking */

/// CkMcastBaseMsg is the base class for all multicast messages
class CkMcastBaseMsg {
    public:
        // Current info about the state of this section
        CkSectionInfo _cookie;
        // A magic number to detect msg corruption
        char magic;
        unsigned short ep;
        
        CkMcastBaseMsg(): magic(_SECTION_MAGIC) {}
        static inline int checkMagic(CkMcastBaseMsg *m) { return m->magic == _SECTION_MAGIC; }
        inline int &gpe(void)      { return _cookie.get_pe(); }
        inline int &redno(void)    { return _cookie.get_redNo(); }
        inline void *&cookie(void) { return _cookie.get_val(); }
};



#define CKSECTIONID_CONSTRUCTOR(index) \
  CkSectionID(const CkArrayID &aid, const CkArrayIndex##index *elems, const int nElems);

/** A class that holds complete info about an array/group section
 *
 * Describes section members, host PEs, current section status etc.
 */
class CkSectionID {
    public:
        /// Minimal section info
        CkSectionInfo _cookie;
        /// The list of array indices that are section members
        CkArrayIndex *_elems;
        /// The number of section members
        int _nElems;
        /** A list of PEs that host section members
         *
         * @note: Two reasons:
         * (i) potentially extend sections to groups
         * (ii) For array sections these point to the processors
         * (ranks in commlib) the destinations array elements are on
         * @note: Currently not saved when pupped across processors
         */
        int *pelist;
        /// The number of PEs that host section members
        int npes;
        
        CkSectionID(): _elems(NULL), _nElems(0), pelist(0), npes(0) {}
        CkSectionID(const CkSectionID &sid);
        CkSectionID(CkSectionInfo &c, CkArrayIndex *e, int n, int *_pelist, int _npes): _cookie(c), _elems(e), _nElems(n), pelist(_pelist), npes(_npes)  {}
        CkSectionID(const CkGroupID &gid, const int *_pelist, const int _npes);
        CKSECTIONID_CONSTRUCTOR(1D)
        CKSECTIONID_CONSTRUCTOR(2D)
        CKSECTIONID_CONSTRUCTOR(3D)
        CKSECTIONID_CONSTRUCTOR(4D)
        CKSECTIONID_CONSTRUCTOR(5D)
        CKSECTIONID_CONSTRUCTOR(6D)
        CKSECTIONID_CONSTRUCTOR(Max)

        inline int getSectionID(){ return _cookie.info.sInfo.cInfo.id; }
        void operator=(const CkSectionID &);
        ~CkSectionID() {
            if (_elems != NULL) delete [] _elems;
            if (pelist != NULL) delete [] pelist;
        }
        void pup(PUP::er &p);
};
PUPmarshall(CkSectionID)

#endif

