/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*
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

#define  MulticastMsg          1
#define  COMLIB_MULTICAST_MESSAGE    2

#define COMLIB_MULTICAST_ALL 0
#define COMLIB_MULTICAST_OLD_SECTION 1
#define COMLIB_MULTICAST_NEW_SECTION 2

class CkSectionInfo {
 public:
    CkArrayID aid;
    int pe;
    union section_type {
        struct sec_mcast {    // used for section multicast
            int redNo;
            void *val;        // point to mCastCookie
        } 
        sCookie;

        struct commlibInfo{     // used for commlib
            int  instId;	//the instance of the comm. lib.
            
            // This field indicates local array indices to multicast to:
            // COMLIB_MULTICAST_ALL for all local elements, 
            // COMLIB_MULTICAST_NEW_SECTION, elements are attached 
            // to this message
            // COMLIB_MULTICAST_OLD_SECTION use previously created section
            char status;      
            int id;      //Used to compare section ID's
        } 
        cInfo;

    } sInfo;
    char type;
    
    CkSectionInfo()  {
        type = 0; pe = -1;
        sInfo.sCookie.val=NULL; sInfo.sCookie.redNo=0;
        sInfo.cInfo.instId = 0;
        sInfo.cInfo.status = 0;
        sInfo.cInfo.id     = 0;
    }

    CkSectionInfo(int t) {
	type = t; pe = -1;
	switch (type) {
	case MulticastMsg:
            sInfo.sCookie.val=NULL; 
            sInfo.sCookie.redNo=0;
            break;
        case COMLIB_MULTICAST_MESSAGE:
            sInfo.cInfo.instId=0;
            sInfo.cInfo.status=0;
            sInfo.cInfo.id=0;
            break;
	default:
            CmiAssert(0);
	}
    }
    CkSectionInfo(void *p) {
	type = MulticastMsg;
	pe = CkMyPe(); 
	sInfo.sCookie.val=p;
	sInfo.sCookie.redNo=0;
    }
    CkSectionInfo(int e, void *p, int r) {
	type = MulticastMsg;
	pe = e; 
	sInfo.sCookie.val=p;
	sInfo.sCookie.redNo=r;
    }
    inline int &get_pe() { return pe; }
    inline int &get_redNo() { CmiAssert(type==MulticastMsg); 
                              return sInfo.sCookie.redNo; }
    inline void * &get_val() { CmiAssert(type==MulticastMsg); 
                               return sInfo.sCookie.val; }
};

PUPbytes(CkSectionInfo) //FIXME: write a real pup routine
PUPmarshall(CkSectionInfo)

class CkSectionID {
public:
  CkSectionInfo   _cookie;		// used by array section multicast
  CkArrayIndexMax *_elems;
  int _nElems;
  
  //Two reasons, (i) potentially extend sections to groups. (ii) For
  //array sections these point to the processors (ranks in the
  //commlib) the destinations array elements are on.
  int *pelist;    
  int npes;
  
public:
  CkSectionID(): _elems(NULL), _nElems(0) {}
  CkSectionID(const CkSectionID &sid);
  CkSectionID(const CkArrayID &aid, const CkArrayIndexMax *elems, const int nElems);
  void operator=(const CkSectionID &);
  ~CkSectionID();
  void pup(PUP::er &p);
};
PUPmarshall(CkSectionID)

#define _SECTION_MAGIC     88       /**< multicast magic number for error checking */
/**
 CkMcastBaseMsg is the base class for all multicast message.
*/
class CkMcastBaseMsg {
 public:
  char magic;
  CkArrayID aid; 
  CkSectionInfo _cookie;
  int ep;

 public:
  CkMcastBaseMsg(): magic(_SECTION_MAGIC) {}
  static inline int checkMagic(CkMcastBaseMsg *m) 
      { return m->magic == _SECTION_MAGIC; }
  inline int &gpe(void) { return _cookie.get_pe(); }
  inline int &redno(void) { return _cookie.get_redNo(); }
  inline void *&cookie(void) { return _cookie.get_val(); }
};

#endif
