#ifndef _CKMULTICAST_H
#define _CKMULTICAST_H

#define  MulticastMsg    1

class CkSectionInfo {
public:
  CkArrayID aid;
  int pe;
  union section_type {
    struct sec_mcast {		// used for section multicast
      int redNo;
      void *val;    // point to mCastCookie
    }  sCookie;
    int  sectionId;		// used for commlib
  } sInfo;
  char  type;
public:
  CkSectionInfo()  {
	type = 0; pe = -1;
	sInfo.sCookie.val=NULL; sInfo.sCookie.redNo=0;
  }
  CkSectionInfo(int t) {
	type = t; pe = -1;
	switch (type) {
	case MulticastMsg:
	  sInfo.sCookie.val=NULL; sInfo.sCookie.redNo=0;
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
  inline int &get_redNo() {CmiAssert(type==MulticastMsg); return sInfo.sCookie.redNo; }
  inline void * &get_val() { CmiAssert(type==MulticastMsg); return sInfo.sCookie.val; }
};
PUPbytes(CkSectionInfo) //FIXME: write a real pup routine
PUPmarshall(CkSectionInfo)

class CkSectionID {
public:
  CkSectionInfo   _cookie;		// used by array section multicast
  CkArrayIndexMax *_elems;
  int _nElems;
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
  static inline int checkMagic(CkMcastBaseMsg *m) { return m->magic == _SECTION_MAGIC; }
  inline int &gpe(void) { return _cookie.get_pe(); }
  inline int &redno(void) { return _cookie.get_redNo(); }
  inline void *&cookie(void) { return _cookie.get_val(); }
};

#endif
