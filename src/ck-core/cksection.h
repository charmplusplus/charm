#ifndef _CKMULTICAST_H
#define _CKMULTICAST_H

#define  MulticastMsg    1

class CkSectionInfo {
public:
  CkArrayID aid;
  union section_type {
    struct sec_mcast {
      int pe;
      void *val;    // point to mCastCookie
      int redNo;
    }  sCookie;
    int  sectionId;
  } sInfo;
  char  type;
public:
  CkSectionInfo() {
	type = MulticastMsg;
	sInfo.sCookie.pe = -1; 
	sInfo.sCookie.val=NULL;
	sInfo.sCookie.redNo=0;
  }
  CkSectionInfo(int t) {
	type = t;
	switch (type) {
	case MulticastMsg:
	  sInfo.sCookie.pe=-1; sInfo.sCookie.val=NULL; sInfo.sCookie.redNo=0;
	  break;
	default:
	  CmiAssert(0);
	}
  }
  CkSectionInfo(void *p) {
	type = MulticastMsg;
	sInfo.sCookie.pe = CkMyPe(); 
	sInfo.sCookie.val=p;
	sInfo.sCookie.redNo=0;
  }
  CkSectionInfo(int e, void *p, int r) {
	type = MulticastMsg;
	sInfo.sCookie.pe = e; 
	sInfo.sCookie.val=p;
	sInfo.sCookie.redNo=r;
  }
  int &get_pe() { CmiAssert(type==MulticastMsg); return sInfo.sCookie.pe; }
  int &get_redNo() {CmiAssert(type==MulticastMsg); return sInfo.sCookie.redNo; }
  void * &get_val() { CmiAssert(type==MulticastMsg); return sInfo.sCookie.val; }
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


#endif
