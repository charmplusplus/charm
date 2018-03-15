/**
@file 
 
  These classes implement array (or group) sections which are a subset of a
  CkArray (or CkGroup).
  Supported operations include section multicast and reduction.
  It is currently implemented using delegation.

  Extracted from charm++.h into separate file on 6/22/2003 by
  Gengbin Zheng.

*/

#ifndef _CKSECTION_H
#define _CKSECTION_H

#include "charm.h"
#include "ckarrayindex.h"

// ----------- CkSectionInfo -----------

/**
 * Contains array/group ID for a section, and state information for CkMulticast
 * to manage the section. This object is also referred to as the section "cookie".
 */
class CkSectionInfo {
public:

  /**
   * For now we still need to encapsulate CkSectionInfo's data
   * in a separate CkSectionInfoStruct because it is used in ckcallback
   * inside a union, and C++03 doesn't support placing objects with non-trivial
   * constructors in unions.
   * TODO When all our supported compilers have C++11 we might want to modify callbackData
   * union to allow objects with non-trivial constructors.
   */
  class CkSectionInfoStruct {
  public:
    /// Pointer to mCastEntry (used by CkMulticast)
    void *val;
    /// Array/group ID of the array/group that has been sectioned
    CkGroupID aid;
    /// The pe on which this object has been created
    int pe;
    /// Counter tracking the last reduction that has traversed this section (used by CkMulticast)
    int redNo;
  };

  CkSectionInfoStruct info;

  CkSectionInfo() {
    info.pe = -1;
    info.redNo = 0;
    info.val = NULL;
  }

  CkSectionInfo(const CkSectionInfoStruct &i): info(i) {}

  CkSectionInfo(CkArrayID _aid, void *p = NULL) {
    info.pe = CkMyPe();
    info.aid = _aid;
    info.val = p;
    info.redNo = 0;
  }

  CkSectionInfo(int e, void *p, int r, CkArrayID _aid) {
    info.pe = e;
    info.aid = _aid;
    info.val = p;
    info.redNo = r;
  }

  inline int   &get_pe() { return info.pe; }
  inline int   &get_redNo() { return info.redNo; }
  inline void  set_redNo(int redNo) { info.redNo = redNo; }
  inline void* &get_val() { return info.val; }
  inline CkGroupID &get_aid() { return info.aid; }
  inline CkGroupID get_aid() const { return info.aid; }

};

PUPbytes(CkSectionInfo)
PUPmarshall(CkSectionInfo)

// ----------- CkMcastBaseMsg -----------

#define _SECTION_MAGIC     88       /* multicast magic number for error checking */

/// CkMcastBaseMsg is the base class for all multicast messages
class CkMcastBaseMsg {
public:
  /// Current info about the state of this section
  CkSectionInfo _cookie;
  /// A magic number to detect msg corruption
  char magic;
  unsigned short ep;

  CkMcastBaseMsg(): magic(_SECTION_MAGIC) {}
  static inline bool checkMagic(CkMcastBaseMsg *m) { return m->magic == _SECTION_MAGIC; }
  inline int &gpe(void)      { return _cookie.get_pe(); }
  inline int &redno(void)    { return _cookie.get_redNo(); }
  inline void *&entry(void) { return _cookie.get_val(); }
};

// ----------- CkSectionID -----------

class CkArrayIndex1D;
class CkArrayIndex2D;
class CkArrayIndex3D;
class CkArrayIndex4D;
class CkArrayIndex5D;
class CkArrayIndex6D;

#define CKSECTIONID_CONSTRUCTOR(index) \
  CkSectionID(const CkArrayID &aid, const CkArrayIndex##index *elems, const int nElems, int factor=USE_DEFAULT_BRANCH_FACTOR);

#define USE_DEFAULT_BRANCH_FACTOR 0

/** A class that holds complete info about an array/group section.
 *
 * Describes section members, host PEs, current section status etc.
 */
class CkSectionID {
public:
  /// Minimal section info (cookie)
  CkSectionInfo _cookie;
  /// The list of array indices that are section members (array sections)
  CkArrayIndex *_elems;
  /// The number of section members
  int _nElems;
  /**
   * \brief A list of PEs that host section members.
   * - For group sections these point to the PEs in the section
   * - For array sections these point to the processors the array elements are on
   * @note For array sections, currently not saved when pupped across processors
   */
  int *pelist;
  /// The number of PEs that host section members
  int npes;
  /// Branching factor in the spanning tree, can be negative
  int bfactor;

  CkSectionID(): _elems(NULL), _nElems(0), pelist(NULL), npes(0),
                  bfactor(USE_DEFAULT_BRANCH_FACTOR) {}
  CkSectionID(const CkSectionID &sid);
  CkSectionID(CkSectionInfo &c, CkArrayIndex *e, int n, int *_pelist, int _npes,
              int factor=USE_DEFAULT_BRANCH_FACTOR): _cookie(c), _elems(e),
              _nElems(n), pelist(_pelist), npes(_npes), bfactor(factor)  {}
  CkSectionID(const CkGroupID &gid, const int *_pelist, const int _npes,
              int factor=USE_DEFAULT_BRANCH_FACTOR);
  CKSECTIONID_CONSTRUCTOR(1D)
  CKSECTIONID_CONSTRUCTOR(2D)
  CKSECTIONID_CONSTRUCTOR(3D)
  CKSECTIONID_CONSTRUCTOR(4D)
  CKSECTIONID_CONSTRUCTOR(5D)
  CKSECTIONID_CONSTRUCTOR(6D)
  CKSECTIONID_CONSTRUCTOR(Max)

  inline CkGroupID get_aid() const { return _cookie.get_aid(); }
  void operator=(const CkSectionID &);
  ~CkSectionID() {
    delete [] _elems;
    delete [] pelist;
  }
  void pup(PUP::er &p);
};
PUPmarshall(CkSectionID)

#endif
