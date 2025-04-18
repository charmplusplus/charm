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

  /// Pointer to mCastEntry (used by CkMulticast)
  void *val;
  /// Array/group ID of the array/group that has been sectioned
  CkGroupID aid;
  /// The pe on which this object has been created
  int pe;
  /// Counter tracking the last reduction that has traversed this section (used by CkMulticast)
  int redNo;

  CkSectionInfo() : val(NULL), pe(-1), redNo(0) { }

  CkSectionInfo(CkArrayID _aid, void *p = NULL)
    : val(p), aid(_aid), pe(CkMyPe()), redNo(0) { }

  CkSectionInfo(int e, void *p, int r, CkArrayID _aid)
    : val(p), aid(_aid), pe(e), redNo(r) { }

  bool operator==(const CkSectionInfo &other) const {
    return (val == other.val && aid == other.aid && pe == other.pe && redNo == other.redNo);
  }

  inline int   &get_pe() { return pe; }
  inline int   &get_redNo() { return redNo; }
  inline void  set_redNo(int _redNo) { redNo = _redNo; }
  inline void* &get_val() { return val; }
  inline CkGroupID &get_aid() { return aid; }
  inline CkGroupID get_aid() const { return aid; }
};

PUPbytes(CkSectionInfo)

// ----------- CkMcastBaseMsg -----------

#define _SECTION_MAGIC     88       /* multicast magic number for error checking */

/// CkMcastBaseMsg is the base class for all multicast messages
class CkMcastBaseMsg {
public:
  /// Current info about the state of this section
  CkSectionInfo _cookie;
  unsigned short ep;
#if CMK_ERROR_CHECKING
 private:
  /// A magic number to detect msg corruption
  char magic = _SECTION_MAGIC;
#endif

 public:
  CkMcastBaseMsg() = default;
  static inline bool checkMagic(CkMcastBaseMsg *m) {
#if CMK_ERROR_CHECKING
    return m->magic == _SECTION_MAGIC;
#else
    return true;
#endif
  }
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
  CkSectionID(const CkArrayID &aid, const CkArrayIndex##index *elems, const int nElems, int factor=USE_DEFAULT_BRANCH_FACTOR); \
  CkSectionID(const CkArrayID &aid, const std::vector<CkArrayIndex##index> &elems, int factor=USE_DEFAULT_BRANCH_FACTOR);

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
  std::vector<CkArrayIndex> _elems;
  /**
   * \brief A list of PEs that host section members.
   * - For group sections these point to the PEs in the section
   * - For array sections these point to the processors the array elements are on
   * @note For array sections, currently not saved when pupped across processors
   */
  std::vector<int> pelist;
  /// Branching factor in the spanning tree, can be negative
  int bfactor;

  CkSectionID(): bfactor(USE_DEFAULT_BRANCH_FACTOR) {}
  CkSectionID(const CkSectionID &sid);
  CkSectionID(CkSectionInfo &c, const CkArrayIndex *e, int n, const int *_pelist, int _npes,
              int factor=USE_DEFAULT_BRANCH_FACTOR): _cookie(c), bfactor(factor)
  {
    _elems.assign(e, e+n);
    pelist.assign(_pelist, _pelist+_npes);
  }
  CkSectionID(CkSectionInfo &c, const std::vector<CkArrayIndex>& e, const std::vector<int>& _pelist,
              int factor=USE_DEFAULT_BRANCH_FACTOR): _cookie(c), _elems(e),
              pelist(_pelist), bfactor(factor)  {}
  CkSectionID(const CkGroupID &gid, const int *_pelist, const int _npes,
              int factor=USE_DEFAULT_BRANCH_FACTOR);
  CkSectionID(const CkGroupID &gid, const std::vector<int>& _pelist,
              int factor=USE_DEFAULT_BRANCH_FACTOR);
  CKSECTIONID_CONSTRUCTOR(1D)
  CKSECTIONID_CONSTRUCTOR(2D)
  CKSECTIONID_CONSTRUCTOR(3D)
  CKSECTIONID_CONSTRUCTOR(4D)
  CKSECTIONID_CONSTRUCTOR(5D)
  CKSECTIONID_CONSTRUCTOR(6D)
  CKSECTIONID_CONSTRUCTOR(Max)

  inline CkGroupID get_aid() const { return _cookie.get_aid(); }
  inline int nElems() const { return _elems.size(); }
  inline int nPes() const { return pelist.size(); }
  void operator=(const CkSectionID &);
  ~CkSectionID() = default;
  void pup(PUP::er &p);
};

#endif
