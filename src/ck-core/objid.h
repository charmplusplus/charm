#ifndef OBJID_H
#define OBJID_H

#include "charm.h"
#include "converse.h"
#include "pup.h"

#define CMK_OBJID_NULL_TAG 0
#define CMK_OBJID_ARRAY_TAG 1
#define CMK_OBJID_SECTION_TAG 2
#define CMK_OBJID_GROUP_TAG 3
#define CMK_OBJID_NODEGROUP_TAG 4
#define CMK_OBJID_SINGLETON_TAG 5

// The default 64-bit ID layout is: 21 collection bits + 24 home bits + 16 element bits + 3 type tag bits.
// Users can override the number of bits for collections using -DCMK_OBJID_COLLECTION_BITS=N.
// Users can override the number of home bits using -DCMK_OBJID_HOME_BITS=N.
// The element bits trade-off with the home and collection bits, while the type tag bits remain constant.

// TODO: Collection bits are not necessarily universal for 64bit IDs. They are only relevant to
// arrays, groups, and node groups.
#ifndef CMK_OBJID_COLLECTION_BITS
#define CMK_OBJID_COLLECTION_BITS 21
#endif

// TODO: This can be determined at runtime for most cases. Special cases are shrink expand and checkpoint restart.
// Home bits allow the home to be stored directly in the ID. This is necessary for location
// management so that anyone in the system knows where to go to get location information
// about an ID that was previously unknown.
#ifndef CMK_OBJID_HOME_BITS
#define CMK_OBJID_HOME_BITS 24
#endif

#define CMK_OBJID_TYPE_TAG_BITS   3
#define CMK_OBJID_ELEMENT_BITS    (64 - CMK_OBJID_HOME_BITS - CMK_OBJID_COLLECTION_BITS - CMK_OBJID_TYPE_TAG_BITS)

// Sanity checks:
static_assert(CMK_OBJID_COLLECTION_BITS > 0,
              "CMK_OBJID_COLLECTION_BITS must be greater than 0!");
static_assert(CMK_OBJID_COLLECTION_BITS < (64 - CMK_OBJID_TYPE_TAG_BITS),
              "CMK_OBJID_COLLECTION_BITS must be less than (64 - CMK_OBJID_TYPE_TAG_BITS)!");
static_assert((CMK_OBJID_COLLECTION_BITS + CMK_OBJID_ELEMENT_BITS + CMK_OBJID_HOME_BITS + CMK_OBJID_TYPE_TAG_BITS) == 64,
              "The total number of collection + element + pe + type tag bits must be 64!");

// TODO: Home may not always be directly correlated to PE. For now though, home is always a PE.
//static_assert(((1ULL << CMK_OBJID_HOME_BITS) - 1) <= CkNumPes(),
//              "The total number of home bits is not enough for the number of PEs being run on!");

namespace ck {

CmiUInt8 createArrayID(CkGroupID aid, int home, CmiUInt8 eid)
{
  return (CMK_OBJID_ARRAY_TAG << (64 - CMK_OBJID_TYPE_TAG_BITS) |
         (CmiUInt8)gid.idx << (CMK_OBJID_HOME_BITS + CMK_OBJID_ELEMENT_BITS) |
          home << CMK_OBJID_ELEMENT_BITS |
          eid);
}

CmiUInt8 createSectionID(int home, CmiUInt8 sid)
{
  return (CMK_OBJID_SECTION_TAG << (64 - CMK_OBJID_TYPE_TAG_BITS) |
          home << (64 - (CMK_OBJID_TYPE_TAG_BITS + CMK_OBJID_HOME_BITS)) |
          sid);
}

/**
 * The basic element identifier
 */
class ObjID {
    /// @note: may have to befriend the ArrayMgr
    public:
        ObjID(): id(0) {}
        ///
        ObjID(const CmiUInt8 id_) : id(id_) { }
        ObjID(const CkGroupID gid, const CmiUInt8 eid)
            : id( ((CmiUInt8)gid.idx << (HOME_BITS + ELEMENT_BITS)) | eid)
        {
            if ( (CmiUInt8)gid.idx > (COLLECTION_MASK >> (HOME_BITS + ELEMENT_BITS)) ) {
              CmiPrintf("\nError> ObjID ran out of collection bits, please try re-building "
                        "Charm++ with a higher number of collection bits using "
                        "-DCMK_OBJID_COLLECTION_BITS=N, such that %d<N<30\n",
                        COLLECTION_BITS);
              // We don't generally recommend collections bits > 30, though it's possible,
              // b/c then ObjID only has < 32 bits for the element ID.
              CmiAbort("Attempting to create too many chare collections!");
            }
            if ( eid > (HOME_MASK | ELEMENT_MASK) ) {
              CmiPrintf("\nError> ObjID ran out of element bits, please try re-building "
                        "Charm++ with a lower number of collection bits using "
                        "-DCMK_OBJID_COLLECTION_BITS=N, such that 3<N<%d\n",
                        COLLECTION_BITS);
              // We don't generally recommend collections bits <= 3 though it's possible
              CmiAbort("Attempting to create too many chare elements!");
            }
        }

        // should tag system be query-able
        // get collection id
        inline CkGroupID getCollectionID() const {
            CkGroupID gid;
            gid.idx = (id & COLLECTION_MASK) >> (HOME_BITS + ELEMENT_BITS);
            return gid;
        }
        inline int getHomeID() const { return (id & HOME_MASK) >> ELEMENT_BITS; }
        /// get element id
        // For now, the element ID is the part used by location management, and consists
        // of the element bits and home bits. This is everything the location management
        // system needs.
        inline CmiUInt8 getElementID() const { return id & (HOME_MASK | ELEMENT_MASK); }
        inline CmiUInt8 getID() const { return id & (COLLECTION_MASK | HOME_MASK | ELEMENT_MASK); }

        enum bits {
          ELEMENT_BITS    = CMK_OBJID_ELEMENT_BITS,
          HOME_BITS       = CMK_OBJID_HOME_BITS,
          COLLECTION_BITS = CMK_OBJID_COLLECTION_BITS,
          TYPE_TAG_BITS   = CMK_OBJID_TYPE_TAG_BITS
        };
        enum masks : CmiUInt8 {
          ELEMENT_MASK    = ((1ULL << ELEMENT_BITS) - 1),
          HOME_MASK       = (((1ULL << HOME_BITS) - 1) << ELEMENT_BITS),
          COLLECTION_MASK = (((1ULL << COLLECTION_BITS) - 1) << (ELEMENT_BITS + HOME_BITS)),
          TYPE_TAG_MASK   = (((1ULL << TYPE_TAG_BITS) - 1) << (ELEMENT_BITS + HOME_BITS + COLLECTION_BITS))
        };

    private:

        /// The actual id data
        CmiUInt8 id;
};

inline bool operator==(ObjID lhs, ObjID rhs) {
  return lhs.getID() == rhs.getID();
}
inline bool operator!=(ObjID lhs, ObjID rhs) {
  return !(lhs == rhs);
}

} // end namespace ck

PUPbytes(ck::ObjID)
#endif // OBJID_H

