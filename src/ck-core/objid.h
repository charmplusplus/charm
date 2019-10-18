#ifndef OBJID_H
#define OBJID_H

#include "charm.h"
#include "converse.h"
#include "pup.h"

// The default 64-bit ID layout is: 21 collection bits + 40 element bits + 3 type tag bits.
// Users can override the number of bits for collections using -DCMK_OBJID_COLLECTION_BITS=N.
// The element bits trade-off with the collection bits, while the type tag bits remain constant.
#ifndef CMK_OBJID_COLLECTION_BITS
#define CMK_OBJID_COLLECTION_BITS 21
#endif

#define CMK_OBJID_TYPE_TAG_BITS   3
#define CMK_OBJID_ELEMENT_BITS    (64 - CMK_OBJID_COLLECTION_BITS - CMK_OBJID_TYPE_TAG_BITS)

// Sanity checks:
static_assert(CMK_OBJID_COLLECTION_BITS > 0,
              "CMK_OBJID_COLLECTION_BITS must be greater than 0!");
static_assert(CMK_OBJID_COLLECTION_BITS < (64 - CMK_OBJID_TYPE_TAG_BITS),
              "CMK_OBJID_COLLECTION_BITS must be less than (64 - CMK_OBJID_TYPE_TAG_BITS)!");
static_assert((CMK_OBJID_COLLECTION_BITS + CMK_OBJID_ELEMENT_BITS + CMK_OBJID_TYPE_TAG_BITS) == 64,
              "The total number of collection + element + type tag bits must be 64!");

namespace ck {

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
            : id( ((CmiUInt8)gid.idx << ELEMENT_BITS) | eid)
        {
            if ( (CmiUInt8)gid.idx > (COLLECTION_MASK >> ELEMENT_BITS) ) {
              CmiPrintf("\nError> ObjID ran out of collection bits, please try re-building "
                        "Charm++ with a higher number of collection bits using "
                        "-DCMK_OBJID_COLLECTION_BITS=N, such that %d<N<30\n",
                        COLLECTION_BITS);
              // We don't generally recommend collections bits > 30, though it's possible,
              // b/c then ObjID only has < 32 bits for the element ID.
              CmiAbort("Attempting to create too many chare collections!");
            }
            if ( eid > ELEMENT_MASK ) {
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
            gid.idx = (id & COLLECTION_MASK) >> ELEMENT_BITS;
            return gid;
        }
        /// get element id
        inline CmiUInt8 getElementID() const { return id & ELEMENT_MASK; }
        inline CmiUInt8 getID() const { return id & (COLLECTION_MASK | ELEMENT_MASK); }

        enum bits {
          ELEMENT_BITS    = CMK_OBJID_ELEMENT_BITS,
          COLLECTION_BITS = CMK_OBJID_COLLECTION_BITS,
          TYPE_TAG_BITS   = CMK_OBJID_TYPE_TAG_BITS
        };
        enum masks : CmiUInt8 {
          ELEMENT_MASK =   ((1ULL << ELEMENT_BITS) - 1),
          COLLECTION_MASK = (((1ULL << COLLECTION_BITS) - 1) << ELEMENT_BITS),
          TYPE_TAG_MASK =   (((1ULL << TYPE_TAG_BITS) - 1) << (ELEMENT_BITS + COLLECTION_BITS))
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

