#ifndef OBJID_H
#define OBJID_H

#include "charm.h"
#include "converse.h"
#include "pup.h"

// The 64-bit object id layout is: 3 type tag bits + COLLECTION bits + PAYLOAD bits,
// most significant first.
//
//   - COLLECTION identifies the chare array (its CkGroupID). The default is 12 bits;
//     override with -DCMK_OBJID_COLLECTION_BITS=N at build time (an AMPI build, which
//     creates one array per virtual rank, needs more).
//   - PAYLOAD identifies the element within the array. Its meaning is decided per
//     array by CkLocMgr, not here: for an array with bounds that fit, it is the
//     packed index; otherwise it is a hash key of the index followed by a number
//     that is unique within the array (see cklocation.h, ck::objid::Layout).
//   - The type tag bits are reserved and unused.
//
// No PE number is stored in the id. The home of an element is computed from the
// id, or from the index, by CkLocMgr (design: doc/objid64-design.md).

#ifndef CMK_OBJID_COLLECTION_BITS
#define CMK_OBJID_COLLECTION_BITS 12
#endif

#define CMK_OBJID_TYPE_TAG_BITS   3
#define CMK_OBJID_PAYLOAD_BITS    (64 - CMK_OBJID_COLLECTION_BITS - CMK_OBJID_TYPE_TAG_BITS)

// Sanity checks:
static_assert(CMK_OBJID_COLLECTION_BITS > 0,
              "CMK_OBJID_COLLECTION_BITS must be greater than 0!");
static_assert(CMK_OBJID_COLLECTION_BITS <= 40,
              "CMK_OBJID_COLLECTION_BITS must leave at least 21 payload bits!");
static_assert((CMK_OBJID_COLLECTION_BITS + CMK_OBJID_PAYLOAD_BITS + CMK_OBJID_TYPE_TAG_BITS) == 64,
              "The total number of collection + payload + type tag bits must be 64!");

namespace ck {

/**
 * The basic element identifier
 */
class ObjID {
    public:
        ObjID(): id(0) {}
        ///
        ObjID(const CmiUInt8 id_) : id(id_) { }
        ObjID(const CkGroupID gid, const CmiUInt8 eid)
            : id( ((CmiUInt8)gid.idx << PAYLOAD_BITS) | eid)
        {
          if ((CmiUInt8)gid.idx > (COLLECTION_MASK >> PAYLOAD_BITS))
          {
            CmiAbort(
                "\nError> ObjID ran out of collection bits: too many chare collections"
                " (gid %" PRIx64 ", limit %" PRIx64 " with %u bits). Rebuild Charm++ with"
                " -DCMK_OBJID_COLLECTION_BITS=N for a larger N.\n",
                (CmiUInt8)gid.idx, (CmiUInt8)(COLLECTION_MASK >> PAYLOAD_BITS),
                COLLECTION_BITS);
          }
          if (eid > PAYLOAD_MASK)
          {
            CmiAbort(
                "\nError> ObjID element payload %" PRIx64 " exceeds %u bits (limit %" PRIx64
                "). Rebuild Charm++ with a smaller -DCMK_OBJID_COLLECTION_BITS=N.\n",
                eid, PAYLOAD_BITS, (CmiUInt8)PAYLOAD_MASK);
          }
        }

        /// The chare array this element belongs to
        inline CkGroupID getCollectionID() const {
            CkGroupID gid;
            gid.idx = (id & COLLECTION_MASK) >> PAYLOAD_BITS;
            return gid;
        }
        /// The element payload: everything location management keys on
        inline CmiUInt8 getElementID() const { return id & PAYLOAD_MASK; }
        inline CmiUInt8 getID() const { return id & (COLLECTION_MASK | PAYLOAD_MASK); }

        enum bits {
          PAYLOAD_BITS    = CMK_OBJID_PAYLOAD_BITS,
          COLLECTION_BITS = CMK_OBJID_COLLECTION_BITS,
          TYPE_TAG_BITS   = CMK_OBJID_TYPE_TAG_BITS
        };
        enum masks : CmiUInt8 {
          PAYLOAD_MASK    = ((1ULL << PAYLOAD_BITS) - 1),
          COLLECTION_MASK = (((1ULL << COLLECTION_BITS) - 1) << PAYLOAD_BITS),
          TYPE_TAG_MASK   = (((1ULL << TYPE_TAG_BITS) - 1) << (PAYLOAD_BITS + COLLECTION_BITS))
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
