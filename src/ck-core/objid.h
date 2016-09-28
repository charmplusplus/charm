#ifndef OBJID_H
#define OBJID_H

#include "charm.h"
#include "converse.h"
#include "pup.h"

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
            CmiAssert( (CmiUInt8)gid.idx <= (COLLECTION_MASK >> ELEMENT_BITS) );
            CmiAssert( eid <= ELEMENT_MASK );
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
          ELEMENT_BITS    = 48,
          COLLECTION_BITS = 13,
          TYPE_TAG_BITS   = 3
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

