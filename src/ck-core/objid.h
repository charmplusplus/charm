#ifndef OBJID_H
#define OBJID_H

#include "charm.h"
#include "converse.h"
#include "pup.h"

namespace ck {

#define ELEMENT_BITS    48
#define COLLECTION_BITS 13
#define TYPE_TAG_BITS    3

#if (ELEMENT_BITS+COLLECTION_BITS+TYPE_TAG_BITS) != 64
#error "Object ID is broken"
#endif

#define ELEMENT_MASK    ((1UL << ELEMENT_BITS) - 1)
#define COLLECTION_MASK (((1UL << COLLECTION_BITS) - 1) << ELEMENT_BITS)
#define TYPE_TAG_MASK   (((1UL << TYPE_TAG_BITS) - 1) << (ELEMENT_BITS + COLLECTION_BITS))

/**
 * The basic element identifier
 */
class ObjID {
    /// @note: may have to befriend the ArrayMgr
    public:
        ObjID(): id(0) {}
        // should tag system be query-able
        // get collection id
        inline CkGroupID getCollectionID() const {
            CkGroupID gid;
            gid.idx = (id & COLLECTION_MASK) >> ELEMENT_BITS;
            return gid;
        }
    private:
        ///
        ObjID(const CkGroupID gid, const CmiUInt8 eid)
            : id( ((CmiUInt8)gid.idx << ELEMENT_BITS) | eid)
        {
            CmiAssert( gid.idx <= (COLLECTION_MASK >> ELEMENT_BITS) );
            CmiAssert( eid <= ELEMENT_MASK );
        }


        /// get element id
        inline CmiUInt8 getElementID() const { return id & ELEMENT_MASK; }
        /// The actual id data
        CmiUInt8 id;
};

// Undef all the macros used here to avoid leaking them
#undef ELEMENT_BITS
#undef COLLECTION_BITS
#undef TYPE_TAG_BITS
#undef ELEMENT_MASK
#undef COLLECTION_MASK
#undef TYPE_TAG_MASK

} // end namespace ck

PUPbytes(ck::ObjID)
#endif // OBJID_H

