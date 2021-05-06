#ifndef OBJID_H
#define OBJID_H

#include "charm.h"
#include "converse.h"
#include "pup.h"

#define CMK_OBJID_NULL_TAG 0ULL
#define CMK_OBJID_ARRAY_TAG 1ULL
#define CMK_OBJID_SECTION_TAG 2ULL
#define CMK_OBJID_GROUP_TAG 3ULL
#define CMK_OBJID_NODEGROUP_TAG 4ULL
#define CMK_OBJID_SINGLETON_TAG 5ULL

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

class BaseID
{
  protected:
    constexpr static CmiUInt8 tag_bits = 3;
    constexpr static CmiUInt8 home_bits = 24;

    constexpr static CmiUInt8 tag_offset = 64 - tag_bits;
    constexpr static CmiUInt8 home_offset = 64 - (tag_bits + home_bits);

    constexpr static CmiUInt8 tag_mask = ((1ULL << tag_bits) - 1) << tag_offset;
    constexpr static CmiUInt8 home_mask = ((1ULL << home_bits) - 1) << home_offset;

    CmiUInt8 _id;
    BaseID() = default;
    BaseID(CmiUInt8 id) : _id(id) {}

  public:
    CmiUInt8 id() const
    {
      return _id;
    }

    CmiUInt8 tag() const
    {
      return _id >> tag_offset;
    }
    CmiUInt8 home() const
    {
      return (_id & home_mask) >> home_offset;
    }
};

class SectionID : public BaseID
{
  private:
    constexpr static CmiUInt8 counter_bits = 37;

  public:
    explicit SectionID(CmiUInt8 id) : BaseID(id)
    {
      CkAssert(tag() == CMK_OBJID_SECTION_TAG);
    }
    SectionID(CmiUInt8 home, CmiUInt8 counter)
    {
      _id = CMK_OBJID_SECTION_TAG << tag_offset | home << home_offset | counter;
    }
};

class ArrayElementID : public BaseID
{
  private:
    constexpr static CmiUInt8 collection_bits = 21;
    constexpr static CmiUInt8 elem_bits = 16;

    constexpr static CmiUInt8 collection_offset = elem_bits;
    constexpr static CmiUInt8 elem_offset = 0;

    constexpr static CmiUInt8 collection_mask = ((1ULL << collection_bits) - 1) << collection_offset;
    constexpr static CmiUInt8 elem_mask = (1ULL << elem_bits) - 1;

  public:
    explicit ArrayElementID(CmiUInt8 id) : BaseID(id)
    {
      CkAssert(tag() == CMK_OBJID_ARRAY_TAG);
    }
    ArrayElementID(CmiUInt8 home, CkGroupID aid, CmiUInt8 eid)
    {
      _id = CMK_OBJID_ARRAY_TAG << tag_offset | home << home_offset |
            (CmiUInt8)aid.idx << collection_offset | eid;
    }

    CmiUInt8 elem() const
    {
      return _id & elem_mask;
    }
};

/**
 * The basic element identifier
 */
class ObjID {
    /// @note: may have to befriend the ArrayMgr
    public:
      static CmiUInt8 createArrayID(CkGroupID aid, CmiUInt8 home, CmiUInt8 eid)
      {
        return (CMK_OBJID_ARRAY_TAG << (64 - CMK_OBJID_TYPE_TAG_BITS) |
               (CmiUInt8)aid.idx << (CMK_OBJID_HOME_BITS + CMK_OBJID_ELEMENT_BITS) |
                home << CMK_OBJID_ELEMENT_BITS |
                eid);
      }

      static CmiUInt8 createSectionID(CmiUInt8 home, CmiUInt8 sid)
      {
        return (CMK_OBJID_SECTION_TAG << (64 - CMK_OBJID_TYPE_TAG_BITS) |
                home << (64 - (CMK_OBJID_TYPE_TAG_BITS + CMK_OBJID_HOME_BITS)) |
                sid);
      }

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

