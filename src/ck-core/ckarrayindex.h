#ifndef CKARRAYINDEX_H
#define CKARRAYINDEX_H

#include "pup.h"
#include "ckhashtable.h"
#include "charm.h"

/// Max number of integers in an array index
#ifndef CK_ARRAYINDEX_MAXLEN
    #define CK_ARRAYINDEX_MAXLEN 3
#endif

#ifndef CK_ARRAYLISTENER_MAXLEN
# define CK_ARRAYLISTENER_MAXLEN 3
#endif

/** @warning: fwd declaration of child class to support crazy ptr cast
 */
class CkArrayIndex;


/**
 * Base class for array index objects used in charm.
 *
 * An array index is just a hash key; a run of integers
 * used to look up an object in a hash table.
 *
 * @note: Should define *all* data members that make up an index object.
 * @warning: Do not instantiate! Always create and use a child class
 * @warning: Do not add constructors / destructors. Class participates in unions
 *
 * @note: Should be completely invisible to most client code except those that directly
 * need to put index objects in unions. This happens only in a few entities in the charm
 * codebase and should not happen at all in user codes.
 */
struct CkArrayIndexBase
{
    public:
        ///Length of index in *integers*
        short int nInts;
        ///Number of dimensions in this index, not valid for user-defined indices
        short int dimension;
        /// The actual index data
        union {
            int index[CK_ARRAYINDEX_MAXLEN];
            short int indexShorts[2 * CK_ARRAYINDEX_MAXLEN];
        };

        /// Obtain usable object from base object. @warning: Dangerous pointer cast to child class!!!
        inline CkArrayIndex& asChild() const { return *(CkArrayIndex*)this; }

        /// Permit serialization
        void pup(PUP::er &p)
        {
            p|nInts;
            p|dimension;
            for (int i=0;i<nInts;i++) p|index[i];
        }
};



/**
 * Actual array index class intended for regular use
 *
 * @warning: Put all data members in base class or they may not be transmitted
 * in envelopes or callbacks (basically in any entity that stores indices in a
 * union). Only add behaviors to this class.
 */
class CkArrayIndex: public CkArrayIndexBase
{
    public:
        /// Default
        CkArrayIndex() { nInts=0; dimension=0; for (int i=0; i<CK_ARRAYINDEX_MAXLEN; i++) index[i] = 0; }

	CkArrayIndex(int idx) {init(1,1,idx);};

        /// Return a pointer to the actual index data
        int *data(void)             {return index; }
        /// Return a const pointer to the actual index data
        const int *data(void) const {return index; }

        /// Return the total number of elements (assuming a dense chare array)
        int getCombinedCount(void) const
        {
            if      (dimension == 1) return data()[0];
            else if (dimension == 2) return data()[0] * data()[1];
            else if (dimension == 3) return data()[0] * data()[1] * data()[2];
            else return 0;
        }

        /// Used for debug prints elsewhere
        void print() { CmiPrintf("%d: %d %d %d\n", nInts, index[0], index[1], index[2]); }

        /// Equality comparison
        CmiBool operator==(const CkArrayIndex& idx) const
        {
            if (nInts != idx.nInts) return CmiFalse;
            for (int i=0; i<nInts; i++)
                if (index[i] != idx.index[i]) return CmiFalse;
            return CmiTrue;
        }

        /// These routines allow CkArrayIndex to be used in a CkHashtableT
        inline CkHashCode hash(void) const
        {
            register int i;
            register const int *d=data();
            register CkHashCode ret=d[0];
            for (i=1;i<nInts;i++)
                ret +=circleShift(d[i],10+11*i)+circleShift(d[i],9+7*i);
            return ret;
        }
        ///
        static CkHashCode staticHash(const void *a,size_t) { return ((const CkArrayIndex *)a)->hash(); }
        ///
        inline int compare(const CkArrayIndex &idx) const { return (idx == *this); }
        ///
        static int staticCompare(const void *a,const void *b, size_t)
        { return (*(const CkArrayIndex *)a == *(const CkArrayIndex *)b); }

        /**
         * @note: input arrayID is ignored
         * @todo: Chee Wai Lee had a FIXME note attached to this method because he
         * felt it was a temporary solution
         */
        CmiObjId *getProjectionID(int arrayID)
        {
            CmiObjId *ret = new CmiObjId;
            int i;
            const int *data=this->data();
            if (OBJ_ID_SZ>=this->nInts)
            {
                for (i=0;i<this->nInts;i++)
                    ret->id[i]=data[i];
                for (i=this->nInts;i<OBJ_ID_SZ;i++)
                    ret->id[i]=0;
            }
            else
            {
                //Must hash array index into LBObjid
                int j;
                for (j=0;j<OBJ_ID_SZ;j++)
                    ret->id[j]=data[j];
                for (i=0;i<this->nInts;i++)
                    for (j=0;j<OBJ_ID_SZ;j++)
                        ret->id[j]+=circleShift(data[i],22+11*i*(j+1))+
                            circleShift(data[i],21-9*i*(j+1));
            }
            return ret;
        }

    protected:
        inline void init(const short num, const short dims, const int x, const int y=0, const int z=0)
        {
            nInts = num;
            dimension = dims;
            index[0] = x;
            index[1] = y;
            index[2] = z;
            for (int i=3; i < CK_ARRAYINDEX_MAXLEN; i++)
                index[i] = 0;
        }

        inline void init(const short num, const short dims,
                         const short u, const short v, const short w,
                         const short x, const short y=0, const short z=0)
        {
            nInts = num;
            dimension = dims;
            indexShorts[0] = u;
            indexShorts[1] = v;
            indexShorts[2] = w;
            indexShorts[3] = x;
            indexShorts[4] = y;
            indexShorts[5] = z;
            for (int i=6; i < 2 * CK_ARRAYINDEX_MAXLEN; i++)
                indexShorts[i] = 0;
        }
};


/**
 * Support applications and other charm codes that still use the (now dead)
 * CkArrayIndexMax class to manipulate array indices. All the functionality is
 * now incorporated into the CkArrayIndex base class itself.
 *
 * It is recommended that newer code directly use the base class when there is
 * need to handle an array index.
 *
 * @todo: After at least one minor release announcing the deprecation,
 * CkArrayIndexMax should no longer be supported.
 */
typedef CkArrayIndex CkArrayIndexMax;

class CkArray;

class CkArrayID {
	CkGroupID _gid;
public:
	CkArrayID() : _gid() { }
	CkArrayID(CkGroupID g) :_gid(g) {}
	inline void setZero(void) {_gid.setZero();}
	inline int isZero(void) const {return _gid.isZero();}
	operator CkGroupID() const {return _gid;}
	CkArray *ckLocalBranch(void) const
		{ return (CkArray *)CkLocalBranch(_gid); }
	static CkArray *CkLocalBranch(CkArrayID id)
		{ return (CkArray *)::CkLocalBranch(id); }
	void pup(PUP::er &p) {p | _gid; }
	int operator == (const CkArrayID& other) const {
		return (_gid == other._gid);
	}
    friend bool operator< (const CkArrayID &lhs, const CkArrayID &rhs) {
        return (lhs._gid < rhs._gid);
    }
};
PUPmarshall(CkArrayID)

#endif // CKARRAYINDEX_H

