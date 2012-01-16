// emacs mode line -*- mode: c++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MSA_DISTARRAY_H
#define MSA_DISTARRAY_H

#include <utility>
#include <algorithm>
#include "msa-DistPageMgr.h"

namespace MSA {

static const int DEFAULT_SYNC_SINGLE = 0;
static const bool MSA_CLEAR_ALL = true;

struct MSA_InvalidHandle { };

template <typename ENTRY>
class Writable
{
    ENTRY &e;

public:
    Writable(ENTRY &e_) : e(e_) {}
    inline const ENTRY& operator= (const ENTRY& rhs) { e = rhs; return rhs; }
};

template <class MSA>
class Accumulable
{
    typedef typename MSA::T ENTRY;
    ENTRY &e;

public:
    Accumulable(ENTRY &e_) : e(e_) {}
    template<typename T>
    inline void operator+=(const T &rhs)
    { MSA::OPS::accumulate(e, rhs); }
    template<typename T>
    inline void accumulate(const T& rhs)
    { MSA::OPS::accumulate(e, rhs); }
};

template<class MSA> class MSARead;
template<class MSA> class MSAWrite;
template<class MSA> class MSAAccum;

template<class MSA>
class MSAHandle
{
protected:
    MSA *msa;
    bool valid;

    inline void checkInvalidate()
    {
#if CMK_ERROR_CHECKING
        checkValid();
        valid = false;
#endif
    }

    MSAHandle(MSA *msa_)
        : msa(msa_), valid(true)
    { }
    inline void checkValid()
    {
#if CMK_ERROR_CHECKING
        if (!valid)
            throw MSA_InvalidHandle();
#endif
    }

public:
    void syncRelease()
    {
        checkInvalidate();
        if (msa->active)
            msa->cache->SyncRelease();
        else
            CmiAbort("sync from an inactive thread!\n");
        msa->active = false;
    }

    void syncDone()
    {
        checkInvalidate();
        msa->sync();
    }

    MSARead<MSA> syncToRead()
    {
        checkInvalidate();
        msa->sync();
        return MSARead<MSA>(msa);
    }

    MSAWrite<MSA> syncToWrite()
    {
        checkInvalidate();
        msa->sync();
        return MSAWrite<MSA>(msa);
    }

    MSAWrite<MSA> syncToReWrite()
    {
        checkInvalidate();
        msa->sync(DEFAULT_SYNC_SINGLE, MSA_CLEAR_ALL);
        return MSAWrite<MSA>(msa);
    }

    MSAAccum<MSA> syncToAccum()
    {
        checkInvalidate();
        msa->sync();
        return MSAAccum<MSA>(msa);
    }

    MSAAccum<MSA> syncToEAccum()
    {
        checkInvalidate();
        msa->sync(DEFAULT_SYNC_SINGLE, MSA_CLEAR_ALL);
        return MSAAccum<MSA>(msa);
    }

    void pup(PUP::er &p)
    {
        p|valid;
        if (valid) {
            if (p.isUnpacking())
                msa = new MSA;
            p|(*msa);
        }
        else if (p.isUnpacking())
            msa = NULL;
    }

    inline int length() { return msa->length(); }

    MSAHandle() : msa(NULL), valid(false) {}
};

template <class MSA>
class MSARead : public MSAHandle<MSA>
{
protected:
    using MSAHandle<MSA>::checkValid;
    using MSAHandle<MSA>::checkInvalidate;
    using MSAHandle<MSA>::msa;

    typedef typename MSA::T ENTRY;

public:
    MSARead(MSA *msa_)
        :  MSAHandle<MSA>(msa_) { }
    MSARead() {}

    // 1D Array access
    inline const ENTRY& get(int x)
    {
        checkValid();
        return msa->get(x);
    }
    inline const ENTRY& operator()(int x) { return get(x); }
    inline const ENTRY& get2(int x)
    {
        checkValid();
        return msa->get2(x);
    }

#ifndef FOR_SUN_CC_ONLY
    // 2D Array access
    inline const ENTRY& get(int x, int y)
    {
        checkValid();
        return msa->get(x, y);
    }
#endif
    inline const ENTRY& operator()(int x, int y) { return get(x, y); }
    inline const ENTRY& get2(int x, int y)
    {
        checkValid();
        return msa->get2(x, y);
    }

#ifndef FOR_SUN_CC_ONLY
    // 3D Array Access
    inline const ENTRY& get(int x, int y, int z)
    {
        checkValid();
        return msa->get(x, y, z);
    }
#endif
    inline const ENTRY& operator()(int x, int y, int z) { return get(x, y, z); }
    inline const ENTRY& get2(int x, int y, int z)
    {
        checkValid();
        return msa->get2(x, y, z);
    }

    // Reads the specified range into the provided buffer in row-major order
    void read(ENTRY *buf, int x1, int y1, int z1, int x2, int y2, int z2)
    {
        checkValid();

        CkAssert(x1 <= x2);
        CkAssert(y1 <= y2);
        CkAssert(z1 <= z2);

        CkAssert(x1 >= msa->xa);
        CkAssert(y1 >= msa->ya);
        CkAssert(z1 >= msa->za);

        CkAssert(x2 <= msa->xb);
        CkAssert(y2 <= msa->yb);
        CkAssert(z2 <= msa->zb);

        int i = 0;

        for (int ix = x1; ix <= x2; ++ix)
            for (int iy = y1; iy <= y2; ++iy)
                for (int iz = z1; iz <= z2; ++iz)
                    buf[i++] = msa->get(ix, iy, iz);
    }
};

template <class MSA>
class MSAWrite : public MSAHandle<MSA>
{
protected:
    using MSAHandle<MSA>::checkValid;
    using MSAHandle<MSA>::checkInvalidate;
    using MSAHandle<MSA>::msa;

    typedef typename MSA::T ENTRY;

public:
    MSAWrite(MSA *msa_)
        : MSAHandle<MSA>(msa_) { }
    MSAWrite() {}

    // 1D Array access
    inline Writable<ENTRY> set(int x)
    {
        checkValid();
        return Writable<ENTRY>(msa->set(x));
    }
    inline Writable<ENTRY> operator()(int x)
    {
        return set(x);
    }

    // 2D Array access
    inline Writable<ENTRY> set(int x, int y)
    {
        checkValid();
        return Writable<ENTRY>(msa->set(x,y));
    }
    inline Writable<ENTRY> operator()(int x, int y)
    {
        return set(x,y);
    }

    // 3D Array access
    inline Writable<ENTRY> set(int x, int y, int z)
    {
        checkValid();
        return Writable<ENTRY>(msa->set(x,y,z));
    }
    inline Writable<ENTRY> operator()(int x, int y, int z)
    {
        return set(x,y,z);
    }

    void write(int x1, int y1, int z1, int x2, int y2, int z2, const ENTRY *buf)
    {
        checkValid();

        CkAssert(x1 <= x2);
        CkAssert(y1 <= y2);
        CkAssert(z1 <= z2);

        CkAssert(x1 >= msa->xa);
        CkAssert(y1 >= msa->ya);
        CkAssert(z1 >= msa->za);

        CkAssert(x2 <= msa->xb);
        CkAssert(y2 <= msa->yb);
        CkAssert(z2 <= msa->zb);

        int i = 0;

        for (int ix = x1; ix <= x2; ++ix)
            for (int iy = y1; iy <= y2; ++iy)
                for (int iz = z1; iz <= z2; ++iz)
                    {
                        msa->set(ix, iy, iz) = buf[i++];
                    }
    }
};

template<class MSA>
class MSAAccum : public MSAHandle<MSA>
{
protected:
    using MSAHandle<MSA>::checkValid;
    using MSAHandle<MSA>::checkInvalidate;
    using MSAHandle<MSA>::msa;

    typedef typename MSA::T ENTRY;

public:
    MSAAccum(MSA *msa_)
        : MSAHandle<MSA>(msa_) { }
    MSAAccum() {}

    // 1D Array Access
    inline Accumulable<MSA> accumulate(int x)
    {
        checkValid();
        return Accumulable<MSA>(msa->accumulate(x));
    }
    inline Accumulable<MSA> operator() (int x)
    { return accumulate(x); }

#ifndef FOR_SUN_CC_ONLY
    // 2D Array Access
    inline Accumulable<MSA> accumulate(int x, int y)
    {
        checkValid();
        return Accumulable<MSA>(msa->accumulate(x,y));
    }
#endif
    inline Accumulable<MSA> operator() (int x, int y)
    { return accumulate(x,y); }

#ifndef FOR_SUN_CC_ONLY
    // 3D Array Access
    inline Accumulable<MSA> accumulate(int x, int y, int z)
    {
        checkValid();
        return Accumulable<MSA>(msa->accumulate(x,y,z));
    }
#endif
    inline Accumulable<MSA> operator() (int x, int y, int z)
    { return accumulate(x,y,z); }

#ifndef FOR_SUN_CC_ONLY
    void accumulate(int x1, int y1, int z1, int x2, int y2, int z2, const ENTRY *buf)
    {
        checkValid();
        CkAssert(x1 <= x2);
        CkAssert(y1 <= y2);
        CkAssert(z1 <= z2);

        CkAssert(x1 >= msa->xa);
        CkAssert(y1 >= msa->ya);
        CkAssert(z1 >= msa->za);

        CkAssert(x2 <= msa->xb);
        CkAssert(y2 <= msa->yb);
        CkAssert(z2 <= msa->zb);

        int i = 0;

        for (int ix = x1; ix <= x2; ++ix)
            for (int iy = y1; iy <= y2; ++iy)
                for (int iz = z1; iz <= z2; ++iz)
                    msa->accumulate(ix, iy, iz, buf[i++]);
    }
#endif
};


/**
   The MSA1D class is a handle to a distributed shared array of items
   of data type ENTRY. There are nEntries total numer of ENTRY's, with
   ENTRIES_PER_PAGE data items per "page".  It is implemented as a
   Chare Array of pages, and a Group representing the local cache.

   The requirements for the templates are:
     ENTRY: User data class stored in the array, with at least:
        - A default constructor and destructor
        - A working assignment operator
        - A working pup routine
     ENTRY_OPS_CLASS: Used to combine values for "accumulate":
        - A method named "getIdentity", taking no arguments and
          returning an ENTRY to use before any accumulation.
        - A method named "accumulate", taking a source/dest ENTRY by reference
          and an ENTRY to add to it by value or const reference.
     ENTRIES_PER_PAGE: Optional integer number of ENTRY objects
        to store and communicate at once.  For good performance,
        make sure this value is a power of two.
 */
template<class ENTRY, class ENTRY_OPS_CLASS, unsigned int ENTRIES_PER_PAGE=MSA_DEFAULT_ENTRIES_PER_PAGE>
class MSA1D
{
public:
    typedef MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CacheGroup_t;
    typedef CProxy_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_CacheGroup_t;
    typedef CProxy_MSA_PageArray<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_PageArray_t;

    typedef ENTRY T;
    typedef ENTRY_OPS_CLASS OPS;
    typedef MSA1D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> thisMSA;
    typedef MSAHandle<thisMSA> Handle;
    typedef MSARead<thisMSA> Read;
    typedef MSAAccum<thisMSA> Accum;
    typedef MSAWrite<thisMSA> Write;
    friend class MSAHandle<thisMSA>;
    friend class MSARead<thisMSA>;
    friend class MSAWrite<thisMSA>;
    friend class MSAAccum<thisMSA>;

protected:
    /// Total number of ENTRY's in the whole array.
    unsigned int nEntries;
    bool initHandleGiven;

    /// Handle to owner of cache.
    CacheGroup_t* cache;
    CProxy_CacheGroup_t cg;

    inline const ENTRY* readablePage(unsigned int page)
    {
        return (const ENTRY*)(cache->readablePage(page));
    }

    // known local page.
    inline const ENTRY* readablePage2(unsigned int page)
    {
        return (const ENTRY*)(cache->readablePage2(page));
    }

    // Returns a pointer to the start of the local copy in the cache of the writeable page.
    // @@ what if begin - end span across two or more pages?
    inline ENTRY* writeablePage(unsigned int page, unsigned int offset)
    {
        return (ENTRY*)(cache->writeablePage(page, offset));
    }

public:
    // @@ Needed for Jade
    inline MSA1D() 
        :initHandleGiven(false) 
    {}

    virtual void pup(PUP::er &p){
        p|nEntries;
        p|cg;
        if (p.isUnpacking()) cache=cg.ckLocalBranch();
    }

    /**
      Create a completely new MSA array.  This call creates the
      corresponding groups, so only call it once per array.
    */
    inline MSA1D(unsigned int nEntries_, unsigned int num_wrkrs, 
                 unsigned int maxBytes=MSA_DEFAULT_MAX_BYTES) 
        : nEntries(nEntries_), initHandleGiven(false)
    {
        // first create the Page Array and the Page Group
        unsigned int nPages = (nEntries + ENTRIES_PER_PAGE - 1)/ENTRIES_PER_PAGE;
        CProxy_PageArray_t pageArray = CProxy_PageArray_t::ckNew(nPages);
        cg = CProxy_CacheGroup_t::ckNew(nPages, pageArray, maxBytes, nEntries, num_wrkrs);
        pageArray.setCacheProxy(cg);
        pageArray.ckSetReductionClient(new CkCallback(CkIndex_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>::SyncDone(NULL), cg));
        cache = cg.ckLocalBranch();
    }

    // Deprecated API for accessing CacheGroup directly.
    inline MSA1D(CProxy_CacheGroup_t cg_) : cg(cg_), initHandleGiven(false)
    {
        cache = cg.ckLocalBranch();
        nEntries = cache->getNumEntries();
    }

    inline ~MSA1D()
    {
        // TODO: how to get rid of the cache group and the page array
        //(cache->getArray()).destroy();
        //cg.destroy();
        // TODO: calling FreeMem does not seem to work. Need to debug it.
				cache->unroll();
 //       cache->FreeMem();
    }

    /**
     * this function is supposed to be called when the thread/object using this array
     * migrates to another PE.
     */
    inline void changePE()
    {
        cache = cg.ckLocalBranch();

        /* don't need to update the number of entries, as that does not change */
    }

    // ================ Accessor/Utility functions ================
    /// Get the total length of the array, across all processors.
    inline unsigned int length() const { return nEntries; }

    inline const CProxy_CacheGroup_t &getCacheGroup() const { return cg; }

    // Avoid using the term "page size" because it is confusing: does
    // it mean in bytes or number of entries?
    inline unsigned int getNumEntriesPerPage() const { return ENTRIES_PER_PAGE; }

    /// Return the page this entry is stored at.
    inline unsigned int getPageIndex(unsigned int idx)
    {
        return idx / ENTRIES_PER_PAGE;
    }

    /// Return the offset, in entries, that this entry is stored at within a page.
    inline unsigned int getOffsetWithinPage(unsigned int idx)
    {
        return idx % ENTRIES_PER_PAGE;
    }

    // ================ MSA API ================

    // We need to know the total number of workers across all
    // processors, and we also calculate the number of worker threads
    // running on this processor.
    //
    // Blocking method, basically does a barrier until all workers
    // enroll.
    inline void enroll(int num_workers)
    {
        // @@ This is a hack to identify the number of MSA1D
        // threads on this processor.  This number is needed for sync.
        //
        // @@ What if a MSA1D thread migrates?
        cache->enroll(num_workers);
    }

    // idx is the element to be read/written
    //
    // This function returns a reference to the first element on the
    // page that contains idx.
    inline ENTRY& getPageBottom(unsigned int idx, MSA_Page_Fault_t accessMode)
    {
        if (accessMode==Read_Fault) {
            unsigned int page = idx / ENTRIES_PER_PAGE;
            return const_cast<ENTRY&>(readablePage(page)[0]);
        } else {
            CkAssert(accessMode==Write_Fault || accessMode==Accumulate_Fault);
            unsigned int page = idx / ENTRIES_PER_PAGE;
            unsigned int offset = idx % ENTRIES_PER_PAGE;
            ENTRY* e=writeablePage(page, offset);
            return e[0];
        }
    }

    inline void FreeMem()
    {
        cache->FreeMem();
    }

    /// Non-blocking prefetch of entries from start to end, inclusive.
    /// Prefetch'd pages are locked into the cache, so you must call
    ///   unlock afterwards.
    inline void Prefetch(unsigned int start, unsigned int end)
    {
        unsigned int page1 = start / ENTRIES_PER_PAGE;
        unsigned int page2 = end / ENTRIES_PER_PAGE;
        cache->Prefetch(page1, page2);
    }

    /// Block until all prefetched pages arrive.
    inline int WaitAll()    { return cache->WaitAll(); }

    /// Unlock all locked pages
    inline void Unlock()    { return cache->UnlockPages(); }

    /// start and end are element indexes.
    /// Unlocks completely spanned pages given a range of elements
    /// index'd from "start" to "end", inclusive.  If start/end does not span a
    /// page completely, i.e. start/end is in the middle of a page,
    /// the entire page is still unlocked--in particular, this means
    /// you should not have several adjacent ranges locked.
    inline void Unlock(unsigned int start, unsigned int end)
    {
        unsigned int page1 = start / ENTRIES_PER_PAGE;
        unsigned int page2 = end / ENTRIES_PER_PAGE;
        cache->UnlockPages(page1, page2);
    }

    inline Write getInitialWrite()
    {
        if (initHandleGiven)
            throw MSA_InvalidHandle();

        initHandleGiven = true;
        return Write(this);
    }

    inline Accum getInitialAccum()
    {
        if (initHandleGiven)
            throw MSA_InvalidHandle();

        initHandleGiven = true;
        return Accum(this);
    }

  // These are the meat of the MSA API, but they are only accessible
  // through appropriate handles (defined in the public section above).
protected:
    /// Return a read-only copy of the element at idx.
    ///   May block if the element is not already in the cache.
    inline const ENTRY& get(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return readablePage(page)[offset];
    }

    inline const ENTRY& operator[](unsigned int idx)
    {
        return get(idx);
    }

    /// Return a read-only copy of the element at idx;
    ///   ONLY WORKS WHEN ELEMENT IS ALREADY IN THE CACHE--
    ///   WILL SEGFAULT IF ELEMENT NOT ALREADY PRESENT.
    ///    Never blocks; may crash if element not already present.
    inline const ENTRY& get2(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return readablePage2(page)[offset];
    }

    /// Return a writeable copy of the element at idx.
    ///    Never blocks; will create a new blank element if none exists locally.
    ///    UNDEFINED if two threads set the same element.
    inline ENTRY& set(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        ENTRY* e=writeablePage(page, offset);
        return e[offset];
    }

    /// Fetch the ENTRY at idx to be accumulated.
    ///   You must perform the accumulation on 
    ///     the return value before calling "sync".
    ///   Never blocks.
    inline ENTRY& accumulate(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return cache->accumulate(page, offset);
    }
    
    /// Add ent to the element at idx.
    ///   Never blocks.
    ///   Merges together accumulates from different threads.
    inline void accumulate(unsigned int idx, const ENTRY& ent)
    {
        ENTRY_OPS_CLASS::accumulate(accumulate(idx),ent);
    }

    /// Synchronize reads and writes across the entire array.
    inline void sync(int single=0, bool clear = false)
    {
        cache->SyncReq(single, clear);
    }
};


// define a 2d distributed array based on the 1D array, support row major and column
// major arrangement of data
template<class ENTRY, class ENTRY_OPS_CLASS, unsigned int ENTRIES_PER_PAGE=MSA_DEFAULT_ENTRIES_PER_PAGE, MSA_Array_Layout_t ARRAY_LAYOUT=MSA_ROW_MAJOR>
class MSA2D : public MSA1D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>
{
public:
    typedef CProxy_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_CacheGroup_t;
    typedef MSA1D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> super;

    typedef ENTRY T;
    typedef ENTRY_OPS_CLASS OPS;
    typedef MSA2D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> thisMSA;
    typedef MSAHandle<thisMSA> Handle;
    typedef MSARead<thisMSA> Read;
    typedef MSAAccum<thisMSA> Accum;
    typedef MSAWrite<thisMSA> Write;
    friend class MSAHandle<thisMSA>;
    friend class MSARead<thisMSA>;
    friend class MSAWrite<thisMSA>;
    friend class MSAAccum<thisMSA>;

protected:
    unsigned int rows, cols;

public:
    // @@ Needed for Jade
    inline MSA2D() : super() {}
    virtual void pup(PUP::er &p) {
       super::pup(p);
       p|rows; p|cols;
    };

    inline MSA2D(unsigned int rows_, unsigned int cols_, unsigned int numwrkrs,
                 unsigned int maxBytes=MSA_DEFAULT_MAX_BYTES)
        :super(rows_*cols_, numwrkrs, maxBytes)
    {
        rows = rows_; cols = cols_;
    }

    inline MSA2D(unsigned int rows_, unsigned int cols_, CProxy_CacheGroup_t cg_)
        : rows(rows_), cols(cols_), super(cg_)
    {}

    // get the 1D index of the given entry as per the row major/column major format
    inline unsigned int getIndex(unsigned int row, unsigned int col)
    {
        unsigned int index;

        if(ARRAY_LAYOUT==MSA_ROW_MAJOR)
            index = row*cols + col;
        else
            index = col*rows + row;

        return index;
    }

    // Which page is (row, col) on?
    inline unsigned int getPageIndex(unsigned int row, unsigned int col)
    {
        return getIndex(row, col)/ENTRIES_PER_PAGE;
    }

    inline unsigned int getOffsetWithinPage(unsigned int row, unsigned int col)
    {
        return getIndex(row, col)%ENTRIES_PER_PAGE;
    }

    inline unsigned int getRows(void) const {return rows;}
    inline unsigned int getCols(void) const {return cols;}
    inline unsigned int getColumns(void) const {return cols;}
    inline MSA_Array_Layout_t getArrayLayout() const {return ARRAY_LAYOUT;}

    inline void Prefetch(unsigned int start, unsigned int end)
    {
        // prefetch the start ... end rows/columns into the cache
        if(start > end)
        {
            unsigned int temp = start;
            start = end;
            end = temp;
        }

        unsigned int index1 = (ARRAY_LAYOUT==MSA_ROW_MAJOR) ? getIndex(start, 0) : getIndex(0, start);
        unsigned int index2 = (ARRAY_LAYOUT==MSA_ROW_MAJOR) ? getIndex(end, cols-1) : getIndex(rows-1, end);

        MSA1D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>::Prefetch(index1, index2);
    }

    // Unlocks pages starting from row "start" through row "end", inclusive
    inline void UnlockPages(unsigned int start, unsigned int end)
    {
        if(start > end)
        {
            unsigned int temp = start;
            start = end;
            end = temp;
        }

        unsigned int index1 = (ARRAY_LAYOUT==MSA_ROW_MAJOR) ? getIndex(start, 0) : getIndex(0, start);
        unsigned int index2 = (ARRAY_LAYOUT==MSA_ROW_MAJOR) ? getIndex(end, cols-1) : getIndex(rows-1, end);

        MSA1D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>::Unlock(index1, index2);
    }

    inline Write getInitialWrite()
    {
        if (super::initHandleGiven)
            throw MSA_InvalidHandle();

        super::initHandleGiven = true;
        return Write(this);
    }

    inline Accum getInitialAccum()
    {
        if (super::initHandleGiven)
            throw MSA_InvalidHandle();

        super::initHandleGiven = true;
        return Accum(this);
    }

protected:
    inline const ENTRY& get(unsigned int row, unsigned int col)
    {
        return super::get(getIndex(row, col));
    }

    // known local
    inline const ENTRY& get2(unsigned int row, unsigned int col)
    {
        return super::get2(getIndex(row, col));
    }

    // MSA2D::
    inline ENTRY& set(unsigned int row, unsigned int col)
    {
        return super::set(getIndex(row, col));
    }
};

/**
   The MSA3D class is a handle to a distributed shared array of items
   of data type ENTRY. There are nEntries total numer of ENTRY's, with
   ENTRIES_PER_PAGE data items per "page".  It is implemented as a
   Chare Array of pages, and a Group representing the local cache.

   The requirements for the templates are:
     ENTRY: User data class stored in the array, with at least:
        - A default constructor and destructor
        - A working assignment operator
        - A working pup routine
     ENTRY_OPS_CLASS: Used to combine values for "accumulate":
        - A method named "getIdentity", taking no arguments and
          returning an ENTRY to use before any accumulation.
        - A method named "accumulate", taking a source/dest ENTRY by reference
          and an ENTRY to add to it by value or const reference.
     ENTRIES_PER_PAGE: Optional integer number of ENTRY objects
        to store and communicate at once.  For good performance,
        make sure this value is a power of two.
 */
template<class ENTRY, class ENTRY_OPS_CLASS, unsigned int ENTRIES_PER_PAGE>
class MSA3D
{
    /// Inclusive lower and upper bounds on entry indices
    int xa, xb, ya, yb, za, zb;
    /// Size of the array in each dimension
    unsigned dim_x, dim_y, dim_z;

public:
    typedef MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CacheGroup_t;
    typedef CProxy_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_CacheGroup_t;
    typedef CProxy_MSA_PageArray<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_PageArray_t;

    typedef ENTRY T;
    typedef ENTRY_OPS_CLASS OPS;
    typedef MSA3D<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> thisMSA;
    typedef MSAHandle<thisMSA> Handle;
    typedef MSARead<thisMSA> Read;
    typedef MSAAccum<thisMSA> Accum;
    typedef MSAWrite<thisMSA> Write;
    friend class MSAHandle<thisMSA>;
    friend class MSARead<thisMSA>;
    friend class MSAWrite<thisMSA>;
    friend class MSAAccum<thisMSA>;

protected:
    /// Total number of ENTRY's in the whole array.
    unsigned int nEntries;
    bool initHandleGiven;

    /// Handle to owner of cache.
    CacheGroup_t* cache;
    CProxy_CacheGroup_t cg;

    inline const ENTRY* readablePage(unsigned int page)
    {
        return (const ENTRY*)(cache->readablePage(page));
    }

    // known local page.
    inline const ENTRY* readablePage2(unsigned int page)
    {
        return (const ENTRY*)(cache->readablePage2(page));
    }

    // Returns a pointer to the start of the local copy in the cache of the writeable page.
    // @@ what if begin - end span across two or more pages?
    inline ENTRY* writeablePage(unsigned int page, unsigned int offset)
    {
        return (ENTRY*)(cache->writeablePage(page, offset));
    }

public:
    // @@ Needed for Jade
    inline MSA3D() 
        :initHandleGiven(false) 
    {}

    virtual void pup(PUP::er &p){
        p|xa; p|xb;
        p|ya; p|yb;
        p|za; p|zb;
        p|dim_x;
        p|dim_y;
        p|dim_z;
        p|nEntries;
        p|cg;
        if (p.isUnpacking()) cache=cg.ckLocalBranch();
    }

    /**
      Create a completely new MSA array.  This call creates the
      corresponding groups, so only call it once per array.

      Valid indices lie in [0,x-1]*[0,y-1]*[0,z-1]
    */
    inline MSA3D(unsigned x, unsigned y, unsigned z, unsigned int num_wrkrs, 
                 unsigned int maxBytes=MSA_DEFAULT_MAX_BYTES)
        : xa(0), ya(0), za(0), xb(x-1), yb(y-1), zb(z-1), dim_x(x), dim_y(y), dim_z(z),
          initHandleGiven(false)
    {
        unsigned nEntries = x*y*z;
        unsigned int nPages = (nEntries + ENTRIES_PER_PAGE - 1)/ENTRIES_PER_PAGE;
        CProxy_PageArray_t pageArray = CProxy_PageArray_t::ckNew(nPages);
        cg = CProxy_CacheGroup_t::ckNew(nPages, pageArray, maxBytes, nEntries, num_wrkrs);
        pageArray.setCacheProxy(cg);
        //pageArray.ckSetReductionClient(new CkCallback(CkIndex_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>::SyncDone(NULL), cg));
        cache = cg.ckLocalBranch();
    }

    /**
      Create a completely new MSA array.  This call creates the
      corresponding groups, so only call it once per array.

      Valid indices lie in [xa,xb]*[ya,yb]*[za,zb]
    */
    inline MSA3D(int xa_, int xb_, int ya_, int yb_, int za_, int zb_,
                 unsigned int num_wrkrs, unsigned int maxBytes=MSA_DEFAULT_MAX_BYTES)
        : xa(xa_), xb(xb_), ya(ya_), yb(yb_), za(za_), zb(zb_),
          dim_x(xb-xa+1), dim_y(yb-ya+1), dim_z(zb-za+1),
          initHandleGiven(false)
    {
        unsigned nEntries = dim_x*dim_y*dim_z;
        unsigned int nPages = (nEntries + ENTRIES_PER_PAGE - 1)/ENTRIES_PER_PAGE;
        CProxy_PageArray_t pageArray = CProxy_PageArray_t::ckNew(nPages);
        cg = CProxy_CacheGroup_t::ckNew(nPages, pageArray, maxBytes, nEntries, num_wrkrs);
        pageArray.setCacheProxy(cg);
        //pageArray.ckSetReductionClient(new CkCallback(CkIndex_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>::SyncDone(NULL), cg));
        cache = cg.ckLocalBranch();
    }

    inline ~MSA3D()
    {
        // TODO: how to get rid of the cache group and the page array
        //(cache->getArray()).destroy();
        //cg.destroy();
        // TODO: calling FreeMem does not seem to work. Need to debug it.
        //cache->unroll();
        //cache->FreeMem();
    }

    /**
     * this function is supposed to be called when the thread/object using this array
     * migrates to another PE.
     */
    inline void changePE()
    {
        cache = cg.ckLocalBranch();

        /* don't need to update the number of entries, as that does not change */
    }

    // ================ Accessor/Utility functions ================

    inline const CProxy_CacheGroup_t &getCacheGroup() const { return cg; }

    // Avoid using the term "page size" because it is confusing: does
    // it mean in bytes or number of entries?
    inline unsigned int getNumEntriesPerPage() const { return ENTRIES_PER_PAGE; }

    inline unsigned int index(unsigned x, unsigned y, unsigned z)
    {
        x -= xa;
        y -= ya;
        z -= za;
        CkAssert(x < dim_x);
        CkAssert(y < dim_y);
        CkAssert(z < dim_z);
        return ((x*dim_y) + y) * dim_z + z;
    }
    
    /// Return the page this entry is stored at.
    inline unsigned int getPageIndex(unsigned int idx)
    {
        return idx / ENTRIES_PER_PAGE;
    }

    /// Return the offset, in entries, that this entry is stored at within a page.
    inline unsigned int getOffsetWithinPage(unsigned int idx)
    {
        return idx % ENTRIES_PER_PAGE;
    }

    // ================ MSA API ================

    // We need to know the total number of workers across all
    // processors, and we also calculate the number of worker threads
    // running on this processor.
    //
    // Blocking method, basically does a barrier until all workers
    // enroll.
    inline void enroll(int num_workers)
    {
        // @@ This is a hack to identify the number of MSA3D
        // threads on this processor.  This number is needed for sync.
        //
        // @@ What if a MSA3D thread migrates?
        cache->enroll(num_workers);
    }

    void enroll()
    {
        cache->enroll();
    }

    // idx is the element to be read/written
    //
    // This function returns a reference to the first element on the
    // page that contains idx.
    inline ENTRY& getPageBottom(unsigned int idx, MSA_Page_Fault_t accessMode)
    {
        if (accessMode==Read_Fault) {
            unsigned int page = idx / ENTRIES_PER_PAGE;
            return const_cast<ENTRY&>(readablePage(page)[0]);
        } else {
            CkAssert(accessMode==Write_Fault || accessMode==Accumulate_Fault);
            unsigned int page = idx / ENTRIES_PER_PAGE;
            unsigned int offset = idx % ENTRIES_PER_PAGE;
            ENTRY* e=writeablePage(page, offset);
            return e[0];
        }
    }

    inline void FreeMem()
    {
        cache->FreeMem();
    }

    /// Non-blocking prefetch of entries from start to end, inclusive.
    /// Prefetch'd pages are locked into the cache, so you must call
    ///   unlock afterwards.
    inline void Prefetch(unsigned int start, unsigned int end)
    {
        unsigned int page1 = start / ENTRIES_PER_PAGE;
        unsigned int page2 = end / ENTRIES_PER_PAGE;
        cache->Prefetch(page1, page2);
    }

    /// Block until all prefetched pages arrive.
    inline int WaitAll()    { return cache->WaitAll(); }

    /// Unlock all locked pages
    inline void Unlock()    { return cache->UnlockPages(); }

    /// start and end are element indexes.
    /// Unlocks completely spanned pages given a range of elements
    /// index'd from "start" to "end", inclusive.  If start/end does not span a
    /// page completely, i.e. start/end is in the middle of a page,
    /// the entire page is still unlocked--in particular, this means
    /// you should not have several adjacent ranges locked.
    inline void Unlock(unsigned int start, unsigned int end)
    {
        unsigned int page1 = start / ENTRIES_PER_PAGE;
        unsigned int page2 = end / ENTRIES_PER_PAGE;
        cache->UnlockPages(page1, page2);
    }

    inline Write getInitialWrite()
    {
        if (initHandleGiven)
            CmiAbort("Trying to get an MSA's initial handle a second time");

        initHandleGiven = true;
        return Write(this);
    }

    inline Accum getInitialAccum()
    {
        if (initHandleGiven)
            CmiAbort("Trying to get an MSA's initial handle a second time");

        initHandleGiven = true;
        return Accum(this);
    }

  // These are the meat of the MSA API, but they are only accessible
  // through appropriate handles (defined in the public section above).
protected:
    /// Return a read-only copy of the element at idx.
    ///   May block if the element is not already in the cache.
    inline const ENTRY& get(unsigned x, unsigned y, unsigned z)
    {
        unsigned int idx = index(x,y,z);
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return readablePage(page)[offset];
    }

    /// Return a read-only copy of the element at idx;
    ///   ONLY WORKS WHEN ELEMENT IS ALREADY IN THE CACHE--
    ///   WILL SEGFAULT IF ELEMENT NOT ALREADY PRESENT.
    ///    Never blocks; may crash if element not already present.
    inline const ENTRY& get2(unsigned x, unsigned y, unsigned z)
    {
        unsigned int idx = index(x,y,z);
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return readablePage2(page)[offset];
    }

    /// Return a writeable copy of the element at idx.
    ///    Never blocks; will create a new blank element if none exists locally.
    ///    UNDEFINED if two threads set the same element.
    inline ENTRY& set(unsigned x, unsigned y, unsigned z)
    {
        unsigned int idx = index(x,y,z);
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        ENTRY* e=writeablePage(page, offset);
        return e[offset];
    }

    /// Fetch the ENTRY at idx to be accumulated.
    ///   You must perform the accumulation on 
    ///     the return value before calling "sync".
    ///   Never blocks.
    inline ENTRY& accumulate(unsigned x, unsigned y, unsigned z)
    {
        unsigned int idx = index(x,y,z);
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return cache->accumulate(page, offset);
    }
    
    /// Add ent to the element at idx.
    ///   Never blocks.
    ///   Merges together accumulates from different threads.
    inline void accumulate(unsigned x, unsigned y, unsigned z, const ENTRY& ent)
    {
        ENTRY_OPS_CLASS::accumulate(accumulate(x,y,z),ent);
    }

    /// Synchronize reads and writes across the entire array.
    inline void sync(int single=0, bool clear = false)
    {
        cache->SyncReq(single, clear);
    }
};

}
#endif
