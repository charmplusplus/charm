// emacs mode line -*- mode: c++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MSA_DISTARRAY_H
#define MSA_DISTARRAY_H

#include <utility>
#include <algorithm>
#include "msa-DistPageMgr.h"


struct MSA_InvalidHandle { };

template <typename ENTRY>
class Writable
{
    ENTRY &e;
    
public:
    Writable(ENTRY &e_) : e(e_) {}
    inline const ENTRY& operator= (const ENTRY& rhs) { e = rhs; return rhs; }
};

template <typename ENTRY, class ENTRY_OPS_CLASS>
class Accumulable
{
    ENTRY &e;
    
public:
    Accumulable(ENTRY &e_) : e(e_) {}
    template<typename T>
    void operator+=(const T &rhs_)
        { ENTRY_OPS_CLASS::accumulate(e, rhs_); }
    template<typename T>
    void accumulate(const T& rhs)
        {
            ENTRY_OPS_CLASS::accumulate(e, rhs);
        }
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

    // Sun's C++ compiler doesn't understand that nested classes are
    // members for the sake of access to private data. (2008-10-23)
    class Read; class Write; class Accum;
    friend class Read; friend class Write; friend class Accum;

	class Handle
	{
    public:
        inline unsigned int length() const { return msa->length(); }

	protected:
        MSA1D *msa;
        bool valid;

        friend class MSA1D;

        void inline checkInvalidate(MSA1D *m) 
        {
            if (m != msa || !valid)
                throw MSA_InvalidHandle();
            valid = false;
        }

        Handle(MSA1D *msa_) 
            : msa(msa_), valid(true) 
        { }
        Handle() : msa(NULL), valid(false) {}
        void checkValid()
        {
            if (!valid)
                throw MSA_InvalidHandle();
        }

    private:
        // Disallow copy construction
        Handle(Handle &);
    };

    class Read : public Handle
    {
    protected:
        friend class MSA1D;
        Read(MSA1D *msa_)
            :  Handle(msa_) { }
        using Handle::checkValid;
        using Handle::checkInvalidate;

    public:
        inline const ENTRY& get(unsigned int idx)
        {
            checkValid();
            return Handle::msa->get(idx); 
        }
        inline const ENTRY& operator[](unsigned int idx) { return get(idx); }
        inline const ENTRY& operator()(unsigned int idx) { return get(idx); }
        inline const ENTRY& get2(unsigned int idx)
        {
            checkValid();
            return Handle::msa->get2(idx);
        }
        Read() {}
    };

    class Write : public Handle
    {
    protected:
        friend class MSA1D;
        Write(MSA1D *msa_)
            : Handle(msa_) { }

    public:
        inline Writable<ENTRY> set(unsigned int idx)
        {
            Handle::checkValid();
            return Writable<ENTRY>(Handle::msa->set(idx));
        }
        inline Writable<ENTRY> operator()(unsigned int idx)
            { return set(idx); }
    };

    class Accum : public Handle
    {
    protected:
        friend class MSA1D;
        Accum(MSA1D *msa_)
            : Handle(msa_) { }
        using Handle::checkInvalidate;
    public:
        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> accumulate(unsigned int idx)
        { 
            Handle::checkValid();
            return Accumulable<ENTRY, ENTRY_OPS_CLASS>(Handle::msa->accumulate(idx));
        }
        inline void accumulate(unsigned int idx, const ENTRY& ent)
        {
            Handle::checkValid();
            Handle::msa->accumulate(idx, ent);
        }

        void contribute(unsigned int idx, const ENTRY *begin, const ENTRY *end)
        {
            Handle::checkValid();
            for (const ENTRY *e = begin; e != end; ++e, ++idx)
                {
                    Handle::msa->accumulate(idx, *e);
                }
        }

        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> operator() (unsigned int idx)
            { return accumulate(idx); }
    };

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

    static const int DEFAULT_SYNC_SINGLE = 0;

    inline Read &syncToRead(Handle &m, int single = DEFAULT_SYNC_SINGLE)
    {
        m.checkInvalidate(this);
        delete &m;
        sync(single);
        return *(new Read(*this));
    }

    inline Write &syncToWrite(Handle &m, int single = DEFAULT_SYNC_SINGLE)
    {
        m.checkInvalidate(this);
        delete &m;
        sync(single);
        return *(new Write(*this));
    }

    inline Accum &syncToAccum(Handle &m, int single = DEFAULT_SYNC_SINGLE)
    {
        m.checkInvalidate(this);
        delete &m;
        sync(single);
        return *(new Accum(*this));
    }

    inline Write &getInitialWrite()
    {
        if (initHandleGiven)
            throw MSA_InvalidHandle();

        Write *w = new Write(*this);
        sync();
        initHandleGiven = true;
        return *w;
    }

    inline Accum &getInitialAccum()
    {
        if (initHandleGiven)
            throw MSA_InvalidHandle();

        Accum *a = new Accum(*this);
        sync();
        initHandleGiven = true;
        return *a;
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

protected:
    unsigned int rows, cols;

public:
    // @@ Needed for Jade
    inline MSA2D() : super() {}
    virtual void pup(PUP::er &p) {
       super::pup(p);
       p|rows; p|cols;
    };

	class Handle
	{
	protected:
        MSA2D *msa;
        bool valid;

        friend class MSA2D;

        inline void checkInvalidate(MSA2D *m)
        {
            if (*msa != m || !valid)
                throw MSA_InvalidHandle();
            valid = false;
        }

        Handle(MSA2D *msa_) 
            : msa(msa_), valid(true) 
        { }
        Handle() : msa(NULL), valid(false) {}

        inline void checkValid()
        {
            if (!valid)
                throw MSA_InvalidHandle();
        }
    private:
        // Disallow copy construction
        Handle(Handle &);
    };

    class Read : public Handle
    {
    private:
        friend class MSA2D;
        Read(MSA2D *msa_)
            :  Handle(msa_) { }

    public: 
        inline const ENTRY& get(unsigned int row, unsigned int col)
        {
            Handle::checkValid();
            return Handle::msa->get(row, col);
        }
        inline const ENTRY& get2(unsigned int row, unsigned int col)
        {
            Handle::checkValid();
            return Handle::msa->get2(row, col);
        }

        inline const ENTRY& operator() (unsigned int row, unsigned int col)
            {
                return get(row,col);
            }

        Read() { }
    };

    class Write : public Handle
    {
    private:
        friend class MSA2D;
        Write(MSA2D *msa_)
            :  Handle(msa_) { }

    public: 
        inline Writable<ENTRY> set(unsigned int row, unsigned int col)
        {
            Handle::checkValid();
            return Writable<ENTRY>(Handle::msa->set(row, col));
        }

        inline Writable<ENTRY> operator()(unsigned int row, unsigned int col)
            { return set(row, col); }
    };

    class Accum : public Handle
    {
    protected:
        friend class MSA2D;
        Accum(MSA2D *msa_)
            : Handle(msa_) { }
        using Handle::checkInvalidate;
    public:
        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> accumulate(unsigned int idx)
        {
            Handle::checkValid();
            return Accumulable<ENTRY, ENTRY_OPS_CLASS>(Handle::msa->accumulate(idx));
        }
        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> accumulate(unsigned int x, unsigned int y)
        {
            Handle::checkValid();
            return Accumulable<ENTRY, ENTRY_OPS_CLASS>(Handle::msa->accumulate(Handle::msa->getIndex(x, y)));
        }
        inline void accumulate(unsigned int idx, const ENTRY& ent)
        {
            Handle::checkValid();
            Handle::msa->accumulate(idx, ent);
        }

        void contribute(unsigned int idx, const ENTRY *begin, const ENTRY *end)
        {
            Handle::checkValid();
            for (const ENTRY *e = begin; e != end; ++e, ++idx)
                {
                    Handle::msa->accumulate(idx, *e);
                }
        }

        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> operator() (unsigned int idx)
            { return accumulate(idx); }
        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> operator() (unsigned int x, unsigned int y)
            { return accumulate(x, y); }
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

    inline Read& syncToRead(Handle &m, int single = super::DEFAULT_SYNC_SINGLE)
    {
        m.checkInvalidate(this);
        delete &m;
        super::sync(single);
        return *(new Read(*this));
    }

    inline Write& syncToWrite(Handle &m, int single = super::DEFAULT_SYNC_SINGLE)
    {
        m.checkInvalidate(this);
        delete &m;
        super::sync(single);
        return *(new Write(*this));
    }

    inline Accum& syncToAccum(Handle &m, int single = super::DEFAULT_SYNC_SINGLE)
    {
        m.checkInvalidate(this);
        delete &m;
        super::sync(single);
        return *(new Accum(*this));
    }

    inline Write& getInitialWrite()
    {
        if (super::initHandleGiven)
            throw MSA_InvalidHandle();

        Write *w = new Write(*this);
        super::initHandleGiven = true;
        return *w;
    }

    inline Accum &getInitialAccum()
    {
        if (super::initHandleGiven)
            throw MSA_InvalidHandle();

        Accum *a = new Accum(*this);
        sync();
        super::initHandleGiven = true;
        return *a;
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

namespace MSA
{
    using std::min;
    using std::max;


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
    unsigned dim_x, dim_y, dim_z;


public:
    typedef MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CacheGroup_t;
    typedef CProxy_MSA_CacheGroup<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_CacheGroup_t;
    typedef CProxy_MSA_PageArray<ENTRY, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_PageArray_t;

    // Sun's C++ compiler doesn't understand that nested classes are
    // members for the sake of access to private data. (2008-10-23)
    class Read; class Write; class Accum;
    friend class Read; friend class Write; friend class Accum;

	class Handle
	{
	protected:
        MSA3D *msa;
        bool valid;

        friend class MSA3D;

        void inline checkInvalidate() 
        {
            if (!valid)
                throw MSA_InvalidHandle();
            valid = false;
        }

        Handle(MSA3D *msa_) 
            : msa(msa_), valid(true) 
        { }
        void checkValid()
        {
            if (!valid)
                throw MSA_InvalidHandle();
        }
        
    public:
        inline void syncRelease()
            {
                checkInvalidate();
                if (msa->active)
                    msa->cache->SyncRelease();
                else
                    CmiAbort("sync from an inactive thread!\n");
                msa->active = false;
            }

        inline void syncDone()
            {
                checkInvalidate();
                msa->sync(DEFAULT_SYNC_SINGLE);
            }

        inline Read syncToRead()
            {
                checkInvalidate();
                msa->sync(DEFAULT_SYNC_SINGLE);
                return Read(msa);
            }

        inline Write syncToWrite()
            {
                checkInvalidate();
                msa->sync(DEFAULT_SYNC_SINGLE);
                return Write(msa);
            }

        inline Write syncToReWrite()
            {
                checkInvalidate();
                msa->sync(DEFAULT_SYNC_SINGLE, MSA_CLEAR_ALL);
                return Write(msa);
            }

        inline Accum syncToAccum()
            {
                checkInvalidate();
                msa->sync(DEFAULT_SYNC_SINGLE);
                return Accum(msa);
            }

        inline Accum syncToEAccum()
        {
            checkInvalidate();
            msa->sync(DEFAULT_SYNC_SINGLE, MSA_CLEAR_ALL);
            return Accum(msa);
        }

        void pup(PUP::er &p)
            {
                p|valid;
                if (valid)
                {
                    if (p.isUnpacking())
                        msa = new MSA3D;
                    p|(*msa);
                }
                else if (p.isUnpacking())
                    msa = NULL;
            }

        Handle() : msa(NULL), valid(false) {}
    };

    class Read : public Handle
    {
    protected:
        friend class MSA3D;
        Read(MSA3D *msa_)
            :  Handle(msa_) { }
        using Handle::checkValid;
        using Handle::checkInvalidate;

    public:
        Read() {}

        inline const ENTRY& get(unsigned x, unsigned y, unsigned z)
        {
            checkValid();
            return Handle::msa->get(x, y, z); 
        }
        inline const ENTRY& operator()(unsigned x, unsigned y, unsigned z) { return get(x, y, z); }
        inline const ENTRY& get2(unsigned x, unsigned y, unsigned z)
        {
            checkValid();
            return Handle::msa->get2(x, y, z);
        }

        // Reads the specified range into the provided buffer in row-major order
        void read(ENTRY *buf, unsigned x1, unsigned y1, unsigned z1, unsigned x2, unsigned y2, unsigned z2)
        {
            checkValid();

            CkAssert(x1 <= x2);
            CkAssert(y1 <= y2);
            CkAssert(z1 <= z2);

            CkAssert(x1 >= 0);
            CkAssert(y1 >= 0);
            CkAssert(z1 >= 0);

            CkAssert(x2 < Handle::msa->dim_x);
            CkAssert(y2 < Handle::msa->dim_y);
            CkAssert(z2 < Handle::msa->dim_z);

            unsigned i = 0;

            for (unsigned ix = x1; ix <= x2; ++ix)
                for (unsigned iy = y1; iy <= y2; ++iy)
                    for (unsigned iz = z1; iz <= z2; ++iz)
                        buf[i++] = Handle::msa->get(ix, iy, iz);
        }
    };

    class Write : public Handle
    {
    protected:
        friend class MSA3D;
        Write(MSA3D *msa_)
            : Handle(msa_) { }

    public:
        Write() {}

        inline Writable<ENTRY> set(unsigned x, unsigned y, unsigned z)
        {
            Handle::checkValid();
            return Writable<ENTRY>(Handle::msa->set(x,y,z));
        }
        inline Writable<ENTRY> operator()(unsigned x, unsigned y, unsigned z)
        {
            return set(x,y,z);
        }

        void write(unsigned x1, unsigned y1, unsigned z1, unsigned x2, unsigned y2, unsigned z2, const ENTRY *buf)
        {
            Handle::checkValid();

            CkAssert(x1 <= x2);
            CkAssert(y1 <= y2);
            CkAssert(z1 <= z2);

            CkAssert(x1 >= 0);
            CkAssert(y1 >= 0);
            CkAssert(z1 >= 0);

            CkAssert(x2 < Handle::msa->dim_x);
            CkAssert(y2 < Handle::msa->dim_y);
            CkAssert(z2 < Handle::msa->dim_z);

            unsigned i = 0;

            for (unsigned ix = x1; ix <= x2; ++ix)
                for (unsigned iy = y1; iy <= y2; ++iy)
                    for (unsigned iz = z1; iz <= z2; ++iz)
                    {
                        if (isnan(buf[i]))
                            CmiAbort("Tried to write a NaN!");
                        Handle::msa->set(ix, iy, iz) = buf[i++];
                    }
        }
#if 0
    private:
        Write(Write &);
#endif
    };

    class Accum : public Handle
    {
    protected:
        friend class MSA3D;
        Accum(MSA3D *msa_)
            : Handle(msa_) { }
        using Handle::checkInvalidate;
    public:
        Accum() {}

        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> accumulate(unsigned int x, unsigned int y, unsigned int z)
        {
            Handle::checkValid();
            return Accumulable<ENTRY, ENTRY_OPS_CLASS>(Handle::msa->accumulate(x,y,z));
        }
        inline void accumulate(unsigned int x, unsigned int y, unsigned int z, const ENTRY& ent)
        {
            Handle::checkValid();
            Handle::msa->accumulate(x,y,z, ent);
        }

        void accumulate(unsigned x1, unsigned y1, unsigned z1, unsigned x2, unsigned y2, unsigned z2, const ENTRY *buf)
        {
            Handle::checkValid();
            CkAssert(x1 <= x2);
            CkAssert(y1 <= y2);
            CkAssert(z1 <= z2);

            CkAssert(x1 >= 0);
            CkAssert(y1 >= 0);
            CkAssert(z1 >= 0);

            CkAssert(x2 < Handle::msa->dim_x);
            CkAssert(y2 < Handle::msa->dim_y);
            CkAssert(z2 < Handle::msa->dim_z);

            unsigned i = 0;

            for (unsigned ix = x1; ix <= x2; ++ix)
                for (unsigned iy = y1; iy <= y2; ++iy)
                    for (unsigned iz = z1; iz <= z2; ++iz)
                        Handle::msa->accumulate(ix, iy, iz, buf[i++]);
        }

        inline Accumulable<ENTRY, ENTRY_OPS_CLASS> operator() (unsigned int x, unsigned int y, unsigned int z)
            { return accumulate(x,y,z); }
    };

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
    */
    inline MSA3D(unsigned x, unsigned y, unsigned z, unsigned int num_wrkrs, 
                 unsigned int maxBytes=MSA_DEFAULT_MAX_BYTES)
        : dim_x(x), dim_y(y), dim_z(z), initHandleGiven(false)
    {
        unsigned nEntries = x*y*z;
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

    static const int DEFAULT_SYNC_SINGLE = 0;
    static const bool MSA_CLEAR_ALL = true;

    inline Write getInitialWrite()
    {
        if (initHandleGiven)
            CmiAbort("Trying to get an MSA's initial handle a second time");

        //Write *w = new Write(*this);
        //sync();
        initHandleGiven = true;
        return Write(this);
    }

    inline Accum getInitialAccum()
    {
        if (initHandleGiven)
            CmiAbort("Trying to get an MSA's initial handle a second time");

        //Accum *a = new Accum(*this);
        //sync();
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
