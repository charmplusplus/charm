// emacs mode line -*- mode: c++; tab-width: 4 -*-
#ifndef MSA_DISTARRAY_H
#define MSA_DISTARRAY_H

#include "msa-DistPageMgr.h"

/* The MSA1D class represents a distributed shared array of items
   of data type ENTRY, nEntries total numer of ENTRY's, with
   ENTRIES_PER_PAGE data items per "page".  It is implemented as a
   Chare Array of pages, and a Group representing the local cache.
   The maximum size of the local cache in bytes is maxBytes. */
template<class ENTRY, unsigned int ENTRIES_PER_PAGE=MSA_DEFAULT_ENTRIES_PER_PAGE>
class MSA1D
{
protected:
    unsigned int nEntries;
    CacheGroup* cache;
    CProxy_CacheGroup cg;

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
    inline ENTRY* writeablePage(unsigned int page, unsigned int begin, unsigned int end)
    {
        return (ENTRY*)(cache->writeablePage(page, begin*sizeof(ENTRY), (end + 1)*sizeof(ENTRY) - 1));
    }

public:
    // @@ Needed for Jade
    inline MSA1D(){}

    inline MSA1D(unsigned int nEntries_, unsigned int num_wrkrs, unsigned int maxBytes=DEFAULT_MAX_BYTES) : nEntries(nEntries_)
    {
        // first create an array and the cache for the pages
        unsigned int nPages = (nEntries + ENTRIES_PER_PAGE - 1)/ENTRIES_PER_PAGE;
        unsigned int bytesPerPage = ENTRIES_PER_PAGE * sizeof(ENTRY);
        CProxy_PageArray pageArray = CProxy_PageArray::ckNew(nPages);
        cg = CProxy_CacheGroup::ckNew(nPages, bytesPerPage, pageArray, maxBytes, nEntries, num_wrkrs);
        pageArray.ckSetReductionClient(new CkCallback(CkIndex_CacheGroup::SyncDone(), cg));
        cache = cg.ckLocalBranch();
    }

    // We need to know the total number of workers across all
    // processors, and we also calculate the number of worker threads
    // running on this processor.
    //
    // blocking method, basically does a barrier until all workers
    // enroll.
    inline void enroll(int num_workers)
    {
        // @@ This is a hack to identify the number of MSA1D
        // threads on this processor.  This number is needed for sync.
        //
        // @@ What if a MSA1D thread migrates?
        cache->enroll(num_workers);
    }

    inline ~MSA1D()
    {
        // TODO: how to get rid of the cache group and the page array
        //(cache->getArray()).destroy();
        //cg.destroy();
        // TODO: calling FreeMem does not seem to work. Need to debug it.
        //cache->FreeMem();
    }

    inline MSA1D(CProxy_CacheGroup cg_) : cg(cg_)
    {
        cache = cg.ckLocalBranch();
        nEntries = cache->getNumEntries();
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

    inline unsigned int length() const { return nEntries; }

    inline const ENTRY& get(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return readablePage(page)[offset];
    }

    inline const ENTRY& get2(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        return readablePage2(page)[offset];
    }

    inline ENTRY& set(unsigned int idx)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        ENTRY* e=writeablePage(page, offset, offset);
        return e[offset];
    }

    inline void sync(int single=0) { cache->SyncReq(single); }

    inline CProxy_CacheGroup getCacheGroup() { return cg; }

    inline void accumulate(unsigned int idx, const ENTRY& ent)
    {
        unsigned int page = idx / ENTRIES_PER_PAGE;
        unsigned int offset = idx % ENTRIES_PER_PAGE;
        cache->accumulate(page, &ent, offset*sizeof(ENTRY), (offset + 1)*sizeof(ENTRY) - 1);
    }

    inline void FreeMem()
    {
        cache->FreeMem();
    }

    // non-blocking prefetch.
    // prefetch'd pages are locked into the cache
    inline void Prefetch(unsigned int start, unsigned int end)
    {
        if(start > end)
        {
            unsigned int temp = start;
            start = end;
            end = temp;
        }

        unsigned int page1 = start / ENTRIES_PER_PAGE;
        unsigned int page2 = end / ENTRIES_PER_PAGE;
        cache->Prefetch(page1, page2);
    }

    inline int WaitAll()    { return cache->WaitAll(); }

    inline void Unlock()    { return cache->UnlockPages(); }

    inline void Unlock(unsigned int start, unsigned int end)
    {
        if(start > end)
        {
            unsigned int temp = start;
            start = end;
            end = temp;
        }

        unsigned int page1 = start / ENTRIES_PER_PAGE; unsigned int offset1 = start % ENTRIES_PER_PAGE;
        unsigned int page2 = end / ENTRIES_PER_PAGE; unsigned int offset2 = end % ENTRIES_PER_PAGE;

        if(offset1 != 0) page1++;
        if(offset2 != ENTRIES_PER_PAGE - 1) page2--;

        cache->UnlockPages(page1, page2);
    }
};


// define a 2d distributed array based on the 1D array, support row major and column
// major arrangement of data
template<class ENTRY, unsigned int ENTRIES_PER_PAGE=MSA_DEFAULT_ENTRIES_PER_PAGE, int ROW_MAJOR=1>
class MSA2D : public MSA1D<ENTRY, ENTRIES_PER_PAGE>
{
protected:
    unsigned int rows, cols;

    // get the index of the given entry as per the row major/column major format
    unsigned int getIndex(unsigned int row, unsigned int col)
    {
        unsigned int index;

        if(ROW_MAJOR)
            index = row*cols + col;
        else
            index = col*rows + row;

        return index;
    }

public:
    // @@ Needed for Jade
    inline MSA2D() : MSA1D<ENTRY, ENTRIES_PER_PAGE>() {}

    inline MSA2D(unsigned int rows_, unsigned int cols_, unsigned int numwrkrs, unsigned int maxBytes=DEFAULT_MAX_BYTES) : MSA1D<ENTRY, ENTRIES_PER_PAGE>(rows_*cols_, numwrkrs, maxBytes)
    {
        rows = rows_; cols = cols_;
    }

    inline MSA2D(unsigned int rows_, unsigned int cols_, CProxy_CacheGroup cg_) : rows(rows_), cols(cols_), MSA1D<ENTRY, ENTRIES_PER_PAGE>(cg_) {}

    inline const ENTRY& get(unsigned int row, unsigned int col)
    {
        return MSA1D<ENTRY, ENTRIES_PER_PAGE>::get(getIndex(row, col));
    }

    inline const ENTRY& get2(unsigned int row, unsigned int col)
    {
        return MSA1D<ENTRY, ENTRIES_PER_PAGE>::get2(getIndex(row, col));
    }

    // MSA2D::
    inline ENTRY& set(unsigned int row, unsigned int col)
    {
        return MSA1D<ENTRY, ENTRIES_PER_PAGE>::set(getIndex(row, col));
    }

    inline void Prefetch(unsigned int start, unsigned int end)
    {
        // prefetch the start ... end rows/columns into the cache
        if(start > end)
        {
            unsigned int temp = start;
            start = end;
            end = temp;
        }

        unsigned int index1 = (ROW_MAJOR) ? getIndex(start, 0) : getIndex(0, start);
        unsigned int index2 = (ROW_MAJOR) ? getIndex(end, cols-1) : getIndex(rows-1, end);

        MSA1D<ENTRY, ENTRIES_PER_PAGE>::Prefetch(index1, index2);
    }

    inline void UnlockPages(unsigned int start, unsigned int end)
    {
        if(start > end)
        {
            unsigned int temp = start;
            start = end;
            end = temp;
        }

        unsigned int index1 = (ROW_MAJOR) ? getIndex(start, 0) : getIndex(0, start);
        unsigned int index2 = (ROW_MAJOR) ? getIndex(end, cols-1) : getIndex(rows-1, end);

        MSA1D<ENTRY, ENTRIES_PER_PAGE>::Unlock(index1, index2);
    }
};

#endif
