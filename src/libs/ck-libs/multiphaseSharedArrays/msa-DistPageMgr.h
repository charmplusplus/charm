// emacs mode line -*- mode: c++; tab-width: 4 -*-

#ifndef MSA_DISTPAGEMGR_H
#define MSA_DISTPAGEMGR_H

#include <charm++.h>
#include <string.h>
#include <list>
#include <stack>
#include "msa-common.h"
// forward decl needed in msa-DistPageMgr.ci, i.e. in msa.decl.h
template<class T> class DefaultEntry;
#include "msa.decl.h"

using namespace std;

inline int min(int a, int b)
{
    return (a<b)?a:b;
}

/**
 * class vmLRUPageReplacementPolicy
 * This class provides the functionality of least recently used page replacement policy.
 * It needs to be notified when a page is accessed using the pageAccessed() function and
 * a page can be selected for replacement using the selectPage() function.
 **/
class vmLRUReplacementPolicy
{
protected:
    const page_ptr_t* pageTable;
    const char* pageLock;
    list<unsigned int> stackOfPages;
    unsigned int lastPageAccessed;

public:
    inline vmLRUReplacementPolicy(const page_ptr_t* pageTable_, const char* pageLock_)
    : pageTable(pageTable_), pageLock(pageLock_), lastPageAccessed(MSA_INVALID_PAGE_NO) {}

    inline void pageAccessed(unsigned int page)
    {
        if(page != lastPageAccessed)
        {
            lastPageAccessed = page;

            // delete this page from the stack and push it at the top
            list<unsigned int>::iterator i;
            for(i = stackOfPages.begin(); i != stackOfPages.end(); i++)
                if(*i == page)
                    i = stackOfPages.erase(i);

            stackOfPages.push_back(page);
        }
    }

    inline unsigned int selectPage()
    {
        if(stackOfPages.size() == 0)
            return MSA_INVALID_PAGE_NO;

        // find a non-empty unlocked page to swap, delete all empty pages from the stack
        list<unsigned int>::iterator i = stackOfPages.begin();
        while(i != stackOfPages.end())
        {
            if(pageTable[*i] == NULL) i = stackOfPages.erase(i);
            else if(pageLock[*i]) i++;
            else break;
        }

        if(i != stackOfPages.end())
            return *i;
        else
            return MSA_INVALID_PAGE_NO;
    }
};

//================================================================

/** This is the interface used to perform the accumulate operation on
    an Entry.  T is the data type.  It may be a primitive one or a
    class.  It must support the default constructor, assignment, copy
    constructor, += operator, typecast from int 0, 1, and pup.
*/
template <class T>
class DefaultEntry {
public:
//     T assign(T &lhs, const T rhs){ lhs = rhs; return lhs;};
    inline virtual T accumulate(T& a, const T& b) { a += b;  return a; }
    // identity for initializing at start of accumulate
    inline virtual T getIdentity() { return (T)0; }
};

template <class T>
class ProductEntry : public DefaultEntry<T> {
public:
    inline T accumulate(T& a, const T& b) { a *= b;  return a; }
    inline T getIdentity() { return (T)1; }
};

template <class T, T minVal>
class MaxEntry : public DefaultEntry<T> {
public:
    inline T accumulate(T& a, const T& b) { a = (a<b)?b:a;  return a; }
    inline T getIdentity() { return minVal; }
};

//================================================================
// THIS CODE IS UNUSED AS YET.

/**
  Holds the untyped data for one MSA page.
  This is the interface CacheGroup uses to access a cached page.
  CacheGroup asks the templated code to create a MSA_Page
  for each new page, then talks to the page directly.
*/
class MSA_Page {
public:
	virtual ~MSA_Page();

	/**
	  Pack or unpack the data in this page.
	  Used to send and receive pages from the network
	  (or even disk, if somebody needs it.)
	*/
	virtual void pup(PUP::er &p) =0;

	/**
	  Merge this page's data into our own.
	  Only parts of this page may have been set.
	*/
	virtual void merge(MSA_Page &otherPage) =0;
};

/**
  Holds the typed data for one MSA page.
  Implementation of puppedPage used by the templated code.
*/
template <
	class ENTRY, 
	class MERGER=DefaultEntry<ENTRY>,
	unsigned int ENTRIES_PER_PAGE=MSA_DEFAULT_ENTRIES_PER_PAGE
>
class MSA_PageT : public MSA_Page {
	/** The contents of this page: array of ENTRIES_PER_PAGE items */
	ENTRY *data;
	/** Merger object */
	MERGER m;
public:
	MSA_PageT() {
		data=new ENTRY[ENTRIES_PER_PAGE];
		for (int i=0;i<ENTRIES_PER_PAGE;i++)
			data[i]=m.getIdentity();
	}
	virtual ~MSA_PageT() {
		delete[] data;
	}
	
	virtual void pup(PUP::er &p) {
		for (int i=0;i<ENTRIES_PER_PAGE;i++)
			p|data[i];
	}

	virtual void merge(MSA_Page &otherPage) {
		for (int i=0;i<ENTRIES_PER_PAGE;i++)
			m.accumulate(data[i],otherPage.data[i]);
	}

	// These accessors might be used by the templated code.
// 	inline ENTRY &operator[](int i) {return data[i];}
// 	inline const ENTRY &operator[](int i) const {return data[i];}
};

//================================================================

template <class ENTRY_TYPE, class ENTRY_OPS_CLASS>
class CacheGroup : public Group
{
protected:
    ENTRY_OPS_CLASS *entryOpsObject;
    unsigned int numberOfWorkerThreads;      // number of worker threads across all processors for this shared array
    // @@ migration?
    unsigned int numberLocalWorkerThreads;   // number of worker threads on THIS processor for this shared array
    unsigned int enrollDoneq;                 // has enroll() been done on this processor?

    ENTRY_TYPE** pageTable;          // the page table for this PE
    unsigned int nPages;            // number of pages
//     unsigned int bytesPerPage;      // bytes per page
    unsigned int numEntriesPerPage;      // number of entries per page

    char* pageState;                // for each available page, indicates whether its a read only
                                    // or read-write copy

    char* pageLock;                 // is this page locked (and hence can't be replaced) or not
                                    // 0 = not locked, 1 = locked

    CProxy_PageArray<ENTRY_TYPE, ENTRY_OPS_CLASS> pageArray;     // a proxy to the page array

    /** This defines the queue of threads waiting for an event or events **/
    typedef struct {
        CthThread thread;               // the current thread of execution
        // used in read fault, i.e. readablePage
        unsigned int nPagesWaiting;     // the number of pages for which the fetch
                                        // request has been issued
        // used in sync to keep track of number of flushed dirty pages
        unsigned int nChangesWaiting;   // the number of pages for which change notification
                                        // has been sent.  Used during sync().
        unsigned int suspended;         // is this thread suspended?
    } thread_info_t;

    CkVec<thread_info_t> threadList;        // a list of all the threads

public:
    typedef CkVec<int> thread_list_t;   // a list of the threads waiting for some event

protected:
    typedef struct
    {
        thread_list_t pageQueues[2];    // 0 for pages, 1 for changes
    } page_queue_entry_t;

    page_queue_entry_t* pageQueue;          // a queue of threads waiting for a given page

    stack<ENTRY_TYPE*> pagePool;     // a pool of unused pages
    vmLRUReplacementPolicy* replacementPolicy;

    // structure for the bounds of a single write
    typedef struct { unsigned int begin; unsigned int end; } writebounds_t;

    // a list of write bounds associated with a given page
    typedef list<writebounds_t> writelist_t;

    writelist_t** writes;           // the write lists for each page

    unsigned int bytes;             // bytes currently malloc'd
    unsigned int max_bytes;         // max allowable bytes to malloc
    unsigned int nEntries;          // number of entries for this array
    unsigned int syncAckCount;      // number of sync ack's we received
    int outOfBufferInPrefetch;      // flag to indicate if the last prefetch ran out of buffers

    int syncThreadCount;            // number of local threads that have issued Sync
    int syncThread;                 // thread that handles final sync call
    /*********************************************************************************/

    inline unsigned int indexToBytes(unsigned int index) { return index * sizeof(ENTRY_TYPE); }
    inline unsigned int bytesToIndex(unsigned long int bytes) { CkAssert(bytes%sizeof(ENTRY_TYPE) ==0); return bytes/sizeof(ENTRY_TYPE); }
    inline unsigned int getBPP() { return indexToBytes(numEntriesPerPage); }

    /*********************************************************************************/
    /** these routines deal with managing the page queue **/
    inline CthThread getThread() { return CthSelf(); }

    inline unsigned int getNumRegisteredThreads() const { return threadList.size(); }

    inline unsigned int getNumSuspendedThreads() const
    {
        unsigned int i, cnt = 0;;
        for(i = 0; i < threadList.size(); i++)
            cnt += threadList[i].suspended;
        return cnt;
    }

    inline int AddThreadToList()
    {
        // Is the current thread already present in the list of
        // threads?
        int entryIdx = GetThreadFromList();
        if(entryIdx == -1)
        {
            thread_info_t newEntry;
            newEntry.thread = getThread();
            newEntry.nPagesWaiting = 0;
            newEntry.nChangesWaiting = 0;
            //ckout << "[" << CkMyPe() << "] adding thread" << endl;
            threadList.push_back(newEntry);
            entryIdx = threadList.size() - 1;
        }

        // DONE @@ bug here, if entryIdx != -1, then we should
        // return entryIdx.  Will it ever happen?
        return entryIdx;
    }

    // increment the number of pages the thread is waiting on.
    // also add thread to the page queue; then thread is woken when the page arrives.
    inline void IncrementPagesWaiting(unsigned int page)
    {
        int entryIdx = AddThreadToList();
        threadList[entryIdx].nPagesWaiting++;

        //ckout << CkMyPe() << " pages waiting1 = " << threadList[entryIdx].nPagesWaiting << endl;
        // add the thread index to the page queue
        AddThreadToQueue(page, 0);
    }

    inline void IncrementChangesWaiting(unsigned int page)
    {

        int entryIdx = AddThreadToList();
        //ckout << "[" << CkMyPe() << "] incrementing changes waiting" << endl;
        threadList[entryIdx].nChangesWaiting++;

        AddThreadToQueue(page, 1);
    }

    inline unsigned int PagesWaiting()
    {
        int entryIdx = AddThreadToList();
        return threadList[entryIdx].nPagesWaiting;
    }

    inline unsigned int ChangesWaiting()
    {
        int entryIdx = AddThreadToList();
        return threadList[entryIdx].nChangesWaiting;
    }

    inline int GetThreadFromList()
    {
        CthThread self = getThread();
        unsigned int size = threadList.size();
        for(unsigned int i = 0; i < size; i++)
            if(threadList[i].thread == self)
                return i;
        return -1;
    }

    inline void AddThreadToQueue(unsigned int page, unsigned int queueType)
    {
        int entry = GetThreadFromQueue(page, queueType);
        if(entry == -1)
        {
            entry = AddThreadToList();
            //ckout << "[" << CkMyPe() << "] adding entry to page queue" << endl;
            pageQueue[page].pageQueues[queueType].push_back(entry);
        } else {
            ckout << "This should never happen\n";
            CkAssert (1==0);
            // The same thread cannot wait twice for the same page.
        }
    }

    inline int GetThreadFromQueue(unsigned int page, unsigned int queueType)
    {
        CthThread self = getThread();
        unsigned int size = pageQueue[page].pageQueues[queueType].size();
        for(unsigned int i = 0; i < size; i++)
        {
            if(threadList[pageQueue[page].pageQueues[queueType][i]].thread == self)
                return pageQueue[page].pageQueues[queueType][i];
        }

        return -1;
    }
/*******************************************************************************/
    //CacheGroup::
    inline void pageFault(unsigned int page, int why)
    {
        // get an empty buffer into which to fetch the new page
        ENTRY_TYPE* nu = pageTable[page];
        if (nu==0)
            nu = getBuffer(); // @@@

        if(nu == NULL)
        {
            //ckout << "Checking for locked pages" << endl;

            if(CheckForLockedPages())
                ckout << "[" << CkMyPe() << "] There are locked pages" << endl;

            CkAbort("Buffer overflow!");
        }

        // issue a request to fetch the new page
        pageTable[page] = nu;
        pageState[page] = why;

        if(why == Read_Fault)
        {
            // If the page has not been requested already, then request it.
            if (pageQueue[page].pageQueues[0].size()==0) {
                pageArray[page].GetPage(thisgroup, CkMyPe(), numEntriesPerPage);
                //ckout << "Requesting page first time"<< endl;
            } else
                ;//ckout << "Requesting page next time.  Skipping request."<< endl;
            //nPagesWaiting++;
            IncrementPagesWaiting(page);
            //ckout << "[" << CkMyPe() << "] suspending thread in pageFault" << endl;
            suspendThread();
            //ckout << "[" << CkMyPe() << "] Resumed thread in pageFault" << endl;
        }
        else if(why == Write_Fault)
        {
            memset(nu, 0, getBPP()); // @@@ Is this OK for variable-size pages?
        }
        else
        {
            zero(page, 0, numEntriesPerPage - 1);  // @@@
        }
    }

    // function to suspend the current thread
    inline void suspendThread()
    {
        // hope fully, thread has been added to the threadList
        // mark the current thread as suspended
        int currThread = AddThreadToList();
        //ckout << "[" << CkMyPe() << "] Suspending thread #" << currThread
        //      << ", changes waiting = [" << ChangesWaiting()
        //      << "], acks waiting = [" << PagesWaiting() << "]" << endl;
        threadList[currThread].suspended = 1;
        //ckout << "[" << CkMyPe() << "] suspending thread # " << currThread << endl;
        CthThread f1 = CthSelf();
        CthSuspend();
        CthThread f2 = CthSelf();
        CkAssert(f1 == f2);
        currThread = AddThreadToList();
        CkAssert(threadList[currThread].suspended == 0);
        //ckout << "[" << CkMyPe() << "] thread # " << currThread << " resumed " << endl;
        threadList[currThread].suspended = 0;
    }

    // function to resume the suspended thread if the conditions are ok
    inline void resumeThread(unsigned int idx)
    {
        CkAssert(idx < threadList.size());
        CkAssert(threadList[idx].suspended != 0);
//         if (threadList[idx].suspended == 0) {
//             ckout << "[" << CkMyPe() << "] Request to resume NON_SUSPENDED thread #" << idx
//                   << ".  Proceeding anyway." << endl;
//         }

        if(threadList[idx].nPagesWaiting == 0 && threadList[idx].nChangesWaiting == 0)
        {
//            CthSetStrategyDefault(threadList[idx].thread);
            threadList[idx].suspended = 0;
            CthAwaken(threadList[idx].thread);
        }
        else
        {
            ckout << "[" << CkMyPe() << "] Request to resume thread #" << idx << " not granted, pages waiting = "
                  << threadList[idx].nPagesWaiting << ", changes = " << threadList[idx].nChangesWaiting << endl;
        }
    }

    // Returns NULL if no page buffer available
    inline ENTRY_TYPE* getBuffer(int async=0) // @@@
    {
        ENTRY_TYPE* nu = NULL;

        // first try the page pool
        if(!pagePool.empty())
        {
            nu = pagePool.top();
            pagePool.pop();
        }

        // else try to allocate the buffer
        if(nu == NULL && bytes + getBPP() <= max_bytes)
        {
            nu = new ENTRY_TYPE[numEntriesPerPage];//malloc(bytesPerPage);
            bytes += getBPP();
        }

        // else swap one of the pages
        if(nu == NULL)
        {
            int pageToSwap = replacementPolicy->selectPage();

            /* here we need to ensure that the page selected is not a locked page */
            if(pageToSwap != MSA_INVALID_PAGE_NO)
            {
                CkAssert(pageLock[pageToSwap] == 0);
                relocatePage(pageToSwap, async);
                nu = pageTable[pageToSwap];
                pageTable[pageToSwap] = 0;
            }
        }

        // otherwise return NULL

        return nu;
    }

    inline void relocatePage(unsigned int page, int async)
    {
        //CkAssert(pageTable[page]);
        if(pageState[page] == Write_Fault || pageState[page] == Accumulate_Fault)
        {
            // the page to be swapped is a writeable page. So inform any
            // changes this node has made to the page manager
            sendChangesToPageArray(page, async);
        }
    }

    // this function assumes that there are no overlapping writes in the list
    inline void sendChangesToPageArray(const unsigned int page, const int async)
    {
        sendNonRLEChangesToPageArray(page, async);
    }

    inline void sendNonRLEChangesToPageArray(const unsigned int page, const int async) // @@@
    {
//         pageArray[page].ReceivePage(pageTable[page], numEntriesPerPage, thisgroup, CkMyPe(), pageState[page]);

        typename list<writebounds_t>::iterator iter;
        unsigned int i;
        // do a run length encoding of the writes. This encoding has the
        // following format:
        // <unsigned int>  : number of runs
        // <unsigned int>+ : offset of each run in the data stream (except first)
        // <for each run: start and end offsets of write followed by data>

        // if there are 'k' runs and there are 'b' bytes per page, then the length
        // of the entire encoding is bounded by:
        // k*sizeof(unsigned int) + k*2*sizeof(unsigned int) + b

        // allocate a buffer for creating the run
        unsigned int numRuns = writes[page]->size();
        unsigned int buffSize = 3*sizeof(unsigned int)*numRuns + getBPP();
        char* buffer = (char*)calloc(1, buffSize);

        //ckout << "[" << CkMyPe() << "] doing RLE encoding for page " << page << " with " << numRuns << "run(s)" << endl;

        if(writes[page] == NULL || writes[page]->size() == 0)
        {
            //ckout << "[" << CkMyPe() << "] send changes request for empty page " << endl;
            return;
        }

        // check the writes list
        for(iter = writes[page]->begin(); iter != writes[page]->end(); iter++)
            CkAssert(iter->begin < getBPP() && iter->end < getBPP() && iter->begin < iter->end);

        unsigned int currOffset = numRuns*sizeof(unsigned int);
        for(iter = writes[page]->begin(), i = 0; iter != writes[page]->end(); i++)
        {
            // write the offset of this write
            ((unsigned int*)buffer)[i] = currOffset;
            unsigned int begin = iter->begin;
            unsigned int end = iter->end;
//             CkAssert(begin < getBPP() && end < getBPP());
            *((unsigned int*)(buffer + currOffset)) = begin; currOffset += sizeof(unsigned int);
            *((unsigned int*)(buffer + currOffset)) = end; currOffset += sizeof(unsigned int);
//             memcpy(buffer + currOffset, (char*)(pageTable[page]) + begin, end - begin + 1);
            ENTRY_TYPE *bufPtr = (ENTRY_TYPE *)(buffer + currOffset);
            ENTRY_TYPE *pagePtr = (ENTRY_TYPE *)((char*)(pageTable[page]) + begin);
            for(int j=0; j< bytesToIndex(end - begin + 1); j++)
                bufPtr[j] = pagePtr[j];
            currOffset += end - begin + 1;
            iter = writes[page]->erase(iter);
        }

        //ckout << "RLE encoding of " << currOffset << " bytes" << ", num writes for this page now = " << writes[page]->size() << endl;

        *((unsigned int*)buffer) = numRuns;

        // send the RLE'd buffer to the page array

        CkAssert(page < nPages); // 0 <= page always, bcoz page is uint
        //ckout << "[" << CkMyPe() << "] sending page " << page << "to page array " << endl;
        pageArray[page].ReceiveRLEPage(buffer, currOffset, getBPP(), thisgroup, CkMyPe(), pageState[page]);
        free(buffer);
        //nChangesWaiting++;
        IncrementChangesWaiting(page);

        // TODO: Is this acknowledgement really necessary ?
        if(!async)
        {
            //ckout << "[" << CkMyPe() << "] suspending thread in send changes to page array" << endl;
            suspendThread();
            //ckout << "[" << CkMyPe() << "] resumed thread in send changed to page array" << endl;
        }
    }

    // this function assumes that there are no overlapping writes in the list
    inline void sendRLEChangesToPageArray(const unsigned int page, const int async) // @@@
    {
        typename list<writebounds_t>::iterator iter;
        unsigned int i;
        // do a run length encoding of the writes. This encoding has the
        // following format:
        // <unsigned int>  : number of runs
        // <unsigned int>+ : offset of each run in the data stream (except first)
        // <for each run: start and end offsets of write followed by data>

        // if there are 'k' runs and there are 'b' bytes per page, then the length
        // of the entire encoding is bounded by:
        // k*sizeof(unsigned int) + k*2*sizeof(unsigned int) + b

        // allocate a buffer for creating the run
        unsigned int numRuns = writes[page]->size();
        unsigned int buffSize = 3*sizeof(unsigned int)*numRuns + getBPP();
        char* buffer = (char*)malloc(buffSize);

        //ckout << "[" << CkMyPe() << "] doing RLE encoding for page " << page << " with " << numRuns << "run(s)" << endl;

        if(writes[page] == NULL || writes[page]->size() == 0)
        {
            //ckout << "[" << CkMyPe() << "] send changes request for empty page " << endl;
            return;
        }

        // check the writes list
        for(iter = writes[page]->begin(); iter != writes[page]->end(); iter++)
            CkAssert(iter->begin < getBPP() && iter->end < getBPP() && iter->begin < iter->end);

        unsigned int currOffset = numRuns*sizeof(unsigned int);
        for(iter = writes[page]->begin(), i = 0; iter != writes[page]->end(); i++)
        {
            // write the offset of this write
            ((unsigned int*)buffer)[i] = currOffset;
            unsigned int begin = iter->begin;
            unsigned int end = iter->end;
            CkAssert(begin < getBPP() && end < getBPP());
            *((unsigned int*)(buffer + currOffset)) = begin; currOffset += sizeof(unsigned int);
            *((unsigned int*)(buffer + currOffset)) = end; currOffset += sizeof(unsigned int);
            memcpy(buffer + currOffset, (char*)(pageTable[page]) + begin, end - begin + 1);
            currOffset += end - begin + 1;
            iter = writes[page]->erase(iter);
        }

        //ckout << "RLE encoding of " << currOffset << " bytes" << ", num writes for this page now = " << writes[page]->size() << endl;

        *((unsigned int*)buffer) = numRuns;

        // send the RLE'd buffer to the page array

        CkAssert(page < nPages); // 0 <= page always, bcoz page is uint
        //ckout << "[" << CkMyPe() << "] sending page " << page << "to page array " << endl;
        pageArray[page].ReceiveRLEPage(buffer, currOffset, getBPP(), thisgroup, CkMyPe(), pageState[page]);
        free(buffer);
        //nChangesWaiting++;
        IncrementChangesWaiting(page);

        // TODO: Is this acknowledgement really necessary ?
        if(!async)
        {
            //ckout << "[" << CkMyPe() << "] suspending thread in send changes to page array" << endl;
            suspendThread();
            //ckout << "[" << CkMyPe() << "] resumed thread in send changed to page array" << endl;
        }
    }

    inline void createWriteList(unsigned int page)
    {
        if(writes[page] == NULL)
            writes[page] = new writelist_t;
    }

    // TODO: combine multiple consecutive spans into a single span
    inline void addToWriteList(unsigned int page, unsigned int beginByte, unsigned int endByte) // @@@@
    {
        createWriteList(page);

        CkAssert(beginByte < getBPP() && endByte < getBPP() && beginByte < endByte);

        // combine consecutive spans into a single span
        typename list<writebounds_t>::iterator i;
        for(i = writes[page]->begin(); i != writes[page]->end(); i++)
        {
            if(i->begin <= beginByte && i->end >= endByte)
                return;
            else if(i->begin == endByte + 1)
            {
                i->begin = beginByte;
                if(pageState[page] == Accumulate_Fault)
                    zero(page, bytesToIndex(beginByte), bytesToIndex(endByte));
                return;
            }
            else if(i->end == beginByte - 1)
            {
                i->end = endByte;
                CkAssert(i->begin < i->end);
                if(pageState[page] == Accumulate_Fault)
                    zero(page, bytesToIndex(beginByte), bytesToIndex(endByte));
                return;
            }
        }

        writebounds_t b;
        b.begin = beginByte; b.end = endByte;
        writes[page]->push_back(b);
        if(pageState[page] == Accumulate_Fault)
            zero(page, bytesToIndex(beginByte), bytesToIndex(endByte));
    }

    // begin, end are indexes
    //
    // CacheGroup::
    void combine(unsigned int page, const ENTRY_TYPE* entryPtr, unsigned int begin, unsigned int end)
    {
        ENTRY_TYPE* pagePtr = pageTable[page] + begin;
        for(unsigned int i = 0; i < (end - begin + 1); i++)
            entryOpsObject->accumulate(pagePtr[i], entryPtr[i]);
    }

    // begin, end are indexes
    // CacheGroup::
    //
    // @@ Rename this function.  zero actually means identity, e.g. if
    // accumulate is SUM, then identity is 0; if product, identity is
    // 1.
    void zero(unsigned int page, unsigned int begin, unsigned int end)
    {
        // @@ assert ((end-begin +1)%sizeof(double) == 0)
        ENTRY_TYPE* pagePtr = pageTable[page] + begin;
        for(unsigned int i = 0; i < (end - begin + 1); i++)
            pagePtr[i] = entryOpsObject->getIdentity();
    }

    // Returns 1 if any page is locked, 0 if no page is locked.
    int CheckForLockedPages()
    {
        for(unsigned int i = 0; i < nPages; i++)
            if(pageLock[i])
                return 1;
        return 0;
    }

public:

    // 
    //
    // CacheGroup::
    inline CacheGroup(unsigned int nPages_, unsigned int bytesPerPage_, CkArrayID pageArrayID,
                      unsigned int max_bytes_, unsigned int nEntries_, unsigned int numberOfWorkerThreads_)
        : nEntries(nEntries_), numberOfWorkerThreads(numberOfWorkerThreads_)
    {
        numberLocalWorkerThreads = 0;  // populated after enroll
        enrollDoneq = 0;  // populated after enroll

        nPages = nPages_;
        //bytesPerPage = bytesPerPage_;
        CkAssert(bytesPerPage_%sizeof(ENTRY_TYPE) == 0);
        numEntriesPerPage = bytesToIndex(bytesPerPage_);
        pageArray = CProxy_PageArray<ENTRY_TYPE, ENTRY_OPS_CLASS>(pageArrayID);
        bytes = 0;
        outOfBufferInPrefetch = 0;
        max_bytes = max_bytes_;
        entryOpsObject = new ENTRY_OPS_CLASS();

        // initialize the page table
        typedef ENTRY_TYPE* entry_type_ptr;
        pageTable = new entry_type_ptr[nPages];
        pageState = new char[nPages];
        pageLock = new char[nPages];
        writes = new writelist_t*[nPages];
        pageQueue = new page_queue_entry_t[nPages];

        // this is the number of sync ack's received till yet
        syncAckCount = 0;
        syncThreadCount = 0;
        syncThread = -1;

        replacementPolicy = new vmLRUReplacementPolicy((page_ptr_t*) pageTable, pageLock);

        for(unsigned int i = 0; i < nPages; i++)
        {
            pageTable[i] = 0;
            writes[i] = 0;
            pageLock[i] = 0;
            pageState[i] = Uninit_State;
        }
    }

    // CacheGroup::
    inline ~CacheGroup()
    {
        FreeMem();

        delete[] pageTable; pageTable = 0;
        delete[] pageState; pageState = 0;
        delete[] pageLock; pageLock = 0;
        delete[] writes; writes = 0;
        delete[] pageQueue; pageQueue = 0;
    }

    /* To change the accumulate function */
    inline void changeEntryOpsObject(ENTRY_OPS_CLASS *e) {
        entryOpsObject = e;
        pageArray.changeEntryOpsObject(e);
    }

    // CacheGroup::
    inline const ENTRY_TYPE* readablePage(unsigned int page)
    {
        // If the page is not allocated, or even if it is, but there's
        // someone waiting for it to come back, then call pageFault.
//         if (page == 6)
//             ckout << "CacheGroup::readablePage " << page
//                   << " " << pageTable[page]
//                   << endl;
        if (!(pageState[page]==Read_Fault || pageState[page]==Uninit_State))
            ckout << "MSA Runtime error: Attempting to read from a page that is still in another mode." << endl;

        if ((pageTable[page] == 0) || (pageQueue[page].pageQueues[0].size()!=0))
            pageFault(page, Read_Fault);
        replacementPolicy->pageAccessed(page);
        return pageTable[page];
    }

    // CacheGroup::
    //
    // known local page
    inline const void* readablePage2(unsigned int page)
    {
        return pageTable[page];
    }

    // CacheGroup::
    // obtains a write lock on the page.
    // begin and end are the offset in bytes into the page.
    // the caller is assumed to write only between begin and end ??
    inline ENTRY_TYPE* writeablePage(unsigned int page, unsigned int beginByte, unsigned int endByte)
    { // @@@@
//         if (page == 6 && beginByte <= 100)
//             ckout << "CacheGroup::writeablePage " << page << " " << beginByte
//                   << " " << endByte
//                   << " " << pageTable[page]
//                   << endl;

        if (!(pageState[page]==Write_Fault || pageState[page]==Uninit_State))
            ckout << "MSA Runtime error: Attempting to write to a page that is still in another mode." << endl;

        if(pageTable[page] == 0)
            pageFault(page, Write_Fault);
        else
            pageState[page] = Write_Fault;

        // NOTE: Since we assume write once semantics, i.e. between two calls to sync,
        // either there can be no write to a location or a single write to a location,
        // a readable page will suffice as a writeable page too, because no one else
        // is going to write to this location. In reality, two locations on the same
        // page can be written by two different threads, in which case we will need
        // an index into the page giving which entry has been modified by this write
        // for that we also include in the writeablePage(), the start and end bytes
        // of the area that will be written by this call. Now we can uniquely identify
        // which bytes within a page are being modified. We store these indices in a
        // list.

        addToWriteList(page, beginByte, endByte);
        replacementPolicy->pageAccessed(page);
        return pageTable[page];
    }

    // CacheGroup::
    inline void accumulate(unsigned int page, const void* entry, unsigned int beginByte, unsigned int endByte)
    {
        if (!(pageState[page]==Accumulate_Fault || pageState[page]==Uninit_State))
            ckout << "MSA Runtime error: Attempting to accumulate to a page that is still in another mode." << endl;

        // What if multiple threads on pe access this page and it needs
        // to be fetched?  See readablePage().  Actually, its OK,
        // since for accumulate, no page is fetched.
        if(pageTable[page] == 0)
            pageFault(page, Accumulate_Fault);
        else
            pageState[page] = Accumulate_Fault;

        addToWriteList(page, beginByte, endByte);
        combine(page, entry, bytesToIndex(beginByte), bytesToIndex(endByte));
        replacementPolicy->pageAccessed(page);
    }

    // nEntriesInPage_ = num entries being sent (0 for empty page, num entries otherwise)
    inline void ReceivePage(unsigned int page, ENTRY_TYPE* pageData, int isEmpty, int nEntriesInPage_)
    {
        CkAssert(nEntriesInPage_ <=  numEntriesPerPage);
        // how many threads are waiting for this page?
        unsigned int size = pageQueue[page].pageQueues[0].size();
        //ckout << CkMyPe() << "Received page " << page << ", queue size = " << size << endl;
        CkAssert(size > 0); //otherwise, why should we be receiving the page?

        // the page we requested for has been received
        if(!isEmpty)
        {
            /* check if a page has been allocated. if not probably this was a false/unused prefetch */
            if(pageTable[page])
                for(unsigned int i = 0; i < numEntriesPerPage; i++)
                    pageTable[page][i] = pageData[i]; // @@@, calls assignment operator
        }
        else
        {
            // the page we requested for is empty, so we week it uninitialized
            // TODO: this should ideally do type specific initialization
            // memset(pageTable[page], 0, bytesPerPage);
            // @@ do we need to allocate the page here?
            if(pageTable[page])
                memset(pageTable[page], 0, getBPP()); // @@@@ OK for now
        }

        // wake up the threads that are waiting for this page.
        for(int i = size-1; i >=0; i--) {
            int thr = pageQueue[page].pageQueues[0][i];
            CkAssert(threadList[thr].nChangesWaiting==0); // Cannot be participating in sync during a read fault.
            //ckout << CkMyPe() << "thread" << thr << ",pages waiting = " << threadList[thr].nPagesWaiting << endl;
            CkAssert(threadList[thr].nPagesWaiting !=0);
            threadList[thr].nPagesWaiting--;
            if(threadList[thr].nPagesWaiting == 0)
            {
                //ckout << "Resuming thread #" << pageQueue[page].pageQueues[0][i] << endl;
                resumeThread(thr);
                // DONE @@ shouldn't we clear pageQueue[page].pageQueues[0] ?
                // DONE @@ do correctly, go i=size to i=0
                pageQueue[page].pageQueues[0].remove(i);
            }
        }

        // @@ In cases of prefetch, the following will not work, so
        // once you start allowing prefetch, just take this out.
        CkAssert(pageQueue[page].pageQueues[0].size() == 0);
    }

    // This EP is invoked during sync to acknowledge that a dirty page
    // has been received and handled by the page owner.  We keep track
    // of the number of ack's yet to arrive in nChangesWaiting.  Once
    // all the dirty pages have been ack'd, we awaken the thread that
    // flushed the page.
    //
    // CacheGroup::
    inline void AckRLEPage(unsigned int page)
    {
        // Number of threads waiting for this page.  Should be exactly
        // one thread, since only one thread per pe does the flush.
        unsigned int size = pageQueue[page].pageQueues[1].size();
        CkAssert(size == 1);

        //ckout << "[" << CkMyPe() << "] ack for page " << page << ", size of queue = " << size << endl;

        // Go through the list of threads waiting for this page to be
        // processed
        for(int i = size-1; i >= 0; i--)
        {
            int thr = pageQueue[page].pageQueues[1][i];
            CkAssert(threadList[thr].nPagesWaiting==0); // Cannot be participating in read fault during a sync.
            threadList[thr].nChangesWaiting--;
            //ckout << "[" << CkMyPe() << "] thread #" << thr
            //      << " nChangesWaiting = " << threadList[thr].nChangesWaiting << endl;
            if(threadList[thr].nChangesWaiting == 0) {
                resumeThread(thr);
                pageQueue[page].pageQueues[1].remove(i);
            }
        }
    }

    // CacheGroup::
    // synchronize all the pages and also clear up the cache
    inline void SyncReq(int single)
    {
        if(single)
        {
            CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> self = thisgroup;
            /*ask all the caches to send their updates to the page array, but we don't need to empty the caches on the other PEs*/
            SingleSync();
            EmptyCache();

            if(ChangesWaiting() != 0)
            {
                //ckout << "[" << CkMyPe() << "] suspending thread in SyncSingle" << endl;
                suspendThread();
                //ckout << "[" << CkMyPe() << "] resumed thread in SyncSingle" << endl;
            }
        }
        else
            Sync();
    }

    // CacheGroup::
    inline void FlushCache()
    {
        int currThread = AddThreadToList();
        CkAssert(threadList[currThread].nChangesWaiting == 0);
        // flush the local cache
        // for each writeable page, send that page to the array element
        for(unsigned int i = 0; i < nPages; i++)
        {
            if(pageTable[i] && (pageState[i] == Write_Fault || pageState[i] == Accumulate_Fault))
            {
//                 ckout << "[" << CkMyPe() << "]" << "FlushCache sending page " << i << endl;
                sendChangesToPageArray(i, 1);
            }
        }
        //ckout << "[" << CkMyPe() << "]" << "FlushCache nChangesWaiting = "
        //      << threadList[currThread].nChangesWaiting << endl;
    }

    // CacheGroup::
    void EmptyCache()
    {
        /* just makes all the pages empty, assuming that the data in those pages has been flushed to the owners */
        for(unsigned int i = 0; i < nPages; i++)
        {
            if(pageTable[i])
            {
                pagePool.push(pageTable[i]);
                pageTable[i] = 0;
                pageLock[i] = 0;
                pageState[i] = Uninit_State;
            }
        }
    }

    // CacheGroup::
    inline void enroll(unsigned int num_workers)
    {
        CkAssert(num_workers == numberOfWorkerThreads); // just to verify
        CkAssert(enrollDoneq == 0);
        numberLocalWorkerThreads++;
        // @@ how to ensure that enroll is called only once?

        CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> self = thisgroup;
        //ckout << "[" << CkMyPe() << "] sending sync ack to PE 0" << endl;
        self[0].enrollAck();
        //ckout << "[" << CkMyPe() << "] suspening thread in Sync() " << endl;
        suspendThread();
        //ckout << "[" << CkMyPe() << "] rsuming thread in Sync()" << endl;

        CkAssert(enrollDoneq == 1);
        return;
    }

    inline void enrollAck()
    {
        CkAssert(CkMyPe() == 0);  // enrollAck is only called on PE 0
        CkAssert(enrollDoneq == 0);  // prevent multiple enroll operations

        syncAckCount++;
        //ckout << "[" << CkMyPe() << "] SyncAckcount = " << syncAckCount << endl;
        if(syncAckCount == numberOfWorkerThreads) {
//             ckout << "[" << CkMyPe() << "]" << "Enroll operation is almost done" << endl;
            syncAckCount = 0;
            enrollDoneq = 1;
            CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> self = thisgroup;
            // What if fewer worker threads than pe's ?  Handled in
            // enrollDone.
            for(unsigned int i = 0; i < CmiNumPes(); i++){
                self[i].enrollDone();
            }
        }
    }

    inline void enrollDone()
    {
//         ckout << "[" << CkMyPe() << "] enrollDone.  Waking threads."
//               <<  " numberOfWorkerThreads=" << numberOfWorkerThreads
//               <<  " local=" << numberLocalWorkerThreads << endl;
        enrollDoneq = 1;
        // What if fewer worker threads than pe's ?  This pe may not
        // have any worker threads.
        if (numberLocalWorkerThreads > 0)
            for(unsigned int i = 0; i < threadList.size(); i++)
                if (threadList[i].suspended != 0)
                    resumeThread(i);
    }

    // CacheGroup::
    inline void SingleSync()
    {
        /* a single thread issued a sync call with all = 1. The first thing to do is to flush the local cache */
        FlushCache();
    }

    // CacheGroup::
    inline void Sync()
    {
        syncThreadCount++;
        //ckout << "[" << CkMyPe() << "] syncThreadCount = " << syncThreadCount << ", registered threads = " << getNumRegisteredThreads()
        //    << ", number of suspended threads = " << getNumSuspendedThreads() << endl;

        // First, all threads on this processor need to reach the sync
        // call; only then can we proceed with merging the data.  Only
        // the last thread on this processor needs to do the FlushCache,
        // etc.  Others just suspend until the sync is over.
        if(syncThreadCount < numberLocalWorkerThreads)
        {
            /* add this thread to the queue of threads waiting for sync to complete and suspend */
            int idx = AddThreadToList();

            // @@ What does this do ?
//          static int first = 0;
//          if(CkMyPe() == 0 && !first)
//          {
//              idx = (idx == 0) ? 1 : 0;
//              //ckout << "[" << CkMyPe() << "] resuming thread #" << idx << endl;
//              resumeThread(idx);
//              first = 1;
//          }

            //ckout << "[" << CkMyPe() << "] Thread suspended in sync" << endl;
            suspendThread();
//          first = 0;
            //ckout << "[" << CkMyPe() << "] Thread #"<<idx<<"resumed in sync" << endl;
            return;
        }

        //ckout << "[" << CkMyPe() << "] Sync started" << endl;

        // flush the cache asynchronously and also empty it
        FlushCache();
        // idea: instead of invalidating the pages, switch it to read
        // mode. That will not work, since the page may have also been
        // modified by another thread.
        EmptyCache();

        //ckout << "[" << CkMyPe() << "] sent all pages to page array, waiting for ack" << endl;
        if(ChangesWaiting() != 0)
        {
            //ckout << "[" << CkMyPe() << "] Suspended in Sync" << endl;
            suspendThread();
            //ckout << "[" << CkMyPe() << "] resumed in sync" << endl;
        }
        //ckout << "[" << CkMyPe() << "] page ack's received, proceeding with synchronization" << endl;

        // at this point, the sync's across the group should
        // synchronize among themselves by each one sending
        // a sync acknowledge message to PE 0. (this is like
        // a reduction over a group
        if(CkMyPe() != 0)
        {
            CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> self = thisgroup;
            //ckout << "[" << CkMyPe() << "] sending sync ack to PE 0" << endl;
            self[0].SyncAck();
            //ckout << "[" << CkMyPe() << "] suspening thread in Sync() " << endl;
            suspendThread();
            //ckout << "[" << CkMyPe() << "] rsuming thread in Sync()" << endl;

            syncThreadCount = 0;
            syncAckCount = 0;
            syncThread = -1;
            return;
        }
        else
        {
            syncAckCount++;
             // DONE @@ what if fewer worker threads than pe's ?
            // @@ what if fewer worker threads than pe's and >1 threads on 1 pe?
            if(syncAckCount < min(numberOfWorkerThreads, CkNumPes()) )
            {
                ckout << "reached " << endl;
                //ckout << "Suspending thread on PE 0 in Synb, syncAckCount = " << syncAckCount << endl;
                syncThread = AddThreadToList();
                suspendThread();
                //ckout << "resumed in thread on PE 0 in Sync "<< endl;
            }

            pageArray.Sync();
            suspendThread();
            //ckout << "resumed thread on PE 0 in Sync "<< endl;

            syncThreadCount = 0;
            syncThread = -1;
            syncAckCount = 0;

            return;
        }
    }

    inline unsigned int getNumEntries() { return nEntries; }
    inline CProxy_PageArray<ENTRY_TYPE, ENTRY_OPS_CLASS> getArray() { return pageArray; }

    // TODO: Can this SyncAck and other simple Acks be made efficient?
    inline void SyncAck()
    {
        CkAssert(CkMyPe() == 0);  // SyncAck is only called on PE 0
        syncAckCount++;
        //ckout << "[" << CkMyPe() << "] SyncAckcount = " << syncAckCount << endl;
        // DONE @@ what if fewer worker threads than pe's ?
        // @@ what if fewer worker threads than pe's and >1 threads on 1 pe?
        if(syncAckCount == min(numberOfWorkerThreads, CkNumPes()))
            resumeThread(syncThread);
    }

    inline void SyncDone()
    {
        //ckout << "[" << CkMyPe() << "] Sync Done indication" << endl;
        //ckout << "[" << CkMyPe() << "] Sync Done indication" << endl;
        for(unsigned int i = 0; i < threadList.size(); i++)
            if (threadList[i].suspended !=0) {
                //ckout << "[" << CkMyPe() << "]" << "Resuming thread " << i << endl;
                resumeThread(i);
            }
    }

    inline void FreeMem()
    {
        for(unsigned int i = 0; i < nPages; i++)
        {
            if(pageTable[i])
                delete [] pageTable[i]; // @@@
            if(writes[i])
                delete writes[i];
        }

        UnlockPages();

        while(!pagePool.empty())
        {
            delete [] pagePool.top();  // @@@
            pagePool.pop();
        }

        // TODO: delete thread list
    }

    /**
     * Issue a prefetch request for the given range of pages. These pages will
     * be locked into the cache, so that they will not be swapped out.
     */
    inline void Prefetch(unsigned int pageStart, unsigned int pageEnd)
    {
        /* prefetching is feasible only if we we did not encounter an out
         * of buffer condition in the previous prefetch call
         */
        if(!outOfBufferInPrefetch)
        {
            //ckout << "prefetching pages " << pageStart << " through " << pageEnd << endl;
            for(unsigned int p = pageStart; p <= pageEnd; p++)
            {
                if(NULL == pageTable[p])
                {

                    /* relocate the buffer asynchronously */
                    ENTRY_TYPE* nu = getBuffer(1);
                    if(NULL == nu)
                    {
                        /* signal that sufficient buffer space is not available */
                        outOfBufferInPrefetch = 1;
                        break;
                    }

                    pageTable[p] = nu;
                    pageState[p] = Read_Fault;

                    pageArray[p].GetPage(thisgroup, CkMyPe(), numEntriesPerPage);
                    IncrementPagesWaiting(p);
                    //ckout << "Prefetch page" << p << ", pages waiting = " << nPagesWaiting << endl;
                    /* don't suspend the thread */
                }

                /* mark the page as being locked */
                pageLock[p] = 1;
            }
        }
    }

    /**
     * Wait for all the prefetch pages to be fetched into the cache.
     * Returns: 0 if prefetch successful, 1 if not
     */
    inline int WaitAll(void)
    {
        if(outOfBufferInPrefetch)
        {
            // we encountered out of buffer in the previous prefetch call, return error
            outOfBufferInPrefetch = 0;

            if(PagesWaiting() != 0) suspendThread();

            UnlockPages();

            return 1;
        }
        else
        {
            // prefetch requests have been successfully issued already, so suspend the
            // thread and wait for completion
            outOfBufferInPrefetch = 0;
            //ckout << "[" << CkMyPe() << "] suspending thread in WaitAll" << endl;
            if(PagesWaiting() != 0) suspendThread();
            //ckout << "[" << CkMyPe() << "] resumed in WAITALL" << endl;
            return 0;
        }
    }

    /**
     * unlock all the pages locked in the cache
     */
    inline void UnlockPages()
    {
        // add all the locked pages to page replacement policy
        for(unsigned int i = 0; i < nPages; i++)
        {
            if(pageLock[i])
                replacementPolicy->pageAccessed(i);
            pageLock[i] = 0;
        }
    }

    /**
     * unlock the given pages
     */
    inline void UnlockPages(unsigned int startPage, unsigned int endPage)
    {
        memset(pageLock + startPage, 0, (endPage - startPage + 1)*sizeof(char));
    }

    inline void emitBufferValue(unsigned int pageNum, unsigned int offset)
    {
        CkAssert( pageNum < nPages );
        CkAssert( offset < numEntriesPerPage );

        if (pageTable[pageNum] == 0)
            ckout << "emitBufferValue: page " << pageNum << " not available in local cache." << endl;
        else
            ckout << "emitBufferValue: [" << pageNum << "," << offset << "] = " << pageTable[pageNum][offset] << endl;
    }

    inline void emit(int offset)
    {
        pageArray[6].emit(offset);
    }

};

// each element of this array is responsible for managing
// the information about a single page. It is in effect the
// "owner" as well as the "manager" for that page.
//
// The size of the page in bytes is not stored, its assumed
// to be explicitly given with each call.
template<class ENTRY_TYPE, class ENTRY_OPS_CLASS> 
class PageArray : public ArrayElement1D
{
protected:
//     page_ptr_t page;                // the page data, @@@
    ENTRY_TYPE *epage;
//     unsigned int pageSize;          // the size of the page in bytes
    unsigned int numEntries;
    unsigned char accumInit;        // flag to indicate whether the page has been initialized for
                                    // accumulate mode
    ENTRY_OPS_CLASS *entryOpsObject;

    unsigned int pageNo() { return thisIndex; }

    inline void allocatePage(unsigned int nEntriesInPage_) // @@@
    {
        if(epage == NULL)
        {
//             page = calloc(1, bytesPerPage); // @@@
            epage = new ENTRY_TYPE[nEntriesInPage_];
//             pageSize = bytesPerPage;
            numEntries = nEntriesInPage_;
            accumInit = 0;
        }
    }

    // begin and end are indexes into the page.
    inline void set(const ENTRY_TYPE* buffer, unsigned int begin, unsigned int end)
    {
        for(unsigned int i = 0; i < (end - begin + 1); i++) {
            epage[begin + i] = buffer[i]; // @@@, calls assignment operator
        }
    }

    // PageArray::
    inline void combine(const ENTRY_TYPE* buffer, unsigned int begin, unsigned int end)
    {
        ENTRY_TYPE* pagePtr = epage + begin;
        for(unsigned int i = 0; i < (end - begin + 1); i++)
            entryOpsObject->accumulate(pagePtr[i], buffer[i]);
    }

    // PageArray::
    inline void zero()
    {
        for(unsigned int i = 0; i < numEntries; i++)
            epage[i] = entryOpsObject->getIdentity();
    }

public:
    inline PageArray() : epage(NULL), numEntries(0), accumInit(0) { entryOpsObject= new ENTRY_OPS_CLASS();}
    inline PageArray(CkMigrateMessage* m) { delete m; }
    inline ~PageArray()
    {
//         if(page) free(page); // @@@
        if(epage) delete [] epage;
    }

    /* To change the accumulate function */
    inline void changeEntryOpsObject(ENTRY_OPS_CLASS *e) {
        entryOpsObject = e;
    }

    // pe = to which to send page
    inline void GetPage(CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> cache, int pe, unsigned int nEntriesInPage_)
    {
        if(epage == NULL)
            cache[pe].ReceivePage(pageNo(), (ENTRY_TYPE*)NULL, 1, 0); // send empty page
        else
            cache[pe].ReceivePage(pageNo(), epage, 0, nEntriesInPage_);  // @@@
    }

    inline void ReceivePage(ENTRY_TYPE *pageData, unsigned int nEntriesInPage_,
                            CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> cache,
                            int pe, int pageState)
    {
        allocatePage(nEntriesInPage_);
        if(accumInit == 0 && pageState == Accumulate_Fault) {
            ckout << "Zeroing the page" << endl;
            zero();
            accumInit = 1;
        }

        if(pageState == Write_Fault)
            set(pageData, 0, nEntriesInPage_-1); // @@@@, 0 should be got from entryOpsObject
        else
            combine(pageData, 0, nEntriesInPage_-1);

        //ckout << "Page #" << thisIndex << " received " << val << " as " << ((pageState == Write_Fault) ? "Write" : "Accumulate") << endl;

        // send the acknowledgement to the sender that we received 1 RLE page
        //ckout << "Sending AckRLE to PE " << pe << endl;
        cache[pe].AckRLEPage(thisIndex);
    }

    inline unsigned int bytesToIndex(unsigned long int bytes) {
        CkAssert(bytes%sizeof(ENTRY_TYPE) ==0); return bytes/sizeof(ENTRY_TYPE); }
    inline unsigned int bytesToIndexNoCheck(unsigned long int bytes) {
        return bytes/sizeof(ENTRY_TYPE); }

    // @@@
    inline void ReceiveRLEPage(char* buffer, unsigned int size, unsigned int bytesPerPage,
                               CProxy_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS> cache, int pe, int pageState)
    {
//         ckout << "[" << CkMyPe() << "] ReceiveRLEPage "
//               << " size=" << size
//               << " bpp=" << bytesPerPage
//               << " sizeof=" << sizeof(ENTRY_TYPE)
//               << " pe=" << pe
//               << endl;
        allocatePage(bytesToIndex(bytesPerPage)); // @@@

        // decode the buffer into the page using OR'ing as the combining operation. This
        // os OK since if an element will not be written twice
        unsigned int runlength = *((unsigned int*)buffer);
        *((unsigned int*)buffer) = runlength*sizeof(unsigned int);

        if(accumInit == 0 && pageState == Accumulate_Fault)
        {
            ckout << "Zeroing the page" << endl;
            zero();
            accumInit = 1;
        }

//         double val;

        for(unsigned int i = 0; i < runlength; i++)
        {
            unsigned int offset = ((unsigned int*)buffer)[i];
            unsigned int beginByte = *((unsigned int*)(buffer + offset)); offset += sizeof(unsigned int);
            unsigned int endByte = *((unsigned int*)(buffer + offset)); offset += sizeof(unsigned int);

            //if(i == 0 && thisIndex == 0)
            //  ckout << "PageArray[" << thisIndex << "] received " << *((double*)(buffer + offset)) << endl;

            if(pageState == Write_Fault)
            {
//                 val = *((double*)(buffer + offset));
                set((ENTRY_TYPE*)(buffer + offset), bytesToIndex(beginByte), bytesToIndexNoCheck(endByte)); // @@@
            }
            else
            {
//                 val = *((double*)(buffer+offset));
                combine((ENTRY_TYPE*)(buffer + offset), beginByte/sizeof(ENTRY_TYPE), endByte/sizeof(ENTRY_TYPE)); // @@@
            }
        }

        //ckout << "Page #" << thisIndex << " received " << val << " as " << ((pageState == Write_Fault) ? "Write" : "Accumulate") << endl;

        // send the acknowledgement to the sender that we received 1 RLE page
        //ckout << "Sending AckRLE to PE " << pe << endl;
        cache[pe].AckRLEPage(thisIndex);
    }

    // PageArray::
    inline void Sync()
    {
        accumInit = 0;
        contribute(0, NULL, CkReduction::concat);
    }

    inline void emit(int index)
    {
        ckout << "emit: " << epage[index] << endl;
    }

    inline void pup(PUP::er& p)
    {
        p | numEntries;  // @@@
        p | accumInit;
        if(numEntries != 0 && p.isUnpacking())
            allocatePage(numEntries);
        p(epage, numEntries);
    }
};

#define CK_TEMPLATES_ONLY
#include "msa.def.h"
#undef CK_TEMPLATES_ONLY

#endif
