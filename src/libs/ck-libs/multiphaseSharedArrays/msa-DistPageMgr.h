// emacs mode line -*- mode: c++; tab-width: 4 ; c-basic-style: stroustrup -*-

#ifndef MSA_DISTPAGEMGR_H
#define MSA_DISTPAGEMGR_H

#include <charm++.h>
#include <string.h>
#include <list>
#include <stack>
#include <map>
#include <set>
#include <vector>
#include "msa-common.h"

// forward decl needed in msa-DistPageMgr.ci, i.e. in msa.decl.h

/// Stores a list of indices to be written out.
struct MSA_WriteSpan_t {
    int start,end;
    inline void pup(PUP::er &p) {
        p|start; p|end;
    }
};

template <class ENTRY, class MERGER,
          unsigned int ENTRIES_PER_PAGE>
class MSA_PageT;
#include "msa.decl.h"

//=======================================================
// Utility Classes

/// Listens for some event on a page or set of pages.
class MSA_Listener {
public:
	MSA_Listener() {}
	virtual ~MSA_Listener();
	/// Getting added to a lister list.
	virtual void add(void) =0;
	/// Event waiting for has occurred.
	virtual void signal(unsigned int pageNo) =0;
};

/// Keeps a list of MSA_Listeners
class MSA_Listeners {
	CkVec<MSA_Listener *> listeners;
public:
	MSA_Listeners();
	~MSA_Listeners();
	
	/// Add this listener to your set.  Calls l->add().
	void add(MSA_Listener *l);
	
	/// Return the number of listeners in our set.
	unsigned int size(void) const {return listeners.size();}
	
	/// Signal all added listeners and remove them from the set.
	void signal(unsigned int pageNo);
};


/** Resumes a thread once all needed pages have arrived */
class MSA_Thread_Listener : public MSA_Listener {
	CthThread thread;    // the suspended thread of execution (0 if not suspended)
	int count;  // number of pages we're still waiting for
public:
	MSA_Thread_Listener() :thread(0), count(0) {}
	
	/// Wait for one more page.
	void add(void);
	
	/// If we're waiting for any pages, suspend our thread.
	void suspend(void);
	
	/// Another page arrived.
	void signal(unsigned int pageNo);
};



/// Fast, fixed-size bitvector class.
template <unsigned int NUM_BITS>
class fixedlength_bitvector {
public:
	/// Data type used to store actual bits in the vector.
	typedef unsigned long store_t;
	enum { store_bits=8*sizeof(store_t) };
	
	/// Number of store_t's in our vector.
	enum { len=(NUM_BITS+(store_bits-1))/store_bits };
	store_t store[len];
	
	fixedlength_bitvector() {reset();}

	/// Fill the entire vector with this value.
	void fill(store_t s) {
		for (int i=0;i<len;i++) store[i]=s;
	}

	void reset(void) {fill(0);}
	
	/// Set-to-1 bit i of the vector.
	void set(unsigned int i) { store[i/store_bits] |= (1lu<<(i%store_bits)); }

	/// Clear-to-0 bit i of the vector.
	void reset(unsigned int i) { store[i/store_bits] &= ~(1lu<<(i%store_bits)); }
	
	/// Return the i'th bit of the vector.
	bool test(unsigned int i) { return (store[i/store_bits] & (1lu<<(i%store_bits))); }
};

/// Stores all housekeeping information about a cached copy of a page: 
///   everything but the actual page data.
template <class ENTRY, unsigned int ENTRIES_PER_PAGE>
class MSA_Page_StateT
{
	/** Write tracking:
		Somehow, we have to identify the entries in a page 
		that have been written to.  Our method for doing this is
		a bitvector: 0's indicate the entry hasn't been written; 
		1's indicate the entry has been written.
	*/
	typedef fixedlength_bitvector<ENTRIES_PER_PAGE> writes_t;
	writes_t writes;
	typedef typename writes_t::store_t writes_store_t;
	enum {writes_bits=writes_t::store_bits};
	
	/// Tracking writes to our writes: a smaller vector, used to 
	///  avoid the large number of 0's in the writes vector.
	///  Bit i of writes2 indicates that store_t i of writes has 1's.
	typedef fixedlength_bitvector<writes_t::len> writes2_t;
	writes2_t writes2;
	typedef typename writes2_t::store_t writes2_store_t;
	enum {writes2_bits=writes2_t::store_bits*writes_t::store_bits};

public:
	/// e.g., Read_Fault for a read-only page.
	MSA_Page_Fault_t state;
	
	/// If true, this page is locked in memory.
	///   Pages get locked so people can safely use the non-checking version of "get".
	bool locked;
	
	/// Threads waiting for this page to be paged in from the network.
	MSA_Listeners readRequests;
	/// Threads waiting for this page to be paged out to the network.
	MSA_Listeners writeRequests;
	
	/// Return true if this page can be safely written back.
	bool canPageOut(void) const {
		return (!locked) && canDelete();
	}
	
	/// Return true if this page can be safely purged from memory.
	bool canDelete(void) const {
		return (readRequests.size()==0) && (writeRequests.size()==0);
	}
	
	MSA_Page_StateT()
		: writes(), writes2(), state(Uninit_State), locked(false),
		  readRequests(), writeRequests()
		{ }

	/// Write entry i of this page.
	void write(unsigned int i) {
		writes.set(i);
		writes2.set(i/writes_t::store_bits);
	}
	
	/// Clear the write list for this page.
	void writeClear(void) {
		for (int i2=0;i2<writes2_t::len;i2++)
			if (writes2.store[i2]) { /* some bits set: clear them all */
				int o=i2*writes2_t::store_bits;
				for (int i=0;i<writes_t::len;i++) 
					writes.store[o+i]=0;
				writes2.store[i2]=0;
			}
	}
	
	/// Return the nearest multiple of m >= v.
	inline int roundUp(int v,int m) {
		return (v+m-1)/m*m;
	}
	
	/// Get a list of our written output values as this list of spans.
	///   Returns the total number of spans written to "span".
	int writeSpans(MSA_WriteSpan_t *span) {
		int nSpans=0;
		
		int cur=0; // entry we're looking at
		while (true) {
			/* skip over unwritten space */
			while (true) { 
				if (writes2.store[cur/writes2_bits]==(writes2_store_t)0) 
					cur=roundUp(cur+1,writes2_bits);
				else if (writes.store[cur/writes_bits]==(writes_store_t)0)
					cur=roundUp(cur+1,writes_bits); 
				else if (writes.test(cur)==false)
					cur++;
				else /* writes.test(cur)==true */
					break;
				if (cur>=ENTRIES_PER_PAGE) return nSpans;
			}
			/* now writes.test(cur)==true */
			span[nSpans].start=cur;
			/* skip over written space */
			while (true) { 
				/* // 0-1 symmetry doesn't hold here, since writes2 may have 1's, but writes may still have some 0's...
				   if (writes2.store[cur/writes2_bits]==~(writes2_store_t)0) 
				   cur=roundUp(cur+1,writes2_bits);
				   else */
				if (writes.store[cur/writes_bits]==~(writes_store_t)0)
					cur=roundUp(cur+1,writes_bits); 
				else if (writes.test(cur)==true)
					cur++;
				else /* writes.test(cur)==false */
					break;
				if (cur>=ENTRIES_PER_PAGE) {
					span[nSpans++].end=ENTRIES_PER_PAGE; /* finish the last span */
					return nSpans;
				}
			}
			/* now writes.test(cur)==false */
			span[nSpans++].end=cur;
		}
	}
};


//=======================================================
// Page-out policy

/**
   class vmPageReplacementPolicy
   Abstract base class providing the interface to the various page
   replacement policies available for use with an MSA
*/
template <class ENTRY_TYPE, unsigned int ENTRIES_PER_PAGE>
class MSA_PageReplacementPolicy
{
public:
	/// Note that a page was just accessed
	virtual void pageAccessed(unsigned int page) = 0;

	/// Ask for the index of a page to discard
	virtual unsigned int selectPage() = 0;
};

/**
   class vmLRUPageReplacementPolicy
   This class provides the functionality of least recently used page replacement policy.
   It needs to be notified when a page is accessed using the pageAccessed() function and
   a page can be selected for replacement using the selectPage() function.
 
   WARNING: a list is absolutely the wrong data structure for this class, 
   because it makes *both* updating as well as searching for a page O(n),
   where n is the number of pages.  A heap would be a better choice,
   as both operations would then become O(lg(n))
*/
template <class ENTRY_TYPE, unsigned int ENTRIES_PER_PAGE>
class vmLRUReplacementPolicy : public MSA_PageReplacementPolicy <ENTRY_TYPE, ENTRIES_PER_PAGE>
{
protected:
    unsigned int nPages;            // number of pages
	const std::vector<ENTRY_TYPE *> &pageTable; // actual data for pages (NULL means page is gone)
	typedef MSA_Page_StateT<ENTRY_TYPE, ENTRIES_PER_PAGE> pageState_t;
	const std::vector<pageState_t *> &pageState;  // state of each page
    std::list<unsigned int> stackOfPages;
    unsigned int lastPageAccessed;

public:
	inline vmLRUReplacementPolicy(unsigned int nPages_, 
								  const std::vector<ENTRY_TYPE *> &pageTable_, 
								  const std::vector<pageState_t *> &pageState_)
		: nPages(nPages_), pageTable(pageTable_), pageState(pageState_), lastPageAccessed(MSA_INVALID_PAGE_NO) {}

    inline void pageAccessed(unsigned int page)
		{
			if(page != lastPageAccessed)
			{
				lastPageAccessed = page;

				// delete this page from the stack and push it at the top
				std::list<unsigned int>::iterator i;
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
			std::list<unsigned int>::iterator i = stackOfPages.begin();
			while(i != stackOfPages.end())
			{
				if(pageTable[*i] == NULL) i = stackOfPages.erase(i);
				else if(!pageState[*i]->canPageOut()) i++;
				else break;
			}

			if(i != stackOfPages.end())
				return *i;
			else
				return MSA_INVALID_PAGE_NO;
		}
};

/**
   class vmNRUPageReplacementPolicy
   This class provides the functionality of not-recently-used page replacement policy.
   It needs to be notified when a page is accessed using the pageAccessed() function and
   a page can be selected for replacement using the selectPage() function.
  
   "not-recently-used" could replace any page that has not been used in the 
   last K accesses; that is, it's a memory-limited version of LRU.
  
   pageAccessed is O(1).
   selectPage best-case is O(K) (if we immediately find a doomed page); 
   worst-case is O(K n) (if there are no doomed pages).
*/
template <class ENTRY_TYPE, unsigned int ENTRIES_PER_PAGE>
class vmNRUReplacementPolicy : public MSA_PageReplacementPolicy <ENTRY_TYPE, ENTRIES_PER_PAGE>
{
protected:
	unsigned int nPages;            // number of pages
	const std::vector<ENTRY_TYPE *> &pageTable; // actual pages (NULL means page is gone)
	typedef MSA_Page_StateT<ENTRY_TYPE, ENTRIES_PER_PAGE> pageState_t;
	const std::vector<pageState_t *> &pageState;  // state of each page
    enum {K=5}; // Number of distinct pages to remember
    unsigned int last[K]; // pages that have been used recently
    unsigned int Klast; // index into last array.
    
    unsigned int victim; // next page to throw out.
    
    bool recentlyUsed(unsigned int page) {
        for (int k=0;k<K;k++) if (page==last[k]) return true;
        return false;
    }

public:
	inline vmNRUReplacementPolicy(unsigned int nPages_, 
								  const std::vector<ENTRY_TYPE *> &pageTable_, 
								  const std::vector<pageState_t *> &pageState_)
		: nPages(nPages_), pageTable(pageTable_), pageState(pageState_), Klast(0), victim(0)
		{
			for (int k=0;k<K;k++) last[k]=MSA_INVALID_PAGE_NO;
		}

    inline void pageAccessed(unsigned int page)
		{
			if (page!=last[Klast]) {
				Klast++; if (Klast>=K) Klast=0;
				last[Klast]=page;
			}
		}

    inline unsigned int selectPage() {
        unsigned int last_victim=victim;
        do {
            victim++; if (victim>=nPages) victim=0;
            if (pageTable[victim]
                &&pageState[victim]->canPageOut()
                &&!recentlyUsed(victim)) {
                /* victim is an allocated, unlocked, non-recently-used page: page him out. */
                return victim;
            }
        } while (victim!=last_victim);
        return MSA_INVALID_PAGE_NO;  /* nobody is pageable */
    }
};

//================================================================

/**
   Holds the typed data for one MSA page.
   Implementation of puppedPage used by the templated code.
*/
template <
	class ENTRY, 
	class MERGER=DefaultEntry<ENTRY>,
	unsigned int ENTRIES_PER_PAGE=MSA_DEFAULT_ENTRIES_PER_PAGE
	>
class MSA_PageT {
    unsigned int n; // number of entries on this page.  Used to send page updates.
	/** The contents of this page: array of ENTRIES_PER_PAGE items */
	ENTRY *data;
	/** Merger object */
	MERGER m;
	bool duplicate;

public:

	MSA_PageT()
		: n(ENTRIES_PER_PAGE), data(new ENTRY[ENTRIES_PER_PAGE]), duplicate(false)
		{
			for (int i=0;i<ENTRIES_PER_PAGE;i++){
				data[i]=m.getIdentity();
			}
		}

    // This constructor is used in PageArray to quickly convert an
    // array of ENTRY into an MSA_PageT.  So we just make a copy of
    // the pointer.  When it comes time to destruct the object, we
    // need to ensure we do NOT delete the data but just discard the
    // pointer.
	MSA_PageT(ENTRY *d):data(d), duplicate(true), n(ENTRIES_PER_PAGE) {
    }
	MSA_PageT(ENTRY *d, unsigned int n_):data(d), duplicate(true), n(n_) {
    }
	virtual ~MSA_PageT() {
 		if (!duplicate) {
            delete [] data;
        }
	}

	virtual void pup(PUP::er &p) {
		p | n;
		/*this pup routine was broken, It didnt consider the case
		  in which n > 0 and data = NULL. This is possible when  
		  sending empty pages. It also doesnt seem to do any allocation
		  for the data variable while unpacking which seems to be wrong
		*/
		bool nulldata = false;
		if(!p.isUnpacking()){
			nulldata = (data == NULL);
		}
		p | nulldata;
		if(nulldata){
			data = NULL;
			return;
		}
		if(p.isUnpacking()){
			data = new ENTRY[n];
		}
		for (int i=0;i<n;i++){
			p|data[i];
		}	
	}

	virtual void merge(MSA_PageT<ENTRY, MERGER, ENTRIES_PER_PAGE> &otherPage) {
		for (int i=0;i<ENTRIES_PER_PAGE;i++)
			m.accumulate(data[i],otherPage.data[i]);
	}

	// These accessors might be used by the templated code.
 	inline ENTRY &operator[](int i) {return data[i];}
 	inline const ENTRY &operator[](int i) const {return data[i];}
    inline ENTRY *getData() { return data; }
};

//=============================== Cache Manager =================================

template <class ENTRY_TYPE, class ENTRY_OPS_CLASS,unsigned int ENTRIES_PER_PAGE>
class MSA_CacheGroup : public CBase_MSA_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>
{
    typedef MSA_PageT<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> page_t;

protected:
    ENTRY_OPS_CLASS *entryOpsObject;
    unsigned int numberOfWorkerThreads;      // number of worker threads across all processors for this shared array
    // @@ migration?
    unsigned int numberLocalWorkerThreads;   // number of worker threads on THIS processor for this shared array
	unsigned int numberLocalWorkerThreadsActive;
    unsigned int enrollDoneq;                 // has enroll() been done on this processor?
    MSA_Listeners enrollWaiters;
    MSA_Listeners syncWaiters;
    std::set<int> enrolledPEs;				// which PEs are involved?

    unsigned int nPages;            ///< number of pages
	std::vector<ENTRY_TYPE*> pageTable;          ///< the page table for this PE: stores actual data.
    typedef MSA_Page_StateT<ENTRY_TYPE,ENTRIES_PER_PAGE> pageState_t;
	std::vector<pageState_t *> pageStateStorage; ///< Housekeeping information for each allocated page.
    
    std::stack<ENTRY_TYPE*> pagePool;     // a pool of unused pages
    
	typedef vmNRUReplacementPolicy<ENTRY_TYPE, ENTRIES_PER_PAGE> vmPageReplacementPolicy;
    MSA_PageReplacementPolicy<ENTRY_TYPE, ENTRIES_PER_PAGE> *replacementPolicy;

    // structure for the bounds of a single write
    typedef struct { unsigned int begin; unsigned int end; } writebounds_t;

    // a list of write bounds associated with a given page
    typedef std::list<writebounds_t> writelist_t;

    writelist_t** writes;           // the write lists for each page

    unsigned int resident_pages;             // pages currently allocated
    unsigned int max_resident_pages;         // max allowable pages to allocate
    unsigned int nEntries;          // number of entries for this array
    unsigned int syncAckCount;      // number of sync ack's we received
    int outOfBufferInPrefetch;      // flag to indicate if the last prefetch ran out of buffers

    int syncThreadCount;            // number of local threads that have issued Sync
    
    
    // used during output
    MSA_WriteSpan_t writeSpans[ENTRIES_PER_PAGE];
    ENTRY_TYPE writeEntries[ENTRIES_PER_PAGE];

    typedef CProxy_MSA_PageArray<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_PageArray_t;
    CProxy_PageArray_t pageArray;     // a proxy to the page array
    typedef CProxy_MSA_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_CacheGroup_t;

    std::map<CthThread, MSA_Thread_Listener *> threadList;

	bool clear;

    /// Return the state for this page, returning NULL if no state available.
    inline pageState_t *stateN(unsigned int pageNo) {
        return pageStateStorage[pageNo];
    }
    
	/// Return the state for this page, allocating if needed.
	pageState_t *state(unsigned int pageNo)
		{
			pageState_t *ret=pageStateStorage[pageNo];
			if (ret==NULL)
			{
				ret=new pageState_t;
				pageStateStorage[pageNo]=ret;
			}
			return ret;
		}

    /// Look up or create the listener for the current thread.
    MSA_Thread_Listener *getListener(void) {
    	CthThread t=CthSelf();
    	MSA_Thread_Listener *l=threadList[t];
		if (l==NULL) {
	    	l=new MSA_Thread_Listener;
	    	threadList[t]=l;
		}
		return l;
    }
    /// Add our thread to this list and suspend
    void addAndSuspend(MSA_Listeners &dest) {
    	MSA_Thread_Listener *l=getListener();
		dest.add(l);
		l->suspend();
    }

    /*********************************************************************************/
    /** these routines deal with managing the page queue **/

    // increment the number of pages the thread is waiting on.
    // also add thread to the page queue; then thread is woken when the page arrives.
    inline void IncrementPagesWaiting(unsigned int page)
		{
			state(page)->readRequests.add(getListener());
		}

    inline void IncrementChangesWaiting(unsigned int page)
		{
			state(page)->writeRequests.add(getListener());
		}

/************************* Page allocation and management **************************/
    
    /// Allocate a new page, removing old pages if we're over the limit.
    /// Returns NULL if no buffer space is available
    inline ENTRY_TYPE* tryBuffer(int async=0) // @@@
		{
			ENTRY_TYPE* nu = NULL;

			// first try the page pool
			if(!pagePool.empty())
			{
				nu = pagePool.top();
				CkAssert(nu != NULL);
				pagePool.pop();
			}

			// else try to allocate the buffer
			if(nu == NULL && resident_pages < max_resident_pages)
			{
				nu = new ENTRY_TYPE[ENTRIES_PER_PAGE];
				resident_pages++;
			}

			// else swap out one of the pages
			if(nu == NULL)
			{
				int pageToSwap = replacementPolicy->selectPage();
				if(pageToSwap != MSA_INVALID_PAGE_NO)
				{
					CkAssert(pageTable[pageToSwap] != NULL);
					CkAssert(state(pageToSwap)->canPageOut() == true);
		
					relocatePage(pageToSwap, async);
					nu = pageTable[pageToSwap];
					pageTable[pageToSwap] = 0;
					delete pageStateStorage[pageToSwap];
					pageStateStorage[pageToSwap]=0;
				}
			}

			// otherwise return NULL

			return nu;
		}
    
    /// Allocate storage for this page, if none has been allocated already.
    ///  Update pageTable, and return the storage for the page.
    inline ENTRY_TYPE* makePage(unsigned int page) // @@@
		{
			ENTRY_TYPE* nu=pageTable[page];
			if (nu==0) {
				nu=tryBuffer();
				if (nu==0) CkAbort("MSA: No available space to create pages.\n");
				pageTable[page]=nu;
			}
			return nu;
		}
    
    /// Throw away this allocated page.
    ///  Returns the page itself, for deletion or recycling.
    ENTRY_TYPE* destroyPage(unsigned int page)
		{
			ENTRY_TYPE* nu=pageTable[page];
			pageTable[page] = 0;
			if (pageStateStorage[page]->canDelete()) {
				delete pageStateStorage[page];
				pageStateStorage[page]=0;
			}
			resident_pages--;
			return nu;
		}
    
    //MSA_CacheGroup::
    void pageFault(unsigned int page, MSA_Page_Fault_t why)
		{
			// Write the page to the page table
			state(page)->state = why;
			if(why == Read_Fault)
			{ // Issue a remote request to fetch the new page
				// If the page has not been requested already, then request it.
				if (stateN(page)->readRequests.size()==0) {
					pageArray[page].GetPage(CkMyPe());
					//ckout << "Requesting page first time"<< endl;
				} else {
					;//ckout << "Requesting page next time.  Skipping request."<< endl;
				}
				MSA_Thread_Listener *l=getListener();
				stateN(page)->readRequests.add(l);
				l->suspend(); // Suspend until page arrives.
			}
			else {
				// Build an empty buffer into which to create the new page
				ENTRY_TYPE* nu = makePage(page);
				writeIdentity(nu);
			}
		}
    
    /// Make sure this page is accessible, faulting the page in if needed.
    // MSA_CacheGroup::
    inline void accessPage(unsigned int page,MSA_Page_Fault_t access)
		{
			if (pageTable[page] == 0) {
//             ckout << "p" << CkMyPe() << ": Calling pageFault" << endl;
				pageFault(page, access);
			}
#if CMK_ERROR_CHECKING
			if (stateN(page)->state!=access) {
				CkPrintf("page=%d mode=%d pagestate=%d", page, access, stateN(page)->state);
				CkAbort("MSA Runtime error: Attempting to access a page that is still in another mode.");
			}
#endif
			replacementPolicy->pageAccessed(page);
		}

    // MSA_CacheGroup::
    // Fill this page with identity values, to prepare for writes or 
    //  accumulates.
    void writeIdentity(ENTRY_TYPE* pagePtr)
		{
			for(unsigned int i = 0; i < ENTRIES_PER_PAGE; i++)
				pagePtr[i] = entryOpsObject->getIdentity();
		}
    
/************* Page Flush and Writeback *********************/
    bool shouldWriteback(unsigned int page) {
        if (!pageTable[page]) return false;
		return (stateN(page)->state == Write_Fault || stateN(page)->state == Accumulate_Fault);
    }
    
    inline void relocatePage(unsigned int page, int async)
		{
			//CkAssert(pageTable[page]);
			if(shouldWriteback(page))
			{
				// the page to be swapped is a writeable page. So inform any
				// changes this node has made to the page manager
				sendChangesToPageArray(page, async);
			}
		}

    inline void sendChangesToPageArray(const unsigned int page, const int async)
		{
			sendRLEChangesToPageArray(page);
	
			MSA_Thread_Listener *l=getListener();
			state(page)->writeRequests.add(l);
			if (!async)
				l->suspend(); // Suspend until page is really gone.
			// TODO: Are write acknowledgements really necessary ?
		}

    // Send the page data as a contiguous block.
    //   Note that this is INCORRECT when writes to pages overlap!
    inline void sendNonRLEChangesToPageArray(const unsigned int page) // @@@
		{
			pageArray[page].PAReceivePage(pageTable[page], ENTRIES_PER_PAGE, CkMyPe(), stateN(page)->state);
		}
    
    // Send the page data as an RLE block.
    // this function assumes that there are no overlapping writes in the list
    inline void sendRLEChangesToPageArray(const unsigned int page)
		{
			ENTRY_TYPE *writePage=pageTable[page];
			int nSpans=stateN(page)->writeSpans(writeSpans);
			if (nSpans==1) 
			{ /* common case: can make very fast */
				int nEntries=writeSpans[0].end-writeSpans[0].start;
				if (entryOpsObject->pupEveryElement()) {
					pageArray[page].PAReceiveRLEPageWithPup(writeSpans,nSpans,
															page_t(&writePage[writeSpans[0].start],nEntries),nEntries,
															CkMyPe(),stateN(page)->state);
				} else {
					pageArray[page].PAReceiveRLEPage(writeSpans,nSpans,
													 &writePage[writeSpans[0].start], nEntries,
													 CkMyPe(),stateN(page)->state);
				}
			} 
			else /* nSpans>1 */ 
			{ /* must copy separate spans into a single output buffer (luckily rare) */
				int nEntries=0;
				for (int s=0;s<nSpans;s++) {
					for (int i=writeSpans[s].start;i<writeSpans[s].end;i++)
						writeEntries[nEntries++]=writePage[i]; // calls assign
				}
				if (entryOpsObject->pupEveryElement()) {
					pageArray[page].PAReceiveRLEPageWithPup(writeSpans,nSpans,
															page_t(writeEntries,nEntries),nEntries,
															CkMyPe(),stateN(page)->state);
				} else {
					pageArray[page].PAReceiveRLEPage(writeSpans,nSpans,
													 writeEntries,nEntries,
													 CkMyPe(),stateN(page)->state);
				}
			}
		}

/*********************** Public Interface **********************/
public:
    // 
    //
    // MSA_CacheGroup::
	inline MSA_CacheGroup(unsigned int nPages_, CkArrayID pageArrayID,
						  unsigned int max_bytes_, unsigned int nEntries_, 
						  unsigned int numberOfWorkerThreads_)
		: numberOfWorkerThreads(numberOfWorkerThreads_),
		  nPages(nPages_),
		  nEntries(nEntries_), 
		  pageTable(nPages, NULL),
		  pageStateStorage(nPages, NULL),
		  pageArray(pageArrayID),
		  max_resident_pages(max_bytes_/(sizeof(ENTRY_TYPE)*ENTRIES_PER_PAGE)),
		  entryOpsObject(new ENTRY_OPS_CLASS),
		  replacementPolicy(new vmPageReplacementPolicy(nPages, pageTable, pageStateStorage)),
		  outOfBufferInPrefetch(0), syncAckCount(0),syncThreadCount(0),
		  resident_pages(0), numberLocalWorkerThreads(0), 
		  numberLocalWorkerThreadsActive(0), enrollDoneq(0),
		  clear(false)
		{
			MSADEBPRINT(printf("MSA_CacheGroup nEntries %d \n",nEntries););
		}

    // MSA_CacheGroup::
    inline ~MSA_CacheGroup()
		{
			FreeMem();
		}

    /* To change the accumulate function TBD @@ race conditions */
    inline void changeEntryOpsObject(ENTRY_OPS_CLASS *e) {
        entryOpsObject = e;
        pageArray.changeEntryOpsObject(e);
    }

    // MSA_CacheGroup::
    inline const ENTRY_TYPE* readablePage(unsigned int page)
		{
			accessPage(page,Read_Fault);
	
			return pageTable[page];
		}

    // MSA_CacheGroup::
    //
    // known local page
    inline const void* readablePage2(unsigned int page)
		{
			return pageTable[page];
		}

    // MSA_CacheGroup::
    // Obtains a writable copy of the page.
    inline ENTRY_TYPE* writeablePage(unsigned int page, unsigned int offset)
		{
			accessPage(page,Write_Fault);

			// NOTE: Since we assume write once semantics, i.e. between two calls to sync,
			// either there can be no write to a location or a single write to a location,
			// a readable page will suffice as a writeable page too, because no one else
			// is going to write to this location. In reality, two locations on the *same*
			// page can be written by two different threads, in which case we will need
			// to keep track of which parts of the page have been written, hence:
			stateN(page)->write(offset);
//     ckout << "write:" << page*ENTRIES_PER_PAGE+offset << endl;
	
			return pageTable[page];
		}

    // MSA_CacheGroup::
    inline ENTRY_TYPE &accumulate(unsigned int page, unsigned int offset)
		{
			accessPage(page,Accumulate_Fault);
			stateN(page)->write(offset);
			return pageTable[page][offset];
		}

    /// A requested page has arrived from the network.
    ///  nEntriesInPage_ = num entries being sent (0 for empty page, num entries otherwise)
    inline void ReceivePageWithPUP(unsigned int page, page_t &pageData, int size)
		{
			ReceivePage(page, pageData.getData(), size);
		}

    inline void ReceivePage(unsigned int page, ENTRY_TYPE* pageData, int size)
		{
			CkAssert(0==size || ENTRIES_PER_PAGE == size);
			// the page we requested has been received
			ENTRY_TYPE *nu=makePage(page);
			if(size!=0)
			{
				for(unsigned int i = 0; i < size; i++)
					nu[i] = pageData[i]; // @@@, calls assignment operator
			}
			else /* isEmpty */
			{
				// the page we requested for is empty, so we can just initialize it.
				writeIdentity(nu);
			}
	
			state(page)->readRequests.signal(page);
		}

    // This EP is invoked during sync to acknowledge that a dirty page
    // has been received and written back to the page owner.  We keep track
    // of the number of ack's yet to arrive in nChangesWaiting.  Once
    // all the dirty pages have been ack'd, we awaken the thread that
    // flushed the page.
    //
    // It's not clear this is useful very often...
    //
    // MSA_CacheGroup::
    inline void AckPage(unsigned int page)
		{
			state(page)->writeRequests.signal(page);
		}

    // MSA_CacheGroup::
    // synchronize all the pages and also clear up the cache
    inline void SyncReq(int single, bool clear_)
		{
			clear = clear || clear_;
			MSADEBPRINT(printf("SyncReq single %d\n",single););
			if(single)
			{
				/*ask all the caches to send their updates to the page
				 * array, but we don't need to empty the caches on the
				 * other PEs*/
				SingleSync();
				EmptyCache();

				getListener()->suspend();
			}
			else{
				Sync(clear_);
			}
		}

    // MSA_CacheGroup::
    inline void FlushCache()
		{
			// flush the local cache
			// for each writeable page, send that page to the array element
			for(unsigned int i = 0; i < nPages; i++)
			{
				if(shouldWriteback(i)) {
					//ckout << "p" << CkMyPe() << "FlushCache: sending page " << i << endl;
					sendChangesToPageArray(i, 1);
				}
			}
		}

    // MSA_CacheGroup::
    void EmptyCache()
		{
			/* just makes all the pages empty, assuming that the data
			 * in those pages has been flushed to the owners */
			for(unsigned int i = 0; i < nPages; i++)
			{
				if(pageTable[i]) pagePool.push(destroyPage(i));
			}
		}

/************************ Enroll ********************/
    /// Enroll phase 1: called by users.
    // MSA_CacheGroup::
    inline void enroll(unsigned int num_workers)
		{
			CkAssert(num_workers == numberOfWorkerThreads); // just to verify
			CkAssert(enrollDoneq == 0);
			numberLocalWorkerThreads++;
			numberLocalWorkerThreadsActive++;
			// @@ how to ensure that enroll is called only once?

			//ckout << "[" << CkMyPe() << "] sending sync ack to PE 0" << endl;
			this->thisProxy[0].enrollAck(CkMyPe());
			//ckout << "[" << CkMyPe() << "] suspening thread in Sync() " << endl;
			addAndSuspend(enrollWaiters);
			//ckout << "[" << CkMyPe() << "] rsuming thread in Sync()" << endl;

			CkAssert(enrollDoneq == 1);
			return;
		}

	// The check in the above function isn't all that useful, and
	// breaks modularity for client code. We don't want to break
	// backward compatibility, so just obviate that one assertion.
	void enroll()
		{
			enroll(numberOfWorkerThreads);
		}

    /// Enroll phase 2: called on PE 0 from everywhere
    inline void enrollAck(int originator)
		{
			CkAssert(CkMyPe() == 0);  // enrollAck is only called on PE 0
			CkAssert(enrollDoneq == 0);  // prevent multiple enroll operations
        
			syncAckCount++;
			enrolledPEs.insert(originator);
			//ckout << "[" << CkMyPe() << "] SyncAckcount = " << syncAckCount << endl;
			if(syncAckCount == numberOfWorkerThreads) {
//             ckout << "[" << CkMyPe() << "]" << "Enroll operation is almost done" << endl;
				syncAckCount = 0;
				enrollDoneq = 1;
				// What if fewer worker threads than pe's ?  Handled in
				// enrollDone.
				this->thisProxy.enrollDone();
			}
		}

    /// Enroll phase 3: called everywhere by PE 0
    inline void enrollDone()
		{
//         ckout << "[" << CkMyPe() << "] enrollDone.  Waking threads."
//               <<  " numberOfWorkerThreads=" << numberOfWorkerThreads
//               <<  " local=" << numberLocalWorkerThreads << endl;
			enrollDoneq = 1;
			enrollWaiters.signal(0);
		}

/******************************** Sync & writeback ***********************/
    // MSA_CacheGroup::
    inline void SingleSync()
		{
			/* a single thread issued a sync call with all = 1. The
			 * first thing to do is to flush the local cache */
			FlushCache();
		}

	void SyncRelease()
		{
			numberLocalWorkerThreadsActive--;

			syncDebug();
			
			if(syncThreadCount < numberLocalWorkerThreadsActive)
			{
				return;
			}

			this->thisProxy[CkMyPe()].FinishSync();
		}

	void syncDebug()
		{
			MSADEBPRINT(printf("Sync  (Total threads: %d, Active: %d, Synced: %d)\n", 
							   numberLocalWorkerThreads, numberLocalWorkerThreadsActive, syncThreadCount));
		}

	void activate()
		{
			numberLocalWorkerThreadsActive++;
			
			CkAssert(numberLocalWorkerThreadsActive <= numberLocalWorkerThreads);
		}

	void FinishSync()
		{
			//ckout << "[" << CkMyPe() << "] Sync started" << endl;

			// flush the cache asynchronously and also empty it
			FlushCache();
			// idea: instead of invalidating the pages, switch it to read
			// mode. That will not work, since the page may have also been
			// modified by another thread.
			EmptyCache();

			// Now, we suspend too (if we had at least one dirty page).
			// We will be awoken when all our dirty pages have been
			// written and acknowledged.
			MSADEBPRINT(printf("Sync calling suspend on getListener\n"););
			getListener()->suspend();
			MSADEBPRINT(printf("Sync awakening after suspend\n"););

			// So far, the sync has been asynchronous, i.e. PE0 might be ahead
			// of PE1.  Next we basically do a barrier to ensure that all PE's
			// are synchronized.

			// at this point, the sync's across the group should
			// synchronize among themselves by each one sending
			// a sync acknowledge message to PE 0. (this is like
			// a reduction over a group)
			if(CkMyPe() != 0)
			{
				this->thisProxy[0].SyncAck(clear);
			}
			else /* I *am* PE 0 */
			{
				SyncAck(clear);
			}
			MSADEBPRINT(printf("Sync all local threads done, going to addAndSuspend\n"););
			/* Wait until sync is reflected from PE 0 */
			addAndSuspend(syncWaiters);
				
			MSADEBPRINT(printf("Sync all local threads done waking up after addAndSuspend\n"););
			//ckout << "[" << CkMyPe() << "] Sync finished" << endl;			
		}

    // MSA_CacheGroup::
    inline void Sync(bool clear_)
		{
			syncThreadCount++;
			//ckout << "[" << CkMyPe() << "] syncThreadCount = " << syncThreadCount << " " << numberLocalWorkerThreads << endl;
			//ckout << "[" << CkMyPe() << "] syncThreadCount = " << syncThreadCount << ", registered threads = " << getNumRegisteredThreads()
			//  << ", number of suspended threads = " << getNumSuspendedThreads() << endl;

			syncDebug();

			clear |= clear_;

			// First, all threads on this processor need to reach the sync
			// call; only then can we proceed with merging the data.  Only
			// the last thread on this processor needs to do the FlushCache,
			// etc.  Others just suspend until the sync is over.
			MSADEBPRINT(printf("Sync  (Total threads: %d, Active: %d, Synced: %d)\n", 
							   numberLocalWorkerThreads, numberLocalWorkerThreadsActive, syncThreadCount));
			if(syncThreadCount < numberLocalWorkerThreadsActive)
			{
				MSADEBPRINT(printf("Sync addAndSuspend\n"));
				addAndSuspend(syncWaiters);
				return;
			}

			FinishSync();
		}

    inline unsigned int getNumEntries() { return nEntries; }
    inline CProxy_PageArray_t getArray() { return pageArray; }

    // TODO: Can this SyncAck and other simple Acks be made efficient?
// Yes - Replace calls to this with contributes to a reduction that calls pageArray.Sync()
    inline void SyncAck(bool clear_)
		{
			CkAssert(CkMyPe() == 0);  // SyncAck is only called on PE 0
			syncAckCount++;
			clear = clear || clear_;
			// DONE @@ what if fewer worker threads than pe's ?
			// @@ what if fewer worker threads than pe's and >1 threads on 1 pe?
			//if(syncAckCount == min(numberOfWorkerThreads, CkNumPes())){
			if (syncAckCount == enrolledPEs.size()) {
				MSADEBPRINT(printf("SyncAck starting reduction on pageArray of size %d number of pages %d\n",
								   nEntries, nPages););
				pageArray.Sync(clear);
			}
		}

    inline void SyncDone(CkReductionMsg *m)
		{
			delete m;
			//ckout << "[" << CkMyPe() << "] Sync Done indication" << endl;
			//ckout << "[" << CkMyPe() << "] Sync Done indication" << endl;
			/* Reset for next sync */
			syncThreadCount = 0;
			syncAckCount = 0;
			clear = false;
			MSADEBPRINT(printf("SyncDone syncWaiters signal to be called\n"););
			syncWaiters.signal(0);
		}

    inline void FreeMem()
		{
			for(unsigned int i = 0; i < nPages; i++)
			{
				if(pageTable[i]) delete [] destroyPage(i);
			}

			while(!pagePool.empty())
			{
				delete [] pagePool.top();  // @@@
				pagePool.pop();
			}
	
			resident_pages=0;
		}

	/** 
		Deregister a client. Decrement the number of local threads. If total number of local threads 
		hits 0 FreeMem()
	*/
	inline void unroll() {
		numberLocalWorkerThreads--;
		if(numberLocalWorkerThreads == 0){
			FreeMem();
		}
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
						ENTRY_TYPE* nu = tryBuffer(1);
						if(NULL == nu)
						{
							/* signal that sufficient buffer space is not available */
							outOfBufferInPrefetch = 1;
							break;
						}

						pageTable[p] = nu;
						state(p)->state = Read_Fault;

						pageArray[p].GetPage(CkMyPe());
						IncrementPagesWaiting(p);
						//ckout << "Prefetch page" << p << ", pages waiting = " << nPagesWaiting << endl;
						/* don't suspend the thread */
					}

					/* mark the page as being locked */
					state(p)->locked = true;
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
				getListener()->suspend();
				UnlockPages();
				return 1;
			}
			else
			{
				// prefetch requests have been successfully issued already, so suspend the
				// thread and wait for completion
				outOfBufferInPrefetch = 0;
				getListener()->suspend();
				return 0;
			}
		}
    
    inline void UnlockPage(unsigned int page) {
        pageState_t *s=stateN(page);
		if(s && s->locked) {
            replacementPolicy->pageAccessed(page);
            s->locked = false;
		}
    }

    /**
     * Unlock all the pages locked in the cache
     */
    inline void UnlockPages()
		{
			// add all the locked pages to page replacement policy
			for(unsigned int page = 0; page < nPages; page++)
				UnlockPage(page);
		}

    /**
     * Unlock the given pages: [startPage ... endPage]
     *  Note that the range is inclusive.
     */
    inline void UnlockPages(unsigned int startPage, unsigned int endPage)
		{
			for(unsigned int page = startPage; page <= endPage; page++)
				UnlockPage(page);
		}

    /// Debugging routine
    inline void emitBufferValue(int ID, unsigned int pageNum, unsigned int offset)
		{
			CkAssert( pageNum < nPages );
			CkAssert( offset < ENTRIES_PER_PAGE );

			//ckout << "p" << CkMyPe() << "ID" << ID;
//         if (pageTable[pageNum] == 0)
//             ckout << "emitBufferValue: page " << pageNum << " not available in local cache." << endl;
//         else
//             ckout << "emitBufferValue: [" << pageNum << "," << offset << "] = " << pageTable[pageNum][offset] << endl;
		}
};

// each element of this array is responsible for managing
// the information about a single page. It is in effect the
// "owner" as well as the "manager" for that page.
//
template<class ENTRY_TYPE, class ENTRY_OPS_CLASS,unsigned int ENTRIES_PER_PAGE> 
class MSA_PageArray : public CBase_MSA_PageArray<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>
{
    typedef CProxy_MSA_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> CProxy_CacheGroup_t;
    typedef MSA_PageT<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE> page_t;
    
protected:
    ENTRY_TYPE *epage;
    ENTRY_OPS_CLASS entryOpsObject;
    CProxy_CacheGroup_t cache;

    unsigned int pageNo() { return this->thisIndex; }

    inline void allocatePage(MSA_Page_Fault_t access) // @@@
		{
			if(epage == NULL)
			{
				epage = new ENTRY_TYPE[ENTRIES_PER_PAGE];
				writeIdentity();
			}
		}

    // begin and end are indexes into the page.
    inline void set(const ENTRY_TYPE* buffer, unsigned int begin, unsigned int end)
		{
			//ckout << "set: " << begin << "," << end << endl;
			for(unsigned int i = 0; i < (end - begin); i++) {
				epage[begin + i] = buffer[i]; // @@@, calls assignment operator
				//ckout << "set val[" << begin+i << "]=" << buffer[i] << endl;
			}
		}

    // MSA_PageArray::
    inline void combine(const ENTRY_TYPE* buffer, unsigned int begin, unsigned int end)
		{
			ENTRY_TYPE* pagePtr = epage + begin;
			for(unsigned int i = 0; i < (end - begin); i++)
				entryOpsObject.accumulate(pagePtr[i], buffer[i]);
		}

    // MSA_PageArray::
    inline void writeIdentity()
		{
			for(unsigned int i = 0; i < ENTRIES_PER_PAGE; i++)
				epage[i] = entryOpsObject.getIdentity();
		}

public:
    inline MSA_PageArray() : epage(NULL) { }
    inline MSA_PageArray(CkMigrateMessage* m) { delete m; }
    
    void setCacheProxy(CProxy_CacheGroup_t &cache_)
		{
			cache=cache_;
		}
    
    void pup(PUP::er& p)
		{
			int epage_present=(epage!=0);
			p|epage_present;
			if (epage_present) {
				if(p.isUnpacking())
					allocatePage(Write_Fault);
				for (int i=0;i<ENTRIES_PER_PAGE;i++)
					p|epage[i];
			}
		}
    
    inline ~MSA_PageArray()
		{
			if(epage) delete [] epage;
		}

    /// Request our page.
    ///   pe = to which to send page
    inline void GetPage(int pe)
		{
			if(epage == NULL) {
				// send empty page
				if (entryOpsObject.pupEveryElement())
					cache[pe].ReceivePageWithPUP(pageNo(), page_t((ENTRY_TYPE*)NULL), 0);
				else
					cache[pe].ReceivePage(pageNo(), (ENTRY_TYPE*)NULL, 0);
			} else {
				// send page with data
				if (entryOpsObject.pupEveryElement())
					cache[pe].ReceivePageWithPUP(pageNo(), page_t(epage), ENTRIES_PER_PAGE);
				else
					cache[pe].ReceivePage(pageNo(), epage, ENTRIES_PER_PAGE);  // send page with data                
			}
		}

    /// Receive a non-runlength encoded page from the network:
    // @@ TBD: ERROR: This does not work for  varsize pages.
    inline void PAReceivePage(ENTRY_TYPE *pageData,
							  int pe, MSA_Page_Fault_t pageState)
		{
			allocatePage(pageState);

			if(pageState == Write_Fault)
				set(pageData, 0, ENTRIES_PER_PAGE);
			else
				combine(pageData, 0, ENTRIES_PER_PAGE);
	
			// send the acknowledgement to the sender that we received the page
			//ckout << "Sending Ack to PE " << pe << endl;
			cache[pe].AckPage(this->thisIndex);
		}

    /// Receive a runlength encoded page from the network:
    inline void PAReceiveRLEPageWithPup(
    	const MSA_WriteSpan_t *spans, unsigned int nSpans, 
        page_t &entries, unsigned int nEntries, 
        int pe, MSA_Page_Fault_t pageState)
		{
			PAReceiveRLEPage(spans, nSpans, entries.getData(), nEntries, pe, pageState);
		}


    inline void PAReceiveRLEPage(
    	const MSA_WriteSpan_t *spans, unsigned int nSpans, 
        const ENTRY_TYPE *entries, unsigned int nEntries, 
        int pe, MSA_Page_Fault_t pageState)
		{
			allocatePage(pageState);
	
			//ckout << "p" << CkMyPe() << "ReceiveRLEPage nSpans=" << nSpans << " nEntries=" << nEntries << endl;
			int e=0; /* consumed entries */
			for (int s=0;s<nSpans;s++) {
				if(pageState == Write_Fault)
					set(&entries[e], spans[s].start,spans[s].end);
				else /* Accumulate_Fault */
					combine(&entries[e], spans[s].start,spans[s].end);
				e+=spans[s].end-spans[s].start;
			} 

			// send the acknowledgement to the sender that we received the page
			//ckout << "Sending AckRLE to PE " << pe << endl;
			cache[pe].AckPage(this->thisIndex);
		}

    // MSA_PageArray::
    inline void Sync(bool clear)
		{
			if (clear && epage)
				writeIdentity();
			MSADEBPRINT(printf("MSA_PageArray::Sync about to call contribute \n"););
			CkCallback cb(CkIndex_MSA_CacheGroup<ENTRY_TYPE, ENTRY_OPS_CLASS, ENTRIES_PER_PAGE>::SyncDone(NULL), cache);
			this->contribute(0, NULL, CkReduction::concat, cb);
		}

    inline void emit(int ID, int index)
		{
			//ckout << "p" << CkMyPe() << "ID" << ID;
//         if(epage == NULL)
//             ckout << "emit: epage is NULL" << endl;
//         else
//             ckout << "emit: " << epage[index] << endl;
		}
};

#define CK_TEMPLATES_ONLY
#include "msa.def.h"
#undef CK_TEMPLATES_ONLY

#endif
