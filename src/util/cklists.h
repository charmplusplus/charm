#ifndef _CKLISTS_H
#define _CKLISTS_H

//#include <converse.h> // for size_t
#include "pup.h"
#include <stdlib.h> // for size_t
#include <string.h> // for memcpy

//"Documentation" class: prevents people from using copy constructors 
class CkNoncopyable {
	//These aren't defined anywhere-- don't use them!
	CkNoncopyable(const CkNoncopyable &c);
	CkNoncopyable &operator=(const CkNoncopyable &c);
public:
	CkNoncopyable(void) {}
};

//Implementation class 
template <class T>
class CkSTLHelper {
protected:
    //Copy nEl elements from src into dest
    void elementCopy(T *dest,const T *src,int nEl) {
      for (int i=0;i<nEl;i++) dest[i]=src[i];
    }
};

// Forward declarations:
template <class T> class CkQ;
template <class T> void pupCkQ(PUP::er &p,CkQ<T> &q);

/// A single-ended FIFO queue.
///  See CkMsgQ if T is a Charm++ message type.
template <class T>
class CkQ : private CkSTLHelper<T>, private CkNoncopyable {
    T *block;
    int blklen;
    int first;
    int len;
    int mask;
    void _expand(void) {
      // blklen is always kept as a power of 2
      int newlen = blklen<<1;
      mask |= blklen;
      if (blklen==0) {
	newlen = 16;
	mask = 0x0f;
      }
      T *newblk = new T[newlen];
      this->elementCopy(newblk,block+first,blklen-first);
      this->elementCopy(newblk+blklen-first,block,first);
      delete[] block; block = newblk;
      blklen = newlen; first = 0;
    }
  public:
    CkQ() : block(NULL), blklen(0), first(0),len(0),mask(0) {}
    CkQ(int sz) :first(0),len(0) {
      int size = 2;
      mask = 0x03;
      while (1<<size < sz) { mask |= 1<<size; size++; }
      blklen = 1<<size;
      block = new T[blklen];
    }
    ~CkQ() { delete[] block; }
    int length(void) { return len; }
    int isEmpty(void) { return (len==0); }
    T deq(void) {
      if(len>0) {
        T &ret = block[first];
	first = (first+1)&mask;
        len--;
      	return ret;
      } else return T(); //For builtin types like int, void*, this is equivalent to T(0)
    }
    void enq(const T &elt) {
      if(len==blklen) _expand();
      block[(first+len)&mask] = elt;
      len++;
    }
    // stack semantics, needed to replace FIFO_QUEUE of converse
    void push(const T &elt) {
      if(len==blklen) _expand();
      first = (first-1+blklen)&mask;
      block[first] = elt;
      len++;
    }
    // stack semantics: look at the element on top of the stack
    T& peek() {
      return block[first];
    }
    // insert an element at pos.
    void insert(int pos, const T &elt) {
      while(len==blklen || pos>=blklen) _expand();
      for (int i=len; i>pos; i--)
        block[(i+first)&mask] = block[(i-1+first)&mask];
      block[(pos+first)&mask] = elt;
      if (pos > len) len = pos+1;
      else len++;
    }
    // delete an element at pos
    T remove(int pos) {
      if (pos >= len) return T(0);
      T ret = block[(pos+first)&mask];
      for (int i=pos; i<len-1; i++)
        block[(i+first)&mask] = block[(i+1+first)&mask];
      len--;
      return ret;
    }
    // delete all elements from pos
    void removeFrom(int pos) {
      CmiAssert (pos < len && pos>=0);
      len = pos;
    }
    //Peek at the n'th item from the queue
    T& operator[](size_t n)
    {
    	n=(n+first)&mask;
    	return block[n];
    }
    // needed to replace FIFO_Enumerate
    T* getArray(void) {
      T *newblk = new T[len];
      int i,j;
      for(i=0,j=first;i<len;i++){
        newblk[i] = block[j];
        j = (j+1)&mask;
      }
      return newblk;
    }
#ifdef _MSC_VER
/* Visual C++ 6.0's operator overloading is buggy,
   so use default operator|, which calls this pup routine. */
     void pup(PUP::er &p) {
        pupCkQ(p,*this);
     }
#endif
};

/// Default pup routine for CkQ: pup each of the elements
template <class T>
inline void pupCkQ(PUP::er &p,CkQ<T> &q) {
    p.syncComment(PUP::sync_begin_array);
    int l=q.length();
    p|l;
    for (int i=0;i<l;i++) {
    	p.syncComment(PUP::sync_item);
    	if (p.isUnpacking()) {
		T t;
		p|t;
		q.enq(t);
	} else {
		p|q[i];
	}
    }
    p.syncComment(PUP::sync_end_array);
}

#ifndef _MSC_VER
/// Default pup routine for CkVec: pup each of the elements
template <class T>
inline void operator|(PUP::er &p,CkQ<T> &q) {pupCkQ(p,q);}
#endif

/// "Flag" class: do not initialize this object.
/// This is sometimes needed for global variables, 
///  when the default zero-initialization is enough.
class CkSkipInitialization {};

// Forward declarations:
template <class T> class CkVec;
template <class T> void pupCkVec(PUP::er &p,CkVec<T> &vec);

/// A typesafe, automatically growing array.
/// Classes used must have a default constructor and working copy constructor.
/// This class is modelled after, but *not* identical to, the
/// (still nonportable) std::vector.
///   The elements of the array are pup'd using plain old "p|elt;".
template <class T>
class CkVec : private CkSTLHelper<T> {
    typedef CkVec<T> this_type;

    T *block; //Elements of vector
    size_t blklen; //Allocated size of block (STL capacity) 
    size_t len; //Number of used elements in block (STL size; <= capacity)
    void makeBlock(int blklen_,int len_) {
       if (blklen_==0) block=NULL; //< saves 1-byte allocations
       else {
         block=new T[blklen_];
         if (block == NULL) blklen_=len_= 0;
       }
       blklen=blklen_; len=len_;
    }
    void freeBlock(void) {
       len=0; blklen=0;
       delete[] block; 
       block=NULL;
    }
    void copyFrom(const this_type &src) {
       makeBlock(src.blklen, src.len);
       this->elementCopy(block,src.block,blklen);
    }
  public:
    CkVec(): block(NULL), blklen(0), len(0) {}
    ~CkVec() { freeBlock(); }
    CkVec(const this_type &src) {copyFrom(src);}
    CkVec(int size) { makeBlock(size,size); } 
    CkVec(int size, int used) { makeBlock(size,used); } 
    CkVec(const CkSkipInitialization &skip) {/* don't initialize */}
    this_type &operator=(const this_type &src) {
      freeBlock();
      copyFrom(src);
      return *this;
    }

    size_t &length(void) { return len; }
    size_t length(void) const {return len;}
    T *getVec(void) { return block; }
    const T *getVec(void) const { return block; }
    
    T& operator[](size_t n) {
      CmiAssert(n<len);
      return block[n]; 
    }
    
    const T& operator[](size_t n) const { 
      CmiAssert(n<len);
      return block[n]; 
    }
    
    /// Reserve at least this much space (changes capacity, size unchanged)
    int reserve(size_t newcapacity) {
      if (newcapacity<=blklen) return 1; /* already there */
      T *oldBlock=block; 
      makeBlock(newcapacity,len);
      if (newcapacity != blklen) return 0;
      this->elementCopy(block,oldBlock,len);
      delete[] oldBlock; //WARNING: leaks if element copy throws exception
      return 1;
    }
    inline size_t capacity(void) const {return blklen;}

    /// Set our length to this value
    int resize(size_t newsize) {
      if (!reserve(newsize)) return 0;
      len=newsize;
      return 1;
    }

    /// Set our length to this value
    void free() {
      freeBlock();
    }

    //Grow to contain at least this position:
    void growAtLeast(size_t pos) {
      if (pos>=blklen) reserve(pos*2+16);
    }
    void insert(size_t pos, const T &elt) {
      if (pos>=len) { 
        growAtLeast(pos);
        len=pos+1;
      }
      block[pos] = elt;
    }
    void remove(size_t pos) {
      if (pos>=len) 
	{
	  CmiAbort("CkVec ERROR: out of bounds\n\n"); 
	  return;
	}
      for (size_t i=pos; i<len-1; i++)
        block[i] = block[i+1];
      len--;
    }
    void removeAll() {
      len = 0;
    }

    void clear()
    {
	freeBlock();
    }

    void insertAtEnd(const T &elt) {insert(length(),elt);}

//STL-compatability:
    void push_back(const T &elt) {insert(length(),elt);}
    size_t size(void) const {return len;}

//verbose position for easier removal
    size_t push_back_v(const T &elt) {insert(length(),elt);return length()-1;}

 
//PUP routine help:
    //Only pup the length of this vector, which is returned:
    int pupbase(PUP::er &p) {
       size_t l=len;
       p(l);
       if (p.isUnpacking()) resize(l); 
       return l;
    }
    
#ifdef _MSC_VER
/* Visual C++ 6.0's operator overloading is buggy,
   so use default operator|, which calls this pup routine. */
     void pup(PUP::er &p) {
        pupCkVec(p,*this);
     }
#endif

		inline void quickSort(){
		  q_sort(0, len - 1,128);
		}

		inline void quickSort(int changeOverSize){
			q_sort(0,len-1,changeOverSize);
		}
		

		inline void q_sort(int left, int right,int changeOverSize){
		  int l_hold, r_hold;
			int pivot;
			
			if(left >= right)
				return;
			
			//swap the center element with the left one
			int mid =  (left+right)/2;
			T temp = block[mid];
			block[mid] = block[left];
			block[left] = temp;
			
		  l_hold = left;
		  r_hold = right;
			pivot = left;
			if(right - left <= changeOverSize){
				bubbleSort(left,right);
				return;
			}
		  while (left < right)
		  {
    		while ((block[right] >= block[pivot]) && (left < right))
		      right--;
		    while ((block[left] <= block[pivot]) && (left < right))
		      left++;
			
				if(left < right){
					T val = block[left];
					block[left] = block[right];
					block[right] = val;
				}
			}
			T val = block[left];
			block[left] = block[pivot];
			block[pivot] = val;
		  pivot = left;
		  left = l_hold;
		  right = r_hold;
		  if (left < pivot)
    		q_sort(left, pivot-1,changeOverSize);
		  if (right > pivot)
    		q_sort(pivot+1, right,changeOverSize);
		}

		inline void bubbleSort(int left,int right){
			for(int i=left;i<=right;i++){
				for(int j=i+1;j<=right;j++){
					if(block[i] >= block[j]){
						T val = block[i];
						block[i] = block[j];
						block[j] = val;
					}
				}
			}
		}

};

/// Default pup routine for CkVec: pup each of the elements
template <class T>
inline void pupCkVec(PUP::er &p,CkVec<T> &vec) {
    size_t len=vec.pupbase(p);
    if (len) PUParray(p,&vec[0],len);
}

#ifndef _MSC_VER
/// Default pup routine for CkVec: pup each of the elements
template <class T>
inline void operator|(PUP::er &p,CkVec<T> &vec) {pupCkVec(p,vec);}
#endif

/* OLD: Deprecated name for vector of basic types. */
#define CkPupBasicVec CkVec

/// Helper for smart pointer classes: allocate a new copy when pup'd.
///  Assumes pointer is non-null
template <class T> 
class CkPupAlwaysAllocatePtr {
public:
	void pup(PUP::er &p,T *&ptr) {
		if (p.isUnpacking()) ptr=new T;
		p|*ptr;
	}
};

/// Helper for smart pointer classes: allocate a new copy when pup'd.
///  Allows pointer to be NULL
template <class T> 
class CkPupAllocatePtr {
public:
	void pup(PUP::er &p,T *&ptr) {
		int isNull=(ptr==0);
		p(isNull);
		if (isNull) ptr=0;
		else {
			if (p.isUnpacking()) ptr=new T;
			p|*ptr;
		}
	}
};

/// Helper for smart pointer classes: copy a PUP::able pointer
template <class T> 
class CkPupAblePtr {
public:
	void pup(PUP::er &p,T *&ptr) {
		p|ptr;
	}
};

/// A not-so-smart smart pointer type: just zero initialized
template <class T, class PUP_PTR=CkPupAllocatePtr<T> > 
class CkZeroPtr {
protected:
	T *storage;
public:
	CkZeroPtr() {storage=0;}
	CkZeroPtr(T *sto) {storage=sto;}
	CkZeroPtr(const CkZeroPtr &src) { storage=src.storage; }
	CkZeroPtr &operator=(const CkZeroPtr &src) {
		storage=src.storage; return *this;
	}
	T *operator=(T *sto) {storage=sto; return sto;}
	
	operator T* () const {return storage;}

	T *release() {
		T *ret=storage; storage=0; return ret;
	}
	
	//Stolen from boost::scoped_ptr:
	T & operator*() const // never throws
		{ return *storage; }
	T * operator->() const // never throws
		{ return storage; }

	
	//Free referenced pointer:
	void destroy(void) {
		delete storage;
		storage=0;
	}
	
        void pup(PUP::er &p) {   
		PUP_PTR ppr;
		ppr.pup(p,storage);
        }
        friend void operator|(PUP::er &p,CkZeroPtr<T,PUP_PTR> &v) {v.pup(p);}
};


///A vector of zero-initialized heap-allocated objects of type T
template <class T, class PUP_PTR=CkPupAllocatePtr<T> >
class CkPupPtrVec : public CkVec< CkZeroPtr<T, PUP_PTR> > {
  public:
  typedef CkPupPtrVec<T,PUP_PTR> this_type;
  typedef CkVec< CkZeroPtr<T, PUP_PTR> > super;
  CkPupPtrVec() {}
  CkPupPtrVec(int size) :super(size) {}
  
  ~CkPupPtrVec() {
    for (size_t i=0;i<this->length();i++)
      this->operator[] (i).destroy();
  }
  void pup(PUP::er &p) { pupCkVec(p,*this); }
  friend void operator|(PUP::er &p,this_type &v) {v.pup(p);}
};

///A vector of pointers-to-subclasses of a PUP::able parent
template <class T>
class CkPupAblePtrVec : public CkVec< CkZeroPtr<T, CkPupAblePtr<T> > > {
 public:
	typedef CkPupAblePtrVec<T> this_type;
	typedef CkVec< CkZeroPtr<T, CkPupAblePtr<T> > > super;
	CkPupAblePtrVec() {}
	CkPupAblePtrVec(int size) :super(size) {}
	CkPupAblePtrVec(const this_type &t) {
		copy_from(t);
	}
	this_type &operator=(const this_type &t) {
		destroy();
		copy_from(t);
		return *this;
	}
	void copy_from(const this_type &t) {
		for (size_t i=0;i<t.length();i++)
			this->push_back((T *)t[i]->clone());
	}
	void destroy(void) {
		for (size_t i=0;i<this->length();i++)
			this->operator[] (i).destroy();
		this->length()=0;
	}
	~CkPupAblePtrVec() {
		destroy();
	}
	void pup(PUP::er &p) { pupCkVec(p,*this); }
	friend void operator|(PUP::er &p,this_type &v) {v.pup(p);}
};

#define MAXMSGS 32

// thread safe pool, also safe with sig io with immediate messages
template <class T>
class SafePool {
  protected:
    int num;
    T msgs[MAXMSGS];
    typedef T (*allocFn)();
    typedef void (*freeFn)(T);
    typedef void (*resetFn)(T);
    allocFn allocfn;
    freeFn  freefn;
    resetFn resetfn;
  public:
    SafePool(allocFn _afn, freeFn _ffn, resetFn _rfn=NULL): allocfn(_afn), freefn(_ffn), resetfn(_rfn) {
      for(int i=0;i<MAXMSGS;i++)
        msgs[i] = allocfn();
      num = MAXMSGS;
    }
    T get(void) {
      /* CkAllocSysMsg() called in .def.h is not thread of sigio safe */
      if (CmiImmIsRunning()) return allocfn();
      return (num ? msgs[--num] : allocfn());
    }
    void put(T m) {
      if (num==MAXMSGS || CmiImmIsRunning())
        freefn(m);
      else {
        if (resetfn!=NULL) resetfn(m);
        msgs[num++] = m;
      }
    }
};

/**
  An array that is broken up into "pages"
  which are separately allocated.  Useful for saving memory
  when storing a long array with few used entries.
  Should be almost as fast as a regular vector.
  
  When nothing is used, this reduces storage space by a factor
  of pageSize*sizeof(T)/sizeof(T*).  For example, an 
  array of 64-bit ints of length 1M takes 8MB; with paging
  it takes just 32KB, plus 2KB for each used block.
  
  The class T must have a pup routine, a default constructor,
  and a copy constructor.
*/
template <class T>
class CkPagedVector : private CkNoncopyable {
	/// This is the number of array entries per page.
	///  For any kind of efficiency, it must be a power of 2,
	///  which lets the compiler turn divides and mods into
	///  bitwise operations.
	enum {pageSize=256u};
	T **pages;
	int nPages;
	void allocatePages(int nEntries) {
		nPages=(nEntries+pageSize-1)/pageSize; // round up
		pages=new T*[nPages];
		for (int pg=0;pg<nPages;pg++) pages[pg]=0;
	}
	void allocatePage(int pg) {
		T *np=new T[pageSize];
		// Initialize the new elements of the page to 0:
		for (int of=0;of<pageSize;of++) np[of]=T();
		pages[pg]=np;
	}
public:
	CkPagedVector() :pages(0), nPages(0) {}
	CkPagedVector(int nEntries) {init(nEntries);}
	~CkPagedVector() {
		for (int pg=0;pg<nPages;pg++) 
			if (pages[pg]) 
				delete[] pages[pg];
		delete[] pages;
	}
	void init(int nEntries) {
		allocatePages(nEntries);
	}
	
	inline T &operator[] (unsigned int i) {
		// This divide and mod should be fast, 
		//  as long as pageSize is a power of 2:
		int pg=i/pageSize; 
		int of=i%pageSize;
		if (pages[pg]==NULL) allocatePage(pg);
		return pages[pg][of];
	}
	
        inline T get (unsigned int i) {
		int pg=i/pageSize; 
		int of=i%pageSize;
                return pages[pg]==NULL? T():pages[pg][of];
        }

	void pupPage(PUP::er &p,int pg) {
		if (p.isUnpacking()) allocatePage(pg);
		for (int of=0;of<pageSize;of++) 
			p|pages[pg][of];
	}
	void pup(PUP::er &p) {
		int pg;
		p|nPages;
		if (!p.isUnpacking()) 
		{ // Packing phase: save each used page
			for (pg=0;pg<nPages;pg++) 
			if (pages[pg]) {
				p|pg;
				pupPage(p,pg);
			}
			pg=-1;
			p|pg;
		} else 
		{ // Unpacking phase: allocate and restore
			allocatePages(nPages*pageSize);
			//cppcheck-suppress uninitvar
			p|pg;
			while (pg!=-1) {
				pupPage(p,pg);
				p|pg;
			}
		}
	}
};

// a special vector, which grows at front, and remove at the end linearly
// the vector keeps an offset, which represents all elements before offset 
// are T(0); lastnull keeps track of the consecutive T(0) (NULL) in the current
// block
template <class T>
class CkCompactVec : private CkSTLHelper<T> {
    typedef CkVec<T> this_type;

    T *block; //Elements of vector
    size_t blklen; //Allocated size of block (STL capacity) 
    size_t len; // total number of used elements in block, including ones before offset
    size_t offset;      // global seqno of the first element in the block
    size_t lastnull;    // the array index of the biggest consecutive NULLs
    void makeBlock(int blklen_,int len_,int offset_=0,int lastnull_=-1) {
       if (blklen_==0) block=NULL; //< saves 1-byte allocations
       else {
         block=new T[blklen_];
         if (block == NULL) blklen_=len_= 0;
       }
       blklen=blklen_; len=len_;
       offset=offset_; lastnull=lastnull_;
    }
    void freeBlock(void) {
       len=0; blklen=0;
       delete[] block; 
       block=NULL;
       offset=0; lastnull=-1;
    }
    void copyFrom(const this_type &src) {
       makeBlock(src.blklen, src.len, src.offset, src.lastnull);
       elementCopy(block,src.block,blklen);
    }
  public:
    CkCompactVec(): block(NULL), blklen(0), len(0), offset(0), lastnull(-1) {}
    ~CkCompactVec() { freeBlock(); }
    CkCompactVec(const this_type &src) {copyFrom(src);}
    CkCompactVec(int size) { makeBlock(size,size); } 
    CkCompactVec(int size, int used) { makeBlock(size,used); } 
    CkCompactVec(const CkSkipInitialization &skip) {/* don't initialize */}
    this_type &operator=(const this_type &src) {
      freeBlock();
      copyFrom(src);
      return *this;
    }

    size_t &length(void) { return len; }
    size_t length(void) const {return len;}
    T *getVec(void) { return block; }
    const T *getVec(void) const { return block; }
    
    T& operator[](size_t n) {
      CmiAssert(n<len && n>=offset);
      return block[n-offset]; 
    }
    
    const T& operator[](size_t n) const { 
      CmiAssert(n-offset<len);
      return n<offset?NULL:block[n-offset]; 
    }
    
    /// Reserve at least this much space (changes capacity, size unchanged)
    int reserve(size_t newcapacity) {
      if (newcapacity-offset<=blklen) return 1; /* already there */
      T *oldBlock=block; 
      makeBlock(newcapacity-offset-lastnull,len,offset,lastnull);
      //if (newcapacity-offset-lastnull != blklen) return 0;
      elementCopy(block,oldBlock+lastnull+1,len-offset-lastnull-1);
      offset+=lastnull+1;   
      lastnull=-1;
      delete[] oldBlock; //WARNING: leaks if element copy throws exception
      return 1;
    }
    inline size_t capacity(void) const {return blklen;}

    /// Set our length to this value
    int resize(size_t newsize) {
      if (!reserve(newsize)) return 0;
      len=newsize;
      return 1;
    }

    /// Set our length to this value
    void free() {
      freeBlock();
    }

    //Grow to contain at least this position:
    void growAtLeast(size_t pos) {
      if (pos>=blklen+offset) reserve(pos*2+16);
    }
    void insert(size_t pos, const T &elt) {
      if (pos>=len) { 
        growAtLeast(pos);
        len=pos+1;
      }
      block[pos-offset] = elt;
    }
    void shrink() {
      for (size_t i=offset+lastnull+1; i<len; i++)
        block[i-offset-lastnull-1] = block[i-offset];
      offset+=lastnull+1; lastnull=-1;
      //printf("shrink: len:%d offset:%d  blklen:%d\n", len, offset, blklen);
    }
    void remove(size_t pos) {
      if (pos < offset) {
        CmiAbort("CkVec ERROR: try to remove non-exisitent element.\n\n");
        return;
      }
      if (pos>=len) 
	{
	  CmiAbort("CkVec ERROR: out of bounds\n\n"); 
	  return;
	}
      if (lastnull >= blklen/2) {   // shrink
        for (size_t i=offset+lastnull+1; i<pos; i++)
          block[i-offset-lastnull-1] = block[i-offset];
        for (size_t i=pos; i<len-1; i++)
          block[i-offset-lastnull-1] = block[i-offset+1];
        offset+=lastnull+1; lastnull=-1;
      }
      else {
      for (size_t i=pos; i<len-1; i++)
        block[i-offset] = block[i-offset+1];
      if (pos-offset==lastnull+1) lastnull++;
      }
      len--;
    }
    void marknull(size_t pos) {
      block[pos-offset] = T(0);
      if (pos == offset+lastnull+1) lastnull++;
      if (lastnull >= blklen/4) shrink();
    }
    void removeAll() {
      len = 0;
      offset=0; lastnull=-1;
    }

    void clear()
    {
	freeBlock();
    }

    void insertAtEnd(const T &elt) {insert(length(),elt);}

//STL-compatability:
    void push_back(const T &elt) {insert(length(),elt);}
    size_t size(void) const {return len;}

//verbose position for easier removal
    size_t push_back_v(const T &elt) {insert(length(),elt);return length()-1;}

 
//PUP routine help:
    //Only pup the length of this vector, which is returned:
    int pupbase(PUP::er &p) {
       size_t l=len;
       p(l);
       if (p.isUnpacking()) resize(l); 
       return l;
    }
    
#ifdef _MSC_VER
/* Visual C++ 6.0's operator overloading is buggy,
   so use default operator|, which calls this pup routine. */
     void pup(PUP::er &p) {
        pupCkVec(p,*this);
        p|offset;
        p|lastnull;
     }
#endif

};


#endif
