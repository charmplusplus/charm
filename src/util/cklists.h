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

/// A single-ended FIFO queue.
///  See CkMsgQ if T is a Charm++ message type.
template <class T>
class CkQ : private CkSTLHelper<T>, private CkNoncopyable {
    T *block;
    int blklen;
    int first;
    int len;
    void _expand(void) {
      int newlen=16+len*2;
      T *newblk = new T[newlen];
      elementCopy(newblk,block+first,blklen-first);
      elementCopy(newblk+blklen-first,block,first);
      delete[] block; block = newblk;
      blklen = newlen; first = 0;
    }
  public:
    CkQ() :first(0),len(0) {
      block = NULL; blklen=0;
    }
    CkQ(int sz) :first(0),len(0) {
      block = new T[blklen=sz];
    }
    ~CkQ() { delete[] block; }
    int length(void) { return len; }
    int isEmpty(void) { return (len==0); }
    T deq(void) {
      if(len>0) {
        T &ret = block[first];
        first = (first+1)%blklen;
        len--;
      	return ret;
      } else return T(); //For builtin types like int, void*, this is equivalent to T(0)
    }
    void enq(const T &elt) {
      if(len==blklen) _expand();
      block[(first+len)%blklen] = elt;
      len++;
    }
    // stack semantics, needed to replace FIFO_QUEUE of converse
    void push(const T &elt) {
      if(len==blklen) _expand();
      first = (first-1+blklen)%blklen;
      block[first] = elt;
      len++;
    }
    // insert an element at pos.
    void insert(int pos, const T &elt) {
      while(len==blklen || pos>=blklen) _expand();
      for (int i=len; i>pos; i--)
        block[(i+first)%blklen] = block[(i-1+first)%blklen];
      block[(pos+first)%blklen] = elt;
      if (pos > len) len = pos+1;
      else len++;
    }
    // delete an element at pos
    T remove(int pos) {
      if (pos >= len) return T(0);
      T ret = block[(pos+first)%blklen];
      for (int i=pos; i<len-1; i++)
        block[(i+first)%blklen] = block[(i+1+first)%blklen];
      len--;
      return ret;
    }
    //Peek at the n'th item from the queue
    T& operator[](size_t n) 
    {
    	n=(n+first)%blklen;
    	return block[n];
    }
    // needed to replace FIFO_Enumerate
    T* getArray(void) {
      T *newblk = new T[len];
      int i,j;
      for(i=0,j=first;i<len;i++){
        newblk[i] = block[j];
        j = (j+1)%blklen;
      }
      return newblk;
    }
};


/// A typesafe, automatically growing array.
/// Classes used must have a default constructor and working copy constructor.
/// This class is modelled after, but *not* identical to, the 
/// (still nonportable) std::vector. 
///   The elements of the array are pup'd using plain old "p|elt;".
template <class T>
class CkVec : private CkSTLHelper<T> {
    typedef CkVec<T> this_type;
    
    T *block; //Elements of vector
    int blklen; //Allocated size of block 
    int len; //Number of used elements in block
    void makeBlock(int blklen_,int len_) {
       block=new T[blklen_];
       blklen=blklen_; len=len_;
    }
    void freeBlock(void) {
       len=0; blklen=0;
       delete[] block; block=NULL;
    }
    void copyFrom(const this_type &src) {
       makeBlock(src.blklen, src.len);
       elementCopy(block,src.block,blklen);
    }
  public:
    CkVec() {block=NULL;blklen=len=0;}
    ~CkVec() { freeBlock(); }
    CkVec(const this_type &src) {copyFrom(src);}
    CkVec(int size) { makeBlock(size,size); } 
    this_type &operator=(const this_type &src) {
      freeBlock();
      copyFrom(src);
      return *this;
    }

    int &length(void) { return len; }
    int length(void) const {return len;}
    T *getVec(void) { return block; }
    const T *getVec(void) const { return block; }
    
    T& operator[](size_t n) { return block[n]; }
    const T& operator[](size_t n) const { return block[n]; }
    
    void setSize(int blklen_) {
      T *oldBlock=block; 
      makeBlock(blklen_,len);
      elementCopy(block,oldBlock,len);
      delete[] oldBlock; //WARNING: leaks if element copy throws exception
    }
    //Grow to contain at least this position:
    void growAtLeast(int pos) {
      if (pos>=blklen) setSize(pos*2+16);
    }
    void insert(int pos, const T &elt) {
      if (pos>=len) { 
        growAtLeast(pos);
        len=pos+1;
      }
      block[pos] = elt;
    }
    void insertAtEnd(const T &elt) {insert(length(),elt);}

//STL-compatability:
    void push_back(const T &elt) {insert(length(),elt);}
    int size(void) const {return len;}
 
//PUP routine:
  protected:
    //Only pup the length of this vector, which is returned:
    int pupbase(PUP::er &p) {
       int l=len;
       p(l);
       if (p.isUnpacking()) { setSize(l); len=l;}
       return l;
    }
  public:
    void pup(PUP::er &p) {
       int l=pupbase(p);
       for (int i=0;i<l;i++) p|block[i];
    }
    friend void operator|(PUP::er &p,this_type &v) {v.pup(p);}
};


///A vector of basic types, which can be pupped as an array
/// (more restricted, but more efficient version of CkVec)
template <class T>
class CkPupBasicVec : public CkVec<T> {
public:
	CkPupBasicVec() {}
	CkPupBasicVec(int size) :CkVec<T>(size) {}
	
        void pup(PUP::er &p) {   
                int l=pupbase(p);
                p(getVec(),l);
        }
        friend void operator|(PUP::er &p,CkPupBasicVec<T> &v) {v.pup(p);}
};



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
	//Default copy constructor, asssignment operator
	T *operator=(T *sto) {storage=sto; return sto;}
	
	operator T* () const {return storage;}

	bool operator==(T *t) const {return storage==t;}
	bool operator!=(T *t) const {return storage!=t;}
	
	//Stolen from boost::scoped_ptr:
	T & operator*() const // never throws
		{ return *storage; }
	T * operator->() const // never throws
		{ return storage; }

	// implicit conversion to "bool"
	operator bool () const // never throws
	{
		return storage == 0;
	}
	
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
class CkPupPtrVec : public CkVec< CkZeroPtr<T, PUP_PTR> >, 
	public CkNoncopyable {
 public:
	typedef CkPupPtrVec<T,PUP_PTR> this_type;
	typedef CkVec< CkZeroPtr<T, PUP_PTR> > super;
	CkPupPtrVec() {}
	CkPupPtrVec(int size) :super(size) {}
	~CkPupPtrVec() {
		for (int i=0;i<length();i++)
			operator[] (i).destroy();
	}
	friend void operator|(PUP::er &p,this_type &v) {v.pup(p);}
};

///A vector of pointers-to-subclasses of a PUP::able parent
template <class T>
class CkPupAblePtrVec : public CkVec< CkZeroPtr<T, CkPupAblePtr<T> > >, 
	public CkNoncopyable {
 public:
	typedef CkPupAblePtrVec<T> this_type;
	typedef CkVec< CkZeroPtr<T, CkPupAblePtr<T> > > super;
	CkPupAblePtrVec() {}
	CkPupAblePtrVec(int size) :super(size) {}
	~CkPupAblePtrVec() {
		for (int i=0;i<length();i++)
			operator[] (i).destroy();
	}
	friend void operator|(PUP::er &p,this_type &v) {v.pup(p);}
};


#endif
