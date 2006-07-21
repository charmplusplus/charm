/*******************************************************************
 * Defines three template classes for usage with memory pooling:

 * - CkPool: uses one single queue per processor as buffer.

 * - CkMultiPool: has one default queue, but more can be used. A little
                  slower due to the overhead of the extra indirection.

 * - CkPoolQueue: additional queues that can be used with CkMultiPool.
********************************************************************/

#include "charm.h"

/// A queue for CkMultiPool, templated with the type it handles, and the number
/// of objects allocated in a single chunk
template <typename type>
class CkPoolQueue {
  type *first;
  int sz;
  int classSize;
  void *allocations;

  CkPoolQueue() {} // private, not usable
public:
  CkPoolQueue(int _sz) : first(NULL), sz(_sz), allocations(NULL) { CkAssert(_sz > 0); }
  void enqueue(type *p) {
    //printf("buffer enqueue\n");
    *(type**)p = first;
    first = p;
  }
  type *dequeue(size_t size) {
    if (first == NULL) {
      classSize = size;
      //printf("buffer dequeue - allocating %d\n",sz*ALIGN8(sizeof(type)));
      first = (type*)malloc(sz * ALIGN8(size) + sizeof(void*));
      type **src;
      type *dest;
      for (int i=0; i<sz-1; ++i) {
	src = (type**)(((char*)first)+i*ALIGN8(size));
	dest = (type*)(((char*)first)+(i+1)*ALIGN8(size));
	//printf("debug: %p %p %p\n",first,src,dest);
	*src = dest;
      }
      src = (type**)(((char*)first)+(sz-1)*ALIGN8(size));
      //printf("debug: last %p %p\n",first,src);
      *src = NULL;

      void **nextAlloc = (void**)(((char*)first) + sz*ALIGN8(size));
      *nextAlloc = allocations;
      allocations = (void*)first;
    }
    //printf("buffer dequeue %p %p\n",first, *(type**)first);
    type *ret = first;
    first = *(type**)first;
    return ret;
  }
  void destroyAll() {
    void *next;
    first = NULL;
    while (allocations != NULL) {
      next = *(void**)(((char*)allocations) + sz*ALIGN8(classSize));
      free(allocations);
      allocations = next;
    }
  }
};

// forward declaration
template <typename type> class CkMultiPool;

/// CkPool has one single static queue per processor that maintains the already
/// allocated buffers. Templated with the type and the number of objects to be
/// allocated in chunk.
/// The typical usage would be to inherit from this class like:
/// class MyClass : public CkPool<MyClass, 32>
template <typename type, unsigned int sz = 16>
class CkPool {
  static CkPoolQueue<type> buffer;
public:

  void *operator new(size_t size) {
    void *ret = buffer.dequeue(size);
    //printf(" - pool operator new %p (size=%d)\n",ret,size);
    return ret;
  }
  void *operator new(size_t size, void *p) {
    return p;
  }
  void operator delete(void *p, size_t size) {
    //printf(" - pool operator delete %p\n",p);
    buffer.enqueue((type*)p);
  }
  friend class CkMultiPool<type>;
  static void destroyAll();
};

template <typename type, unsigned int sz>
CkPoolQueue<type> CkPool<type,sz>::buffer = CkPoolQueue<type>(sz);

template <typename type, unsigned int sz>
void CkPool<type,sz>::destroyAll() {
  //printf(" - pool destroying all\n");
  buffer.destroyAll();
}

/// CkMultiPool allows the user to have both a default queue, and specific
/// queues from which to allocate. This should be more useful when deletion of
/// queues and/or purge will be implemented. There is a pointer saved for each
/// element allocated, so the correct queue will be used during deallocation.
template <typename type>
class CkMultiPool {
public:
  void *operator new(size_t sz) {
    type *ret = CkPool<type>::buffer.dequeue(sz+sizeof(CkPoolQueue<type>*));
    CkPoolQueue<type> **bufferP = (CkPoolQueue<type> **)(ALIGN8((int)((char*)ret)+sz+sizeof(CkPoolQueue<type>*)) - sizeof(CkPoolQueue<type>*));
    *bufferP = &CkPool<type>::buffer;
    //printf(" - pool operator new default %p\n",bufferP);
    return ret;
  }
  void *operator new(size_t sz, CkPoolQueue<type> *buf) {
    type *ret = buf->dequeue(sz+sizeof(CkPoolQueue<type>*));
    CkPoolQueue<type> **bufferP = (CkPoolQueue<type> **)(ALIGN8((int)((char*)ret)+sz+sizeof(CkPoolQueue<type>*)) - sizeof(CkPoolQueue<type>*));
    *bufferP = buf;
    //printf(" - pool operator new with buffer %p\n",bufferP);
    return ret;
  }
  void *operator new(size_t size, void *p) {
    return p;
  }
  void operator delete(void *p, size_t sz) {
    CkPoolQueue<type> **bufferP = (CkPoolQueue<type> **)(ALIGN8((int)((char*)p)+sz+sizeof(CkPoolQueue<type>*)) - sizeof(CkPoolQueue<type>*));
    //printf(" - pool operator delete %p\n",bufferP);
    (*bufferP)->enqueue((type*)p);
  }
};
