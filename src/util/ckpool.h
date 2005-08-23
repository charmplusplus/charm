/*******************************************************************
 * Defines three template classes for usage with memory pooling:

 * - CkPool: uses one single queue per processor as buffer.

 * - CkMultiPool: has one default queue, but more can be used. A little
                  slower due to the overhead of the extra indirection.

 * - CkPoolQueue: additional queues that can be used with CkMultiPool.
********************************************************************/

#ifndef ALIGN8
#define ALIGN8(x)        (int)((~7)&((x)+7))
#endif

/// Base class for multiple queues, only for internal usage
template <typename type>
class CkPoolQueueBase {
public:
  virtual void enqueue(type *p) = 0;
  virtual type *dequeue(size_t size) = 0;
};

/// A queue for CkMultiPool, templated with the type it handles, and the number
/// of objects allocated in a single chunk
template <typename type, unsigned int sz>
class CkPoolQueue : public CkPoolQueueBase<type> {
  type *first;

public:
  CkPoolQueue() { first = NULL; }
  void enqueue(type *p) {
    //printf("buffer enqueue\n");
    *(type**)p = first;
    first = p;
  }
  type *dequeue(size_t size) {
    if (first == NULL) {
      //printf("buffer dequeue - allocating %d\n",sz*ALIGN8(sizeof(type)));
      first = (type*)malloc(sz * ALIGN8(size));
      type **src;
      type *dest;
      for (unsigned int i=0; i<sz-1; ++i) {
	src = (type**)(((char*)first)+i*ALIGN8(size));
	dest = (type*)(((char*)first)+(i+1)*ALIGN8(size));
	//printf("debug: %p %p %p\n",first,src,dest);
	*src = dest;
      }
      src = (type**)(((char*)first)+(sz-1)*ALIGN8(size));
      //printf("debug: last %p %p\n",first,src);
      *src = NULL;
    }
    //printf("buffer dequeue %p %p\n",first, *(type**)first);
    type *ret = first;
    first = *(type**)first;
    return ret;
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
  static CkPoolQueue<type, sz> buffer;
public:

  void *operator new(size_t size) {
    void *ret = buffer.dequeue(size);
    //printf(" - pool operator new %p (size=%d)\n",ret,size);
    return ret;
  }
  void operator delete(void *p, size_t size) {
    //printf(" - pool operator delete %p\n",p);
    buffer.enqueue((type*)p);
  }
  friend class CkMultiPool<type>;
};

template <typename type, unsigned int sz>
CkPoolQueue<type,sz> CkPool<type,sz>::buffer;

/// CkMultiPool allows the user to have both a default queue, and specific
/// queues from which to allocate. This should be more useful when deletion of
/// queues and/or purge will be implemented. There is a pointer saved for each
/// element allocated, so the correct queue will be used during deallocation.
template <typename type>
class CkMultiPool {
public:
  void *operator new(size_t sz) {
    type *ret = CkPool<type>::buffer.dequeue(sz+sizeof(CkPoolQueueBase<type>*));
    CkPoolQueueBase<type> **bufferP = (CkPoolQueueBase<type> **)(ALIGN8((int)((char*)ret)+sz+sizeof(CkPoolQueueBase<type>*)) - sizeof(CkPoolQueueBase<type>*));
    *bufferP = &CkPool<type>::buffer;
    //printf(" - pool operator new default %p\n",bufferP);
    return ret;
  }
  void *operator new(size_t sz, CkPoolQueueBase<type> *buf) {
    type *ret = buf->dequeue(sz+sizeof(CkPoolQueueBase<type>*));
    CkPoolQueueBase<type> **bufferP = (CkPoolQueueBase<type> **)(ALIGN8((int)((char*)ret)+sz+sizeof(CkPoolQueueBase<type>*)) - sizeof(CkPoolQueueBase<type>*));
    *bufferP = buf;
    //printf(" - pool operator new with buffer %p\n",bufferP);
    return ret;
  }
  void operator delete(void *p, size_t sz) {
    CkPoolQueueBase<type> **bufferP = (CkPoolQueueBase<type> **)(ALIGN8((int)((char*)p)+sz+sizeof(CkPoolQueueBase<type>*)) - sizeof(CkPoolQueueBase<type>*));
    //printf(" - pool operator delete %p\n",bufferP);
    (*bufferP)->enqueue((type*)p);
  }
};
