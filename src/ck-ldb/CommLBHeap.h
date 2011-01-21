/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef __COMM_LB_HEAP
#define __COMM_LB_HEAP

class ObjectRecord {
 public:
  double val;	// value on which the heap is built
  int pe;
  int pos; 
  int id; // should replace other Ids.
};

class hRecord {
 public:
  short deleted; // boolean
  ObjectRecord *info;
};

class hIterator {
 public:
  int next;
};

class ObjectHeap {
 private:
  hRecord *h;
  int count;
  int size;

  void swap(int i, int j) {
    hRecord temp = h[i];
    h[i] = h[j];
    h[j] = temp;
  }
  
 public:
  ObjectHeap(int size);
  ~ObjectHeap()  { delete [] h; }
  int numElements();
  int insert(ObjectRecord *);
  ObjectRecord *deleteMax();
  ObjectRecord *iterator(hIterator *);
  ObjectRecord *next(hIterator *);
};

#endif

/*@}*/
