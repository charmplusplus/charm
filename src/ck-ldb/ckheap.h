/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _HEAP_H_
#define _HEAP_H_

class InfoRecord;

class heapRecord
{ public:
  short deleted; // boolean
  InfoRecord *info;
};

class heapIterator{
public:
  int next;
};

class minHeap
{
private:
  heapRecord *h;
  int count;
  int size;
  void swap(int i, int j) {
    heapRecord temp = h[i];
    h[i] = h[j];
    h[j] = temp;
  }
  
public:
  minHeap(int size);
  ~minHeap();
  int numElements() { return count; }
  int insert(InfoRecord *);
  InfoRecord *deleteMin();
  InfoRecord *iterator(heapIterator *);
  InfoRecord *next(heapIterator *);
  void update(InfoRecord *);
private:
  int least(int a, int b, int c);
  void update(int index);
};

class maxHeap
{
private:
  heapRecord *h;
  int count;
  int size;

  void swap(int i, int j) {
    heapRecord temp = h[i];
    h[i] = h[j];
    h[j] = temp;
  }
  
public:  
  maxHeap(int size);
  ~maxHeap();
  int numElements();
  int insert(InfoRecord *);
  InfoRecord *deleteMax();
  InfoRecord *iterator(heapIterator *);
  InfoRecord *next(heapIterator *);
};


#endif /* _HEAP_H_ */

/*@}*/
