/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#include "elements.h"

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
  void swap(int i, int j) 
    {
      heapRecord temp = h[i];
      h[i] = h[j];
      h[j] = temp;
    }
  
public:
  minHeap(int size);
  int numElements();
  int insert(InfoRecord *);
  InfoRecord *deleteMin();
  InfoRecord *iterator(heapIterator *);
  InfoRecord *next(heapIterator *);
};

class maxHeap
{
private:
  heapRecord *h;
  int count;
  int size;

  void swap(int i, int j) 
    {
      heapRecord temp = h[i];
      h[i] = h[j];
      h[j] = temp;
    }
  
public:  
  maxHeap(int size);
  int numElements();
  int insert(InfoRecord *);
  InfoRecord *deleteMax();
  InfoRecord *iterator(heapIterator *);
  InfoRecord *next(heapIterator *);
};

