/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef __COMM_LB_HEAP
#define __COMM_LB_HEAP

class ObjectRecord{
 public:
  double load;
  int pe;
  int pos; 
  int id; // should replace other Ids.
};

class hRecord
{ public:
  short deleted; // boolean
  ObjectRecord *info;
};

class hIterator{
public:
  int next;
};

class ObjectHeap
{
private:
  hRecord *h;
  int count;
  int size;

  void swap(int i, int j) 
    {
      hRecord temp = h[i];
      h[i] = h[j];
      h[j] = temp;
    }
  
public:  
  ObjectHeap(int size);
  int numElements();
  int insert(ObjectRecord *);
  ObjectRecord *deleteMax();
  ObjectRecord *iterator(hIterator *);
  ObjectRecord *next(hIterator *);
};

#endif

/*@}*/
