/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#include "CommLBHeap.h"

#include "iostream.h"
// Heap of pointers. The field to be compared is:

ObjectHeap::ObjectHeap(int size)
{
  this->size = size;
  h = new hRecord[size];
  count = 0;
}

int ObjectHeap::numElements()
{
  return count;
}

int ObjectHeap::insert(InfoRecord *x)
{
  h[count].info = x;
  h[count].deleted  = 0;
  int current = count;
  count++;

  if (count >= size) {
    cout << "Heap overflow. \n" ; 
    return -1;}

  int parent = (current - 1)/2;
  while (current != 0)
    {
      if (h[current].info->load > h[parent].info->load)
	{
	  swap(current, parent);
	  current = parent;
	  parent = (current-1)/2;
	}
      else
	break;
    }
  return 0;
}

ObjectRecord *ObjectHeap::deleteMax()
{
  if (count == 0) return 0;
  ObjectRecord *tmp = h[0].info;
  int best;

  h[0] = h[count-1];
  count--;

  int current = 0;
  int c1 = 1;
  int c2 = 2;
  while (c1 < count)
    {
      if (c2 >= count)
	best = c1;
      else
	{
	  if (h[c1].info->load > h[c2].info->load)
	    best = c1;
	  else
	    best = c2;
	}
      if (h[best].info->load > h[current].info->load)
	{
	  swap(best, current);
	  current = best;
	  c1 = 2*current + 1;
	  c2 = c1 + 1;
	}
      else
	break;
    }
  return tmp;
}


ObjectRecord *ObjectHeap::iterator(hIterator *iter){
  iter->next = 1;
  if (count == 0)
    return 0;
  return h[0].info;
}

ObjectRecord *ObjectHeap::next(hIterator *iter){
  if (iter->next >= count)
    return 0;
  iter->next += 1;
  return h[iter->next - 1].info;
}
