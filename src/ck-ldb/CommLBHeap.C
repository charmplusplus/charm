/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "charm++.h"
#include "CommLBHeap.h"

ObjectHeap::ObjectHeap(int sz) {
  size = sz;
  h = new hRecord[sz];
  count = 0;
}

int ObjectHeap::numElements() {
  return count;
}

int ObjectHeap::insert(ObjectRecord *x) {
  h[count].info = x;
  h[count].deleted  = 0;
  int current = count;
  count++;

  if (count >= size) {
    CkPrintf("Heap overflow. \n"); 
    return -1;}

  int parent = (current - 1)/2;
  while (current != 0)
    {
      if (h[current].info->val > h[parent].info->val)
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

ObjectRecord *ObjectHeap::deleteMax() {
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
	  if (h[c1].info->val > h[c2].info->val)
	    best = c1;
	  else
	    best = c2;
	}
      if (h[best].info->val > h[current].info->val)
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

ObjectRecord *ObjectHeap::iterator(hIterator *iter) {
  iter->next = 1;
  if (count == 0)
    return 0;
  return h[0].info;
}

ObjectRecord *ObjectHeap::next(hIterator *iter) {
  if (iter->next >= count)
    return 0;
  iter->next += 1;
  return h[iter->next - 1].info;
}

/*@}*/
