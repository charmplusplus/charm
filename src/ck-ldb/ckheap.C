/**
 * \addtogroup CkLdb
*/
/*@{*/

class minheap;
class maxHeap;

#include "elements.h"
#include "ckheap.h"

// Heap of pointers. The field to be compared is:

minHeap::minHeap(int nsize)
{
  size = nsize;
  h = new heapRecord[size];
  count = 0;
}

minHeap::~minHeap()
{
  delete [] h;
}

int minHeap::insert(InfoRecord *x)
{
  int current;

  if (count < size) {
    h[count].info = x;
    h[count].deleted = 0;
    current = count;
    count++;
  } else {
    printf("minHeap overflow. \n") ; 
    return -1;
  }

  int parent = (current - 1)/2;
  while (current != 0)
    {
      if (h[current].info->load < h[parent].info->load)
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

InfoRecord *minHeap::deleteMin()
{
  if (count == 0) return 0;

  InfoRecord *tmp = h[0].info;
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
	  if (h[c1].info->load < h[c2].info->load)
	    best = c1;
	  else
	    best = c2;
	}
      if (h[best].info->load < h[current].info->load)
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

InfoRecord *minHeap::iterator(heapIterator *iter){
  iter->next = 1;
  if (count == 0)
    return 0;
  return h[0].info;
}
InfoRecord *minHeap::next(heapIterator *iter){
  if (iter->next >= count)
    return 0;
  iter->next += 1;
  return h[iter->next - 1].info;
}

int minHeap::least(int a, int b, int c){
    int smaller;
                                                                                
    if(h[a].info->load < h[b].info->load)
      smaller=a;
    else
      smaller=b;
                                                                                
    if(h[smaller].info->load < h[c].info->load)
      return smaller;
    else
      return c;
}

void minHeap::update(InfoRecord *x) {
    // find index
    // TODO:  OPTIMIZE it!
    int index;
    for (index=0; index<numElements(); index++) 
      if (x == h[index].info) break;
    if (index == numElements()) {
      CmiAbort("minHeap: update a non-existent element!\n");
    }
    update(index);
}

void minHeap::update(int index) {
    int parent = (index-1)/2;
                                                                                
    if((index != 0) && h[index].info->load < h[parent].info->load) {
      swap(parent,index);
      update(parent);
    }
                                                                                
    int c1 = 2*index+1;
    int c2 = 2*index+2;
                                                                                
    if(c2<numElements()){
      int smaller = least(index,c1,c2);
      if(smaller != index){
        swap(smaller,index);
        update(smaller);
        return;
      }
    }
    if(c1<numElements() && h[c1].info->load < h[index].info->load) {
      swap(c1,index);
      update(c1);
      return;
    }
}

//*****************


maxHeap::maxHeap(int nsize)
{
  size = nsize;
  h = new heapRecord[size];
  count = 0;
}

maxHeap::~maxHeap()
{
  delete [] h;
}

int maxHeap::numElements()
{
  return count;
}

int maxHeap::insert(InfoRecord *x)
{
  int current;

  if (count < size) {
    h[count].info = x;
    h[count].deleted  = 0;
    current = count;
    count++;
  } else {
    printf("maxHeap overflow. \n"); 
    return -1;
  }

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

InfoRecord *maxHeap::deleteMax()
{
  if (count == 0) return 0;
  InfoRecord *tmp = h[0].info;
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


InfoRecord *maxHeap::iterator(heapIterator *iter){
  iter->next = 1;
  if (count == 0)
    return 0;
  return h[0].info;
}

InfoRecord *maxHeap::next(heapIterator *iter){
  if (iter->next >= count)
    return 0;
  iter->next += 1;
  return h[iter->next - 1].info;
}

/*@}*/
