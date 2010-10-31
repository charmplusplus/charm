#include <vector>
#include <utility>
#include <cstdlib>

#include "shared-alloc.h"

typedef std::pair<void*, int> Alloc;

static std::vector<Alloc> allocs;

void *shalloc(size_t s, int i)
{
  // Make space to record at least i allocations
  if(allocs.size() <= i)
    allocs.resize(i+1, Alloc((void*)0,0));

  // Ensure allocation i is initialized
  if(allocs[i].first == 0)
    allocs[i].first = malloc(s);

  // Increment its reference count
  allocs[i].second++;

  // Return the address
  return allocs[i].first;
}

void shfree(void *p, int i)
{
  // Check that pointer matches
  //CkAssert(p == allocs[i].first);
	    
  // Decrement refcount
  allocs[i].second--;

  // Free if 0
  if (allocs[i].second == 0)
    {
      free(allocs[i].first);
      allocs[i].first = 0;
    }
}


// Note that these are liable to crash and burn in the face of alignment 
// requirements
void* operator new(size_t sz, int i)
{ return shalloc(sz, i); }
void operator delete(void *p, int i)
{ shfree(p, i); }
void* operator new[](size_t sz, int i)
{ return shalloc(sz, i); }
void operator delete[](void *p, int i)
{ shfree(p, i); }
