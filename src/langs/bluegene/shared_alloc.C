#include <vector>
#include <utility>
#include <cstdlib>

#include "shared-alloc.h"

typedef std::pair<void*, int> Alloc;

static std::vector<Alloc> allocs;

void *shalloc(int i, size_t s)
{
  // Make space to record at least i allocations
  if(allocs.size() <= i)
    allocs.resize(i+1, Alloc(0,0));

  // Ensure allocation i is initialized
  if(allocs[i].first == 0)
    allocs[i].first = malloc(s);

  // Increment its reference count
  allocs[i].second++;

  // Return the address
  return allocs[i].first;
}

void shfree(int i, void *p)
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
