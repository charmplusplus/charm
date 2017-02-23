#ifdef __APPLE__

#include <cstdlib>
#include "converse.h"

// clang on darwin requires explicit overriding of new and delete functions.

void* operator new(std::size_t bytes)
{
  return malloc(bytes);
}

void operator delete(void* ptr)
{
  free(ptr);
}

#endif
