
#include "test.h"

#if defined test_sharedlib

test_thread_local int extern_global_sharedlibrary;

#if defined test_staticvars
test_thread_local static int static_global_sharedlibrary;
int * get_static_global_sharedlibrary()
{
  return &static_global_sharedlibrary;
}

int * get_static_local_sharedlibrary()
{
  test_thread_local static int static_local_sharedlibrary;
  return &static_local_sharedlibrary;
}
#endif

#endif
