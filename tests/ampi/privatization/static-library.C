
#include "test-cxx.h"

test_thread_local int extern_global_staticlibrary;

#if defined test_staticvars
test_thread_local static int static_global_staticlibrary;
int * get_static_global_staticlibrary()
{
  return &static_global_staticlibrary;
}

int * get_static_local_staticlibrary()
{
  test_thread_local static int static_local_staticlibrary;
  return &static_local_staticlibrary;
}
#endif
