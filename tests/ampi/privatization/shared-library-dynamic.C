
#include "test-cxx.h"

#if defined test_dynamiclib

test_thread_local int extern_global_sharedlibrary_dynamic;

#if defined test_staticvars
test_thread_local static int static_global_sharedlibrary_dynamic;
extern "C"
int * get_static_global_sharedlibrary_dynamic()
{
  return &static_global_sharedlibrary_dynamic;
}

extern "C"
int * get_static_local_sharedlibrary_dynamic()
{
  test_thread_local static int static_local_sharedlibrary_dynamic;
  return &static_local_sharedlibrary_dynamic;
}
#endif

#endif
