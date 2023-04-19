
#include "test.h"

#if defined test_sharedlib

#if defined test_globalvars
int extern_global_shared;

#if defined test_staticvars
static int static_global_shared;
int * get_static_global_shared()
{
  return &static_global_shared;
}

int * get_scoped_global_shared()
{
  static int scoped_global_shared;
  return &scoped_global_shared;
}
#endif
#endif

#if defined test_threadlocalvars
THREAD_LOCAL int extern_threadlocal_shared;

#if defined test_staticvars
static THREAD_LOCAL int static_threadlocal_shared;
int * get_static_threadlocal_shared()
{
  return &static_threadlocal_shared;
}

int * get_scoped_threadlocal_shared()
{
  static THREAD_LOCAL int scoped_threadlocal_shared;
  return &scoped_threadlocal_shared;
}
#endif
#endif

#endif
