
#include "test.h"

#if defined test_globalvars
int extern_global_static;

#if defined test_staticvars
static int static_global_static;
int * get_static_global_static()
{
  return &static_global_static;
}

int * get_scoped_global_static()
{
  static int scoped_global_static;
  return &scoped_global_static;
}
#endif
#endif

#if defined test_threadlocalvars
thread_local int extern_threadlocal_static;

#if defined test_staticvars
thread_local static int static_threadlocal_static;
int * get_static_threadlocal_static()
{
  return &static_threadlocal_static;
}

int * get_scoped_threadlocal_static()
{
  thread_local static int scoped_threadlocal_static;
  return &scoped_threadlocal_static;
}
#endif
#endif
