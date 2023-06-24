
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
THREAD_LOCAL int extern_threadlocal_static;

#if defined test_staticvars
static THREAD_LOCAL int static_threadlocal_static;
int * get_static_threadlocal_static()
{
  return &static_threadlocal_static;
}

int * get_scoped_threadlocal_static()
{
  static THREAD_LOCAL int scoped_threadlocal_static;
  return &scoped_threadlocal_static;
}
#endif
#endif
