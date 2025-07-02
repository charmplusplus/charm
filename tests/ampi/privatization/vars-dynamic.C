
#include "test.h"

#if defined test_dynamiclib

#if defined test_globalvars
extern CMI_EXPORT int extern_global_dynamic;
int extern_global_dynamic;

#if defined test_staticvars
static int static_global_dynamic;
CLINKAGE CMI_EXPORT int * get_static_global_dynamic()
{
  return &static_global_dynamic;
}

CLINKAGE CMI_EXPORT int * get_scoped_global_dynamic()
{
  static int scoped_global_dynamic;
  return &scoped_global_dynamic;
}
#endif
#endif

#if defined test_threadlocalvars
extern THREAD_LOCAL CMI_EXPORT int extern_threadlocal_dynamic;
THREAD_LOCAL int extern_threadlocal_dynamic;

#if defined test_staticvars
static THREAD_LOCAL int static_threadlocal_dynamic;
CLINKAGE CMI_EXPORT int * get_static_threadlocal_dynamic()
{
  return &static_threadlocal_dynamic;
}

CLINKAGE CMI_EXPORT int * get_scoped_threadlocal_dynamic()
{
  static THREAD_LOCAL int scoped_threadlocal_dynamic;
  return &scoped_threadlocal_dynamic;
}
#endif
#endif

#endif
