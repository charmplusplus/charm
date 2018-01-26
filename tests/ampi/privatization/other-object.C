
#include "test.h"

test_thread_local int extern_global_otherobject;

#if defined test_staticvars
test_thread_local static int static_global_otherobject;
int * get_static_global_otherobject()
{
  return &static_global_otherobject;
}

int * get_static_local_otherobject()
{
  test_thread_local static int static_local_otherobject;
  return &static_local_otherobject;
}
#endif
