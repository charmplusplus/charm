// Global Variable Privatization Test - C++

#include <stdio.h>
#include "mpi.h"
#include "test.h"
#include "test-cxx.h"

#if defined test_dynamiclib
# include <dlfcn.h>
#endif


test_thread_local extern int extern_global_sameobject;
test_thread_local int extern_global_sameobject;
static int * get_extern_global_sameobject()
{
  return &extern_global_sameobject;
}

#if defined test_staticvars
test_thread_local static int static_global_sameobject;
static int * get_static_global_sameobject()
{
  return &static_global_sameobject;
}

static int * get_static_local_sameobject()
{
  test_thread_local static int static_local_sameobject;
  return &static_local_sameobject;
}
#endif

static int * get_extern_global_otherobject()
{
  return &extern_global_otherobject;
}

static int * get_extern_global_staticlibrary()
{
  return &extern_global_staticlibrary;
}

#if defined test_sharedlib
static int * get_extern_global_sharedlibrary()
{
  return &extern_global_sharedlibrary;
}
#endif


void perform_test_batch(int & failed, int & rank, int & my_wth)
{
  if (rank == 0) printf("Beginning round of testing.\n");

#if defined test_dynamiclib
  int * extern_global_sharedlibrary_dynamic_ptr = nullptr;
#if defined test_staticvars
  int_ptr_accessor get_static_global_sharedlibrary_dynamic_ptr = nullptr;
  int_ptr_accessor get_static_local_sharedlibrary_dynamic_ptr = nullptr;
#endif
  void * dynamiclib = dlopen("libcxx-" privatization_method_str "-shared-library-dynamic.so", RTLD_NOW);
  if (!dynamiclib)
  {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
  }
  else
  {
    extern_global_sharedlibrary_dynamic_ptr = (int *)dlsym(dynamiclib, "extern_global_sharedlibrary_dynamic");
    if (!extern_global_sharedlibrary_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#if defined test_staticvars
    get_static_global_sharedlibrary_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_static_global_sharedlibrary_dynamic");
    if (!get_static_global_sharedlibrary_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
    get_static_local_sharedlibrary_dynamic_ptr = (int_ptr_accessor)dlsym(dynamiclib, "get_static_local_sharedlibrary_dynamic");
    if (!get_static_local_sharedlibrary_dynamic_ptr)
      fprintf(stderr, "dlsym failed: %s\n", dlerror());
#endif
  }
#endif


  if (rank == 0) printf("Testing: extern global, in same object\n");
  test_privatization(failed, rank, my_wth, *get_extern_global_sameobject());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in same object\n");
  test_privatization(failed, rank, my_wth, *get_static_global_sameobject());
  if (rank == 0) printf("Testing: static local, in same object\n");
  test_privatization(failed, rank, my_wth, *get_static_local_sameobject());
#endif

  if (rank == 0) printf("Testing: extern global, in other object\n");
  test_privatization(failed, rank, my_wth, *get_extern_global_otherobject());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in other object\n");
  test_privatization(failed, rank, my_wth, *get_static_global_otherobject());
  if (rank == 0) printf("Testing: static local, in other object\n");
  test_privatization(failed, rank, my_wth, *get_static_local_otherobject());
#endif

  if (rank == 0) printf("Testing: extern global, in static library\n");
  test_privatization(failed, rank, my_wth, *get_extern_global_staticlibrary());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in static library\n");
  test_privatization(failed, rank, my_wth, *get_static_global_staticlibrary());
  if (rank == 0) printf("Testing: static local, in static library\n");
  test_privatization(failed, rank, my_wth, *get_static_local_staticlibrary());
#endif

#if defined test_sharedlib
  if (rank == 0) printf("Testing: extern global, in shared library\n");
  test_privatization(failed, rank, my_wth, *get_extern_global_sharedlibrary());
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in shared library\n");
  test_privatization(failed, rank, my_wth, *get_static_global_sharedlibrary());
  if (rank == 0) printf("Testing: static local, in shared library\n");
  test_privatization(failed, rank, my_wth, *get_static_local_sharedlibrary());
#endif
#endif

#if defined test_dynamiclib
  if (rank == 0) printf("Testing: extern global, in shared library, linked dynamically\n");
  if (extern_global_sharedlibrary_dynamic_ptr)
    test_privatization(failed, rank, my_wth, *extern_global_sharedlibrary_dynamic_ptr);
  else
    if (rank == 0) printf(result_indent "Skipped.\n");
#if defined test_staticvars
  if (rank == 0) printf("Testing: static global, in shared library, linked dynamically\n");
  if (get_static_global_sharedlibrary_dynamic_ptr)
    test_privatization(failed, rank, my_wth, *get_static_global_sharedlibrary_dynamic_ptr());
  else
    if (rank == 0) printf(result_indent "Skipped.\n");
  if (rank == 0) printf("Testing: static local, in shared library, linked dynamically\n");
  if (get_static_local_sharedlibrary_dynamic_ptr)
    test_privatization(failed, rank, my_wth, *get_static_local_sharedlibrary_dynamic_ptr());
  else
    if (rank == 0) printf(result_indent "Skipped.\n");
#endif
#endif

#if defined test_dynamiclib
  dlclose(dynamiclib);
#endif

  if (rank == 0) printf("Round of testing complete.\n");
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  privatization_test_framework();

  MPI_Finalize();

  return 0;
}
